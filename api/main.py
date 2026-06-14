"""
FastAPI Main Application
Serves as the backend for the Identity Risk Detector.
Exposes endpoints for user risk list, detailed profiles, breach simulation, privilege graph, and false positive feedback.
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

from src.pipeline import run_pipeline
from src.graph import export_graph_to_html
from src.breach import simulate_user_breach
from src.novelty import compute_twin_deviations, detect_recurrence_pattern, run_reversibility_simulation
from src.explain import generate_explanation

app = FastAPI(
    title="Identity Sprawl & Privilege Abuse Detection API",
    description="Backend API for real-time identity risk analysis and breach impact simulation",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory storage for pipeline data
DATA = {}
FEEDBACK_FILE = Path(__file__).parent.parent / "data" / "feedback.json"


@app.on_event("startup")
def startup_event():
    """Run the pipeline on startup to pre-load all features and risks."""
    global DATA
    print("[api] Running pipeline on startup...")
    DATA = run_pipeline()
    
    users_df = DATA["users_df"].copy()

    if "adjusted_risk_level" in users_df.columns:
        users_df["risk_level"] = users_df["adjusted_risk_level"]

    users_dict = users_df.to_dict(orient="index")

    export_graph_to_html(
    DATA["G"],
    "dashboard/privilege_graph.html",
    users_dict
)


def reload_pipeline():
    """Reloads the pipeline when feedback is received."""
    global DATA
    print("[api] Reloading pipeline due to state changes...")
    DATA = run_pipeline()
    
    users_df = DATA["users_df"].copy()

    if "adjusted_risk_level" in users_df.columns:
        users_df["risk_level"] = users_df["adjusted_risk_level"]

    users_dict = users_df.to_dict(orient="index")

    export_graph_to_html(
       DATA["G"],
       "dashboard/privilege_graph.html",
       users_dict
)


class FeedbackRequest(BaseModel):
    user_id: str


@app.get("/api/users")
def get_users():
    """Get the priority queue list of all users and their risk levels."""
    if not DATA:
        raise HTTPException(status_code=503, detail="Pipeline data not loaded")
    
    df = DATA["users_df"]
    # Select key columns to present in the list
    cols = [
        "username", "email", "department", "job_title", "privilege_level",
        "days_inactive", "system_count", "blast_radius", "events_per_user",
        "anomaly_score", "adjusted_score", "risk_level", "adjusted_risk_level",
        "sod_violations_count", "exception_tags"
    ]
    
    # Reset index to include user_id in the JSON output
    safe_df = df[cols].reset_index()
    fill_values = {
        "username": "",
        "email": "",
        "department": "",
        "job_title": "",
        "privilege_level": "user",
        "days_inactive": 0,
        "system_count": 0,
        "blast_radius": 0,
        "events_per_user": 0,
        "anomaly_score": 0.0,
        "adjusted_score": 0.0,
        "risk_level": "LOW",
        "adjusted_risk_level": "LOW",
        "sod_violations_count": 0,
    }
    for col, val in fill_values.items():
        if col in safe_df.columns:
            safe_df[col] = safe_df[col].fillna(val)
    if "exception_tags" in safe_df.columns:
        safe_df["exception_tags"] = safe_df["exception_tags"].apply(lambda x: x if isinstance(x, list) else [])

    users_list = safe_df.to_dict(orient="records")
    return users_list


@app.get("/api/users/{user_id}")
def get_user_detail(user_id: str):
    """Retrieve the comprehensive risk profile of a specific user."""
    if not DATA:
        raise HTTPException(status_code=503, detail="Pipeline data not loaded")
    
    df = DATA["users_df"]
    if user_id not in df.index:
        raise HTTPException(status_code=404, detail="User not found")
        
    user_row = df.loc[user_id]
    user_events = DATA["events_df"][DATA["events_df"]["user_id"] == user_id]
    
    # 1. Base details
    profile = {
        "user_id": user_id,
        "username": user_row["username"],
        "email": user_row["email"],
        "department": user_row["department"],
        "job_title": user_row["job_title"],
        "privilege_level": user_row["privilege_level"],
        "days_inactive": int(user_row["days_inactive"]),
        "systems_access": user_row.get("systems_access", str(user_row.get("systems_list", []))),
        "hire_date": str(user_row["hire_date"]),
        "tenure_days": int(user_row["tenure_days"]),
    }
    
    # 2. Risk Metrics
    risk_metrics = {
        "anomaly_score": float(user_row["anomaly_score"]),
        "adjusted_score": float(user_row["adjusted_score"]),
        "risk_level": user_row["risk_level"],
        "adjusted_risk_level": user_row["adjusted_risk_level"],
        "exception_tags": user_row["exception_tags"],
        "system_count": int(user_row["system_count"]),
        "blast_radius": int(user_row["blast_radius"]),
        "events_per_user": int(user_row["events_per_user"]),
        "recent_event_count": int(user_row["recent_event_count"]),
        "after_hours_event_ratio": float(user_row["after_hours_event_ratio"]),
        "high_sensitivity_export_count": int(user_row["high_sensitivity_export_count"]),
        "admin_op_off_hours_count": int(user_row["admin_op_off_hours_count"]),
        "failure_rate": float(user_row["failure_rate"]),
        "confidence": float(user_row["confidence"]),
        "confidence_basis": user_row["confidence_basis"]
    }
    
    # 3. Separation of Duty Violations
    sod_list = DATA["sod_violations"].get(user_id, [])
    
    # 4. Identity Digital Twin Comparison
    twin_dev = compute_twin_deviations(user_row, DATA["twin_profiles"])
    
    # 5. Reversibility simulator
    systems_raw = user_row.get("systems_list", [])

    if isinstance(systems_raw, str):
        systems_list = [
        s.strip()
        for s in systems_raw.split("|")
        if s.strip()
       ]
    elif isinstance(systems_raw, list):
        systems_list = systems_raw
    else:
        systems_list = []

    reversibility = run_reversibility_simulation(
        user_id,
        systems_list,
        user_events
)
    
    # 6. Recurrence Check
    recurrence = detect_recurrence_pattern(user_events)
    
    # 7. Package everything for the LLM Narrator input
    sod_viol_names = [v["name"] for v in sod_list]
    llm_payload = {
        "user_id": user_id,
        "username": user_row["username"],
        "department": user_row["department"],
        "privilege_level": user_row["privilege_level"],
        "days_inactive": int(user_row["days_inactive"]),
        "system_count": int(user_row["system_count"]),
        "blast_radius": int(user_row["blast_radius"]),
        "anomaly_score": float(user_row["adjusted_score"]),
        "risk_level": user_row["adjusted_risk_level"],
        "exception_tags": user_row["exception_tags"],
        "sod_violations_count": len(sod_list),
        "sod_violations": sod_list,
        "high_sensitivity_export_count": int(user_row["high_sensitivity_export_count"]),
        "admin_op_off_hours_count": int(user_row["admin_op_off_hours_count"]),
        "recent_event_count": int(user_row["recent_event_count"]),
        "after_hours_event_ratio": float(user_row["after_hours_event_ratio"]),
        "confidence": float(user_row["confidence"]),
        "confidence_basis": user_row["confidence_basis"],
        "recurrence_info": recurrence,
        "is_stale_admin": int(user_row.get("is_stale_admin", 0)),
        "is_stale_power_user": int(user_row.get("is_stale_power_user", 0))
    }
    
    # Generate LLM or template report
    llm_report = generate_explanation(llm_payload)
    
    return {
        "profile": profile,
        "risk_metrics": risk_metrics,
        "sod_violations": sod_list,
        "digital_twin": twin_dev,
        "reversibility": reversibility,
        "recurrence": recurrence,
        "llm_report": llm_report
    }


@app.get("/api/simulation/{user_id}")
def get_breach_simulation(user_id: str):
    """Run a breach impact simulation for a specific user."""
    if not DATA:
        raise HTTPException(status_code=503, detail="Pipeline data not loaded")
        
    df = DATA["users_df"]
    if user_id not in df.index:
        raise HTTPException(status_code=404, detail="User not found")
        
    sim = simulate_user_breach(
        user_id, df, DATA["events_df"], DATA["G"], DATA["user_metrics"]
    )
    return sim


@app.get("/api/graph")
def get_graph_html():
    """Returns the privilege graph HTML."""
    graph_path = Path("dashboard/privilege_graph.html")
    if not graph_path.exists():
        if DATA:
            users_dict = DATA["users_df"].to_dict(orient="index")
            export_graph_to_html(DATA["G"], str(graph_path), users_dict)
        else:
            raise HTTPException(status_code=503, detail="Graph not generated yet")
            
    return HTMLResponse(content=graph_path.read_text(encoding="utf-8"))


@app.post("/api/feedback")
def submit_fp_feedback(request: FeedbackRequest):
    """Marks a user profile as a False Positive and recalculates risks."""
    if not DATA:
        raise HTTPException(status_code=503, detail="Pipeline data not loaded")
        
    user_id = request.user_id
    df = DATA["users_df"]
    if user_id not in df.index:
        raise HTTPException(status_code=404, detail="User not found")
        
    user_row = df.loc[user_id]
    
    # Extract behavioral and static features
    fp_feature_cols = [
        "days_inactive", "system_count", "recent_event_count", 
        "after_hours_event_ratio", "high_sensitivity_export_count", 
        "admin_op_off_hours_count", "failure_rate"
    ]
    
    features = {col: float(user_row[col]) for col in fp_feature_cols}
    
    # Load existing feedback
    existing = []
    if FEEDBACK_FILE.exists():
        try:
            with open(FEEDBACK_FILE, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []
            
    # Check if already added
    if any(item.get("user_id") == user_id for item in existing):
        return {"status": "already_marked", "message": f"User {user_id} is already in the feedback loop."}
        
    # Append new feedback
    existing.append({
        "user_id": user_id,
        "username": user_row["username"],
        "features": features
    })
    
    # Create directory if needed and save
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(existing, f, indent=2)
        
    print(f"[api] Marked {user_id} as FP. Appended feature vector to {FEEDBACK_FILE}")
    
    # Hot-reload pipeline
    reload_pipeline()
    
    return {"status": "success", "message": f"User {user_id} successfully registered as False Positive. Pipeline reloaded."}