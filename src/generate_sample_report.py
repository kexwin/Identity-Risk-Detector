"""
Report Generator Utility
Runs the pipeline, extracts flagged accounts (HIGH/CRITICAL), generates explanations,
and saves the compiled JSON report to output/sample_report.json.
"""

import json
from pathlib import Path
from src.pipeline import run_pipeline
from src.explain import generate_explanation
from src.novelty import compute_twin_deviations, detect_recurrence_pattern, run_reversibility_simulation


def main():
    print("[sample_report] Initializing sample report generation...")
    results = run_pipeline()
    
    users_df = results["users_df"]
    events_df = results["events_df"]
    sod_violations = results["sod_violations"]
    twin_profiles = results["twin_profiles"]
    
    # Filter for flagged users (CRITICAL and HIGH)
    flagged = users_df[users_df["adjusted_risk_level"].isin(["CRITICAL", "HIGH"])]
    print(f"[sample_report] Flagged accounts found: {len(flagged)}")
    
    reports = []
    for uid, row in flagged.iterrows():
        user_events = events_df[events_df["user_id"] == uid]
        sod_list = sod_violations.get(uid, [])
        systems_list = row["systems_list"] if isinstance(row["systems_list"], list) else []
        reversibility = run_reversibility_simulation(uid, systems_list, user_events)
        recurrence = detect_recurrence_pattern(user_events)
        
        # Build LLM payload
        llm_payload = {
            "user_id": uid,
            "username": row["username"],
            "department": row["department"],
            "privilege_level": row["privilege_level"],
            "days_inactive": int(row["days_inactive"]),
            "system_count": int(row["system_count"]),
            "blast_radius": int(row["blast_radius"]),
            "anomaly_score": float(row["adjusted_score"]),
            "risk_level": row["adjusted_risk_level"],
            "exception_tags": row["exception_tags"],
            "sod_violations_count": len(sod_list),
            "sod_violations": sod_list,
            "high_sensitivity_export_count": int(row["high_sensitivity_export_count"]),
            "admin_op_off_hours_count": int(row["admin_op_off_hours_count"]),
            "recent_event_count": int(row["recent_event_count"]),
            "after_hours_event_ratio": float(row["after_hours_event_ratio"]),
            "confidence": float(row["confidence"]),
            "confidence_basis": row["confidence_basis"],
            "recurrence_info": recurrence,
            "is_stale_admin": int(row.get("is_stale_admin", 0)),
            "is_stale_power_user": int(row.get("is_stale_power_user", 0))
        }
        
        # Generate narrative report
        report = generate_explanation(llm_payload)
        reports.append(report)
        
    out_file = Path(__file__).parent.parent / "output" / "sample_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, "w") as f:
        json.dump(reports, f, indent=2)
        
    print(f"[sample_report] Successfully generated and wrote {len(reports)} risk reports to {out_file}")


if __name__ == "__main__":
    main()
