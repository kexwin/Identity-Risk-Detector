"""
Streamlit Dashboard
Provides an interactive, premium security analyst interface for risk monitoring, breach simulation, and feedback loops.
"""

import sys
import os
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import json
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from sklearn.decomposition import PCA

# Page configuration
st.set_page_config(
    page_title="Identity Sprawl & Privilege Abuse Detector",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Server Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# Custom dark-theme styling
st.markdown("""
<style>
    .reportview-container {
        background: #090d16;
    }
    .metric-container {
        background: linear-gradient(135deg, #161f38, #0e172a);
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(to right, #60a5fa, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .badge {
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        text-transform: uppercase;
    }
    .badge-critical { background-color: #ef4444; color: white; }
    .badge-high { background-color: #f97316; color: white; }
    .badge-medium { background-color: #eab308; color: black; }
    .badge-low { background-color: #10b981; color: white; }

    .card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# Fallback Mechanism: Import pipeline direct if API is down
@st.cache_data(show_spinner="Running risk detection pipeline...")
def run_pipeline_direct():
    from src.pipeline import run_pipeline
    return run_pipeline()


def get_data(endpoint):
    """Fetch data from FastAPI, or fall back to running the pipeline locally."""
    try:
        r = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
        if r.status_code == 200:
            return r.json(), False
    except Exception:
        pass

    # Fallback to local import execution
    data_dict = run_pipeline_direct()

    if endpoint == "/api/users":
        df = data_dict["users_df"]
        cols = [
            "user_id",
            "username",
            "email",
            "department",
            "job_title",
            "privilege_level",
            "days_inactive",
            "system_count",
            "blast_radius",
            "events_per_user",
            "anomaly_score",
            "adjusted_score",
            "risk_level",
            "adjusted_risk_level",
            "sod_violations_count",
            "exception_tags"
        ]
        available_cols = [c for c in cols if c in df.columns]
        print("Available columns:", df.columns.tolist())
        print("Using columns:", available_cols)
        return df[available_cols].reset_index().to_dict(orient="records"), True

    elif endpoint.startswith("/api/users/"):
        user_id = endpoint.split("/")[-1]
        df = data_dict["users_df"]
        if user_id not in df.index:
            return None, True

        user_row = df.loc[user_id]
        user_events = data_dict["events_df"][data_dict["events_df"]["user_id"] == user_id]

        from src.novelty import compute_twin_deviations, run_reversibility_simulation, detect_recurrence_pattern
        from src.explain import generate_explanation

        sod_list = data_dict["sod_violations"].get(user_id, [])
        twin_dev = compute_twin_deviations(user_row, data_dict["twin_profiles"])
        systems_list = user_row.get("systems_list", []) if isinstance(user_row.get("systems_list"), list) else []
        reversibility = run_reversibility_simulation(user_id, systems_list, user_events)
        recurrence = detect_recurrence_pattern(user_events)

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

        llm_report = generate_explanation(llm_payload)

        return {
            "profile": {
                "user_id": user_id,
                "username": user_row["username"],
                "email": user_row["email"],
                "department": user_row["department"],
                "job_title": user_row["job_title"],
                "privilege_level": user_row["privilege_level"],
                "days_inactive": int(user_row["days_inactive"]),
                "systems_access": user_row.get("systems_access", "|".join(user_row.get("systems_list", []) if isinstance(user_row.get("systems_list"), list) else [])),
                "hire_date": str(user_row["hire_date"]),
                "tenure_days": int(user_row["tenure_days"])
            },
            "risk_metrics": {
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
            },
            "sod_violations": sod_list,
            "digital_twin": twin_dev,
            "reversibility": reversibility,
            "recurrence": recurrence,
            "llm_report": llm_report
        }, True

    elif endpoint.startswith("/api/simulation/"):
        user_id = endpoint.split("/")[-1]
        from src.breach import simulate_user_breach
        df = data_dict["users_df"]
        return simulate_user_breach(user_id, df, data_dict["events_df"], data_dict["G"], data_dict["user_metrics"]), True

    return None, True


def submit_feedback(user_id, is_fallback):
    """Submit false positive feedback."""
    if not is_fallback:
        try:
            r = requests.post(f"{API_BASE_URL}/api/feedback", json={"user_id": user_id}, timeout=5)
            if r.status_code == 200:
                st.success(r.json().get("message", "Feedback registered!"))
                st.cache_data.clear()
                st.rerun()
                return
        except Exception as e:
            st.error(f"Failed to submit to API: {e}")

    FEEDBACK_FILE = Path(__file__).parent.parent / "data" / "feedback.json"
    data_dict = run_pipeline_direct()
    df = data_dict["users_df"]
    user_row = df.loc[user_id]

    fp_feature_cols = [
        "days_inactive", "system_count", "recent_event_count",
        "after_hours_event_ratio", "high_sensitivity_export_count",
        "admin_op_off_hours_count", "failure_rate"
    ]
    features = {col: float(user_row[col]) for col in fp_feature_cols}

    existing = []
    if FEEDBACK_FILE.exists():
        try:
            with open(FEEDBACK_FILE, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []

    if not any(item.get("user_id") == user_id for item in existing):
        existing.append({
            "user_id": user_id,
            "username": user_row["username"],
            "features": features
        })
        FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(existing, f, indent=2)

    st.success(f"Feedback saved locally for {user_id}! Pipeline re-calculated.")
    st.cache_data.clear()
    st.rerun()


# Load full users list
users_data, is_fallback = get_data("/api/users")
df_all = pd.DataFrame(users_data)

# Header title
st.markdown("##  <span class='gradient-text'>Identity Sprawl & Privilege Abuse Detection Portal</span>", unsafe_allow_html=True)
if is_fallback:
    st.info(" Dashboard running in **Local Engine Mode** (FastAPI server offline). Local calculations active.")
else:
    st.success(" Connected to **FastAPI Detection Engine**.")

# KPI scorecards
kpi_cols = st.columns(4)
total_users = len(df_all)
flagged_users = len(df_all[df_all["adjusted_risk_level"].isin(["CRITICAL", "HIGH", "MEDIUM"])])
critical_users = len(df_all[df_all["adjusted_risk_level"] == "CRITICAL"])
total_sod = df_all["sod_violations_count"].sum()

with kpi_cols[0]:
    st.markdown(f"""
    <div class='metric-container'>
        <div class='metric-title'>Total Users Scanned</div>
        <div class='metric-value'>{total_users}</div>
    </div>
    """, unsafe_allow_html=True)
with kpi_cols[1]:
    st.markdown(f"""
    <div class='metric-container'>
        <div class='metric-title'>Flagged Risk Accounts</div>
        <div class='metric-value'>{flagged_users}</div>
    </div>
    """, unsafe_allow_html=True)
with kpi_cols[2]:
    st.markdown(f"""
    <div class='metric-container'>
        <div class='metric-title'>Critical Risk Accounts</div>
        <div class='metric-value' style='background: linear-gradient(to right, #f87171, #ef4444); -webkit-background-clip: text;'>{critical_users}</div>
    </div>
    """, unsafe_allow_html=True)
with kpi_cols[3]:
    st.markdown(f"""
    <div class='metric-container'>
        <div class='metric-title'>GDPR SoD Violations</div>
        <div class='metric-value' style='background: linear-gradient(to right, #fbbf24, #f59e0b); -webkit-background-clip: text;'>{total_sod}</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# Sidebar filters
st.sidebar.markdown("###  Filter Risk Queue")
risk_filter = st.sidebar.multiselect("Risk Level", options=["CRITICAL", "HIGH", "MEDIUM", "LOW"], default=["CRITICAL", "HIGH", "MEDIUM"])
dept_filter = st.sidebar.multiselect("Department", options=sorted(df_all["department"].unique()), default=sorted(df_all["department"].unique()))

# Define categorical sorting for Risk Levels
risk_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
df_all["_risk_rank"] = df_all["adjusted_risk_level"].map(risk_order)

# Apply filters and sort by risk level, then score
df_filtered = df_all[
    df_all["adjusted_risk_level"].isin(risk_filter) &
    df_all["department"].isin(dept_filter)
].sort_values(["_risk_rank", "adjusted_score"], ascending=[True, False])

# Main layout tabs
tab_queue, tab_graph, tab_clusters, tab_compliance, tab_exec_summary = st.tabs([
    " Priority Queue",
    " Privilege Access Graph",
    " Behavioural Clusters",
    " Compliance Status",
    " Executive Summary"
])

with tab_queue:
    st.markdown("### Risk Queue")
    display_cols = ["user_id", "username", "department", "privilege_level", "days_inactive", "system_count", "adjusted_score", "adjusted_risk_level", "sod_violations_count"]

    df_display = df_filtered[display_cols].copy()
    df_display.columns = ["User ID", "Username", "Department", "Privilege", "Days Inactive", "Systems", "Risk Score", "Risk Level", "SoD Violations"]

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    st.write("---")
    st.markdown("###  Deep-Dive User Risk Investigator")

    target_users = df_filtered["user_id"] + " (" + df_filtered["username"] + " - " + df_filtered["adjusted_risk_level"] + ")"
    selected_option = st.selectbox("Select user to investigate", options=target_users)

    if selected_option:
        selected_uid = selected_option.split(" ")[0]

        detail, _ = get_data(f"/api/users/{selected_uid}")

        if detail:
            profile = detail["profile"]
            risk = detail["risk_metrics"]
            llm_rep = detail["llm_report"]

            det_cols = st.columns([1, 1])

            with det_cols[0]:
                st.markdown(f"#### Account Details: {profile['username']}")
                st.markdown(f"""
                - **User ID**: `{profile['user_id']}`
                - **Email**: `{profile['email']}`
                - **Role / Job Title**: `{profile['job_title']}`
                - **Privilege Level**: `{profile['privilege_level']}`
                - **Department**: `{profile['department']}`
                - **Tenure**: `{profile['tenure_days']} days`
                - **Systems Access**: `{profile['systems_access']}`
                """)

                st.markdown("#### Detection Context & Exceptions")
                tags = risk["exception_tags"]
                if tags:
                    for t in tags:
                        st.markdown(f" **{t}**")
                else:
                    st.write("None")

                if "SIMILAR_TO_KNOWN_FP" in tags:
                    st.warning(" User has already been adjusted by the FP Feedback Loop.")
                else:
                    if st.button("Mark as False Positive", key=f"fp_{selected_uid}"):
                        submit_feedback(selected_uid, is_fallback)

            with det_cols[1]:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk["adjusted_score"],
                    title={'text': "Adjusted Risk Score"},
                    domain={'x': [0, 1], 'y': [0, 1]},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#3b82f6"},
                        'steps': [
                            {'range': [0, 70], 'color': "#10b981"},
                            {'range': [70, 84], 'color': "#eab308"},
                            {'range': [84, 95], 'color': "#f97316"},
                            {'range': [95, 100], 'color': "#ef4444"}
                        ]
                    }
                ))
                fig.update_layout(height=250, margin=dict(t=30, b=0, l=30, r=30), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"**Risk Severity**: {risk['adjusted_risk_level']}")
                st.markdown(f"**Confidence Score**: {risk['confidence']} (Basis: *{risk['confidence_basis']}*)")

            st.write("---")
            st.markdown("####  SOC Analyst Narrative (LLM Generated)")

            st.info(f"**Confidence**: {llm_rep.get('confidence')} | **Basis**: {llm_rep.get('confidence_basis')}")

            st.markdown("##### Key Findings")
            for f in llm_rep.get("findings", []):
                sev_color = "[HIGH]" if f['severity'] == "HIGH" or f['severity'] == "CRITICAL" else "[MED]"
                st.markdown(f"{sev_color} **{f['finding']}** ({f['severity']})")
                st.markdown(f"> {f['details']}")
                st.markdown(f"*Recommendation*: {f['recommendation']}")

            st.markdown("##### Suggested Mitigation Steps")
            for sa in llm_rep.get("suggested_actions", []):
                st.markdown(f"- [ ] {sa}")

            st.markdown(f" **Next Escalation Path**: `{llm_rep.get('next_escalation')}`")

            st.write("---")
            st.markdown("####  Breach Impact Simulation")
            sim_button = st.button("Simulate Compromise", key=f"sim_{selected_uid}")

            if sim_button:
                sim, _ = get_data(f"/api/simulation/{selected_uid}")
                if sim:
                    sim_cols = st.columns(2)
                    with sim_cols[0]:
                        st.metric("Blast Radius Score", sim["blast_radius_score"], f"{sim['blast_radius_deviation']:.1f} vs peer median")
                        st.write("**Exposed Systems**:")
                        for sys in sim["systems_exposed"]:
                            is_sens = sys in sim["exposed_sensitive_systems"]
                            sens_badge = "High-sensitivity" if is_sens else "Normal"
                            st.markdown(f"- **{sys}** ({sens_badge})")
                    with sim_cols[1]:
                        st.markdown("**Compliance & Governance Risks (GDPR Art 5(1)(f) mapping)**")
                        risks = sim["compliance_risks"]
                        if risks:
                            for r in risks:
                                st.error(f"**{r['risk_type']}** ({r['framework']})\n{r['description']}")
                        else:
                            st.success("No critical compliance risk flags triggered for exposed systems.")

                    st.write("**Access Minimization Reversibility**")

                    reversibility = sim["reversibility_analysis"]

                    if reversibility:
                        for r_info in reversibility:
                            system_name = r_info.get("system", "Unknown System")

                            if r_info.get("safe_to_revoke", False):
                                st.success(
                                    f"[SAFE] {system_name}: "
                                    f"{r_info.get('remediation_recommendation', 'Safe to revoke')}"
                                )
                            else:
                                st.warning(
                                    f"[ACTIVE] {system_name}: "
                                    f"{r_info.get('remediation_recommendation', 'Still actively used')}"
                                )

                            if r_info.get("reasoning"):
                                st.caption(r_info["reasoning"])
                    else:
                        st.info("No reversibility analysis available.")

            st.write("---")
            st.markdown("####  Identity Digital Twin Analysis")
            twin = detail["digital_twin"]
            twin_cols = st.columns(3)
            with twin_cols[0]:
                st.metric("System Count Deviation", f"{twin['system_count_deviation']:+.1f}", f"Expected: {twin['expected_system_count']}")
            with twin_cols[1]:
                st.metric("Privilege Deviation", f"{twin['privilege_level_deviation']:+.1f}", "vs typical department mode")
            with twin_cols[2]:
                st.metric("Department Peer Group Count", f"n={twin['peer_count']}", f"Department: {twin['department_twin']}")

            st.write("---")
            st.markdown("####  Separation of Duties (SoD) Violations (S5)")
            sod_list = detail["sod_violations"]
            if sod_list:
                for s in sod_list:
                    st.error(f"**{s['name']}** ({s['severity']}) - {s['gdpr_ref']}\n{s['description']}")
            else:
                st.success("No Separation of Duties violations detected for this account.")

with tab_graph:
    st.markdown("### Interactive Privilege Access Graph")
    st.write("Nodes represent users (circles) and systems (triangles). Users are colored by risk level (Red: Critical, Orange: High, Yellow: Medium, Teal: Low). Systems are colored Blue.")

    if not is_fallback:
        st.components.v1.iframe(src=f"{API_BASE_URL}/api/graph", height=650)
    else:
        graph_path = Path("dashboard/privilege_graph.html")
        if graph_path.exists():
            html_content = graph_path.read_text(encoding="utf-8")
            components.html(html_content, height=650, scrolling=True)
        else:
            st.warning("Graph HTML file not generated yet. Select a user in the Priority Queue to run the pipeline.")

with tab_clusters:
    st.markdown("### Behavioral Clustering (Bonus S3)")
    st.write("KMeans clustering applied to flagged user accounts (CRITICAL, HIGH, MEDIUM) to group them into distinct risk behavioral patterns.")

    from src.cluster import cluster_users, get_cluster_summary
    from sklearn.preprocessing import StandardScaler

    data_dict = run_pipeline_direct()
    users_df = data_dict["users_df"]

    flagged_users_df = users_df[
        users_df["adjusted_risk_level"].isin(["CRITICAL", "HIGH", "MEDIUM"])
    ].copy()

    if not flagged_users_df.empty:
        feature_df = data_dict.get("feature_df", flagged_users_df)

        cluster_df = cluster_users(
            feature_df=feature_df,
            scored_df=flagged_users_df,
            users_df=users_df,
            n_clusters=5
        )
        cluster_info = get_cluster_summary()

        if cluster_df is not None and not cluster_df.empty:
            flagged_clustered = flagged_users_df.merge(cluster_df, on="user_id", how="left")
        else:
            flagged_clustered = flagged_users_df

        user_cols = [
            "days_inactive", "system_count", "system_count_vs_dept_median",
            "has_sensitive_system", "recent_event_count", "after_hours_event_ratio",
            "high_sensitivity_export_count", "admin_op_off_hours_count", "failure_rate",
            "tenure_days", "flagged_event_count"
        ]
        available_cols = [c for c in user_cols if c in flagged_clustered.columns]

        if len(available_cols) >= 2:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(flagged_clustered[available_cols])

            pca = PCA(n_components=2)
            pcs = pca.fit_transform(scaled)

            flagged_clustered = flagged_clustered.copy()
            flagged_clustered["PC1"] = pcs[:, 0]
            flagged_clustered["PC2"] = pcs[:, 1]

            if "cluster_id" in flagged_clustered.columns and cluster_info and "profiles" in cluster_info:
                flagged_clustered["Cluster Name"] = flagged_clustered["cluster_id"].map(
                    lambda cid: cluster_info["profiles"].get(cid, {}).get("cluster_label", f"Cluster {cid}")
                )
            else:
                flagged_clustered["Cluster Name"] = "Unclustered"

            hover_cols = [c for c in ["user_id", "department", "privilege_level", "adjusted_score", "adjusted_risk_level"] if c in flagged_clustered.columns]

            fig = px.scatter(
                flagged_clustered.reset_index(),
                x="PC1",
                y="PC2",
                color="Cluster Name",
                hover_name="username" if "username" in flagged_clustered.columns else None,
                hover_data=hover_cols,
                title="PCA Projection of Flagged Accounts' Feature Space",
                color_discrete_sequence=px.colors.qualitative.G10
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "white"},
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough feature columns available for PCA projection.")

        if cluster_info and "profiles" in cluster_info:
            st.write("#### Cluster Characterization Details")
            for cid, info in cluster_info["profiles"].items():
                with st.expander(f"{info.get('cluster_label', f'Cluster {cid}')} (Size: {info.get('user_count', 'N/A')} users)"):
                    st.markdown(f"""
                    - **Average Inactive Days**: `{info.get('mean_days_inactive', 'N/A')}`
                    - **Average System Access Count**: `{info.get('mean_system_count', 'N/A')}`
                    - **Mean Anomaly Score**: `{info.get('mean_anomaly_score', 'N/A')}`
                    - **Mean Total Events**: `{info.get('mean_total_events', 'N/A')}`
                    - **Dominant Privilege**: `{info.get('dominant_privilege', 'N/A')}`
                    - **Departments**: `{', '.join(info.get('departments', []))}`
                    """)
        else:
            st.info("No cluster summary available.")
    else:
        st.info("No flagged risk accounts available for clustering.")

with tab_compliance:
    st.markdown("### Compliance Mapping")
    data_dict = run_pipeline_direct()
    comp_report = data_dict.get('comp_report', {}).get('framework_summaries', {})
    if not comp_report:
        st.info("No compliance data available.")
    else:
        cols = st.columns(len(comp_report))
        for i, (framework, details) in enumerate(comp_report.items()):
            with cols[i]:
                st.markdown(f"### {details.get('framework_name', framework)}")
                st.metric("Total Violations", details.get('total_findings', 0))
                sev = details.get('severity_distribution', {})
                high_sev = sev.get('HIGH', 0) + sev.get('CRITICAL', 0)
                st.metric("High Severity", high_sev, delta="Action Required", delta_color="inverse")
                actions = details.get('remediation_actions', [])
                timeline = actions[0].get('timeline', 'N/A') if actions else 'N/A'
                st.info(f"Timeline: {timeline}")

with tab_exec_summary:
    st.markdown("### Executive Summary")
    data_dict = run_pipeline_direct()
    summary = data_dict.get('exec_summary', {})
    st.markdown(summary.get('risk_summary', 'No summary available.'))

    st.markdown("### Recommended Priorities")
    for priority in summary.get('recommended_priorities', []):
        st.info(priority)

    st.markdown("### Attack Path Analysis")
    paths = data_dict.get('attack_paths', [])
    for p in paths[:5]:
        blast = p.get('blast_radius', {})
        score = blast.get('total_exposure_score', 0)
        with st.expander(f"Path originating at {p.get('username', 'Unknown')} (Exposure Score: {score})"):
            st.write(f"**Description:** {p.get('recommendation', 'N/A')}")
            st.write(f"**Initial Systems:** {', '.join(blast.get('direct_systems', []))}")
            st.write(f"**Lateral Targets:** {', '.join(blast.get('indirect_systems', []))}")