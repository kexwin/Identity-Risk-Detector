"""
Pipeline Orchestration
Coordinates all 6 stages of the detection pipeline.
"""

import os
import json
import time
import pandas as pd
import networkx as nx
from pathlib import Path
from typing import Dict, Any

from src.ingest import load_data
from src.graph import build_privilege_graph, compute_graph_metrics, detect_sod_violations as graph_detect_sod, export_graph_html
from src.features import engineer_features
from src.model import score_events, aggregate_event_flags, score_users
from src.context import apply_exceptions, generate_findings
from src.novelty import apply_novelty_features, check_reversibility, detect_recurrence, get_digital_twin_profiles, calibrate_confidence
from src.explain import generate_explanation, generate_executive_summary, generate_report
from src.cluster import cluster_users, get_cluster_summary
from src.compliance import map_compliance, generate_compliance_report
from src.attack_path import analyze_attack_paths
from src.sod import detect_sod_violations as sod_detect_sod


# Aliases for backward compatibility in tests
def load_users(data_dir: str = "data") -> pd.DataFrame:
    users_df, _ = load_data(data_dir)
    return users_df


def load_events(data_dir: str = "data") -> pd.DataFrame:
    _, events_df = load_data(data_dir)
    return events_df


def get_system_weight(system_name: str) -> int:
    weights = {"PROD_DB": 3, "Azure_AD": 2, "File_Share": 1}
    return weights.get(system_name, 1)


def build_access_graph(users_df: pd.DataFrame):
    G = build_privilege_graph(users_df)
    metrics = compute_graph_metrics(G, users_df)
    return G, metrics


def compute_features(users_df: pd.DataFrame, events_df: pd.DataFrame, G: nx.Graph, user_metrics: Dict[str, Any]) -> pd.DataFrame:
    df = engineer_features(users_df, events_df, user_metrics)
    df["cross_dept_resource_access"] = 0
    return df


def run_anomaly_scoring(features_df: pd.DataFrame, events_df: pd.DataFrame):
    users_df = load_users()
    scored_events = score_events(events_df, users_df)
    features_df = aggregate_event_flags(scored_events, features_df)
    scored_users = score_users(features_df)
    return scored_users, scored_events


def run_pipeline(data_dir: str = "data", output_dir: str = "output") -> dict:
    """Run the complete identity risk detection pipeline."""
    start_time = time.time()

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print("Starting Identity Risk Detection Pipeline...")

    # Stage 1: Ingestion
    print("\n[1/6] Ingesting Data...")
    users_df, events_df = load_data(data_dir)

    # Stage 2: Graph Construction
    print("[2/6] Building Privilege Graph...")
    G = build_privilege_graph(users_df)
    graph_metrics = compute_graph_metrics(G, users_df)
    # Graph-level SOD (list of dicts)
    graph_sod = graph_detect_sod(users_df)
    # Stage 5/Compliance SOD (dict and counts)
    sod_violations, sod_counts = sod_detect_sod(users_df)

    # Stage 3: Feature Engineering
    print("[3/6] Engineering Features (Percentile-derived)...")
    feature_df = engineer_features(users_df, events_df, graph_metrics)

    # Stage 4: ML Anomaly Scoring
    print("[4/6] ML Anomaly Scoring (Isolation Forest)...")
    # 4.1 Score events first
    scored_events_df = score_events(events_df, users_df)

    # 4.2 Aggregate flagged event counts into feature_df
    feature_df = aggregate_event_flags(scored_events_df, feature_df)

    # 4.3 Score users (Isolation Forest trains on features, including flagged_event_count)
    scored_users_df = score_users(feature_df)

    # Stage 5: Context, Exceptions & Novelty Layers
    print("[5/6] Applying Context, Exceptions & Novelty...")
    # 5.1 Build digital twins
    twin_profiles = get_digital_twin_profiles(users_df)

    # 5.2 Apply novelty features
    enhanced_scored_df = apply_novelty_features(scored_users_df, feature_df, users_df, events_df, graph_metrics)

    # 5.3 Apply exceptions (which now includes feedback loop check)
    final_scored_df = apply_exceptions(enhanced_scored_df, users_df, events_df)

    # Add sod_violations_count column for dashboard
    final_scored_df["sod_violations_count"] = final_scored_df["user_id"].map(sod_counts)

    # Ensure final_scored_df has index set to user_id for API lookups
    if "user_id" in final_scored_df.columns:
        final_scored_df = final_scored_df.set_index("user_id")

    exception_tags = dict(zip(final_scored_df.index, final_scored_df['exception_tags']))

    # Bonus: Clustering
    cluster_input_scored = final_scored_df.reset_index() if final_scored_df.index.name == "user_id" else final_scored_df.copy()
    cluster_df = cluster_users(feature_df, cluster_input_scored, users_df=users_df)
    cluster_summary = get_cluster_summary()

    # Extra: Attack Path Analysis
    attack_paths = analyze_attack_paths(G, cluster_input_scored, users_df)

    # Generate findings & package results for each user
    all_results = []
    for user_id, user_row in final_scored_df.iterrows():
        f_row = feature_df[feature_df['user_id'] == user_id].iloc[0] if user_id in feature_df['user_id'].values else pd.Series()
        u_events = events_df[events_df['user_id'] == user_id]

        # Context findings
        findings = generate_findings(
            user_row.rename_axis("user_id") if hasattr(user_row, "rename_axis") else user_row,
            f_row,
            u_events,
            exception_tags.get(user_id, []),
            graph_metrics,
            graph_sod
        )

        # Map compliance
        findings = map_compliance(findings)

        # Assemble full record
        user_data = user_row.to_dict()
        user_data["user_id"] = user_id
        u_info = users_df[users_df['user_id'] == user_id]
        if not u_info.empty:
            u_info_dict = u_info.iloc[0].to_dict()
            user_data.update({
                'username': u_info_dict.get('username', 'unknown'),
                'department': u_info_dict.get('department', ''),
                'job_title': u_info_dict.get('job_title', ''),
                'privilege_level': u_info_dict.get('privilege_level', 'user')
            })
        user_data['findings'] = findings

        # Add risk_score mapping (some places use 'risk_score', others 'adjusted_score')
        user_data['risk_score'] = user_data.get('anomaly_score', 0.0)

        # Novelty: Reversibility & Recurrence
        user_data['reversibility_checks'] = check_reversibility(user_id, users_df, events_df)
        user_data['recurrence_patterns'] = detect_recurrence(user_id, events_df)

        # Stage 6: Explanation Generation
        explanation = generate_explanation(user_data)
        all_results.append(explanation)

    print("[6/6] Generating Reports & Visualizations...")

    # Filter to only CRITICAL and HIGH accounts for the sample report
    high_risk_results = [r for r in all_results if r.get('risk_level') in ['CRITICAL', 'HIGH']]

    # Generate Summaries
    exec_summary = generate_executive_summary(all_results, users_df, events_df)
    comp_report = generate_compliance_report(all_results)

    # Compile final report
    risk_counts = final_scored_df['risk_level'].value_counts().to_dict()
    metadata = {
        "total_users": len(users_df),
        "total_events": len(events_df),
        "critical_risks": risk_counts.get('CRITICAL', 0),
        "high_risks": risk_counts.get('HIGH', 0),
        "processing_time_seconds": round(time.time() - start_time, 2)
    }

    final_report = generate_report(high_risk_results, metadata)

    # Write JSON report
    report_path = Path(output_dir) / 'sample_report.json'
    with open(report_path, 'w') as f:
        json.dump(final_report, f, indent=2)

    # Export graph
    graph_path = str(Path(output_dir) / 'privilege_graph.html')
    export_graph_df = final_scored_df.reset_index() if final_scored_df.index.name == "user_id" else final_scored_df.copy()
    export_graph_html(G, export_graph_df, graph_path)

    # ------------------------------------------------------------------
    # Merge original user metadata back into scored dataframe
    # ------------------------------------------------------------------
    user_meta_cols = [
        "user_id",
        "username",
        "email",
        "department",
        "job_title",
        "privilege_level",
        "hire_date",
        "tenure_days",
        "systems_access",
        "systems_list",
        "system_count"
    ]

    user_meta_df = users_df[user_meta_cols].copy()

    final_scored_df = (
        user_meta_df
        .set_index("user_id")
        .join(final_scored_df, how="left", rsuffix="_score")
    )

    for col in [
        "department_score",
        "privilege_level_score",
        "system_count_score",
        "tenure_days_score"
    ]:
        if col in final_scored_df.columns:
            final_scored_df.drop(columns=[col], inplace=True)

    # Add events_per_user to final_scored_df for dashboard
    events_per_user = events_df.groupby("user_id").size().rename("events_per_user")
    final_scored_df = final_scored_df.join(events_per_user, how="left").fillna({"events_per_user": 0})
    final_scored_df["events_per_user"] = final_scored_df["events_per_user"].astype(int)

    # Add adjusted_score and adjusted_risk_level to final_scored_df
    final_scored_df["adjusted_score"] = final_scored_df["anomaly_score"]
    final_scored_df["adjusted_risk_level"] = final_scored_df["risk_level"]

    elapsed = time.time() - start_time
    print(f"\n Pipeline completed successfully in {elapsed:.2f} seconds")

    return {
        'users_df': final_scored_df,
        'events_df': events_df,
        'all_results': all_results,
        'high_risk_results': high_risk_results,
        'exec_summary': exec_summary,
        'comp_report': comp_report,
        'cluster_summary': cluster_summary,
        'attack_paths': attack_paths,
        'metadata': metadata,
        'G': G,
        'sod_violations': sod_violations,
        'twin_profiles': twin_profiles,
        'user_metrics': graph_metrics
    }