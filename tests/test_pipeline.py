"""
Automated Pipeline Tests
Tests all stages of the risk detection pipeline.
"""

import pytest
import pandas as pd
import networkx as nx
from src.ingest import load_data, get_reference_date
from src.graph import build_privilege_graph, compute_graph_metrics
from src.features import engineer_features, compute_percentile_thresholds
from src.model import score_events, score_users
from src.context import apply_exceptions
from src.sod import detect_sod_violations
from src.pipeline import run_pipeline


def test_stage1_ingestion():
    users_df, events_df = load_data("data")
    
    assert len(users_df) == 300
    assert len(events_df) == 900
    assert "user_id" in users_df.columns
    assert "systems_list" in users_df.columns


def test_stage2_graph():
    users_df, _ = load_data("data")
    G = build_privilege_graph(users_df)
    metrics = compute_graph_metrics(G, users_df)
    
    assert isinstance(G, nx.Graph)
    assert len(metrics["blast_radius"]) == 300
    assert "USR00000" in metrics["blast_radius"]


def test_stage3_features():
    users_df, events_df = load_data("data")
    G = build_privilege_graph(users_df)
    metrics = compute_graph_metrics(G, users_df)
    features_df = engineer_features(users_df, events_df, metrics)
    
    assert len(features_df) == 300
    assert "days_inactive" in features_df.columns
    assert "is_stale_admin" in features_df.columns
    assert "is_stale_power_user" in features_df.columns


def test_stage4_model():
    users_df, events_df = load_data("data")
    G = build_privilege_graph(users_df)
    metrics = compute_graph_metrics(G, users_df)
    features_df = engineer_features(users_df, events_df, metrics)
    
    scored_events_df = score_events(events_df, users_df)
    from src.model import aggregate_event_flags
    features_df = aggregate_event_flags(scored_events_df, features_df)
    scored_users_df = score_users(features_df)
    
    assert "anomaly_score" in scored_users_df.columns
    assert "risk_level" in scored_users_df.columns
    assert scored_users_df["anomaly_score"].min() >= 0.0
    assert scored_users_df["anomaly_score"].max() <= 100.0


def test_stage5_context():
    users_df, events_df = load_data("data")
    G = build_privilege_graph(users_df)
    metrics = compute_graph_metrics(G, users_df)
    features_df = engineer_features(users_df, events_df, metrics)
    scored_events_df = score_events(events_df, users_df)
    from src.model import aggregate_event_flags
    features_df = aggregate_event_flags(scored_events_df, features_df)
    scored_users_df = score_users(features_df)
    from src.novelty import apply_novelty_features
    enhanced_scored_df = apply_novelty_features(scored_users_df, features_df, users_df, events_df, metrics)
    
    final_users_df = apply_exceptions(enhanced_scored_df, users_df, events_df)
    assert "anomaly_score" in final_users_df.columns
    assert "risk_level" in final_users_df.columns
    assert "exception_tags" in final_users_df.columns


def test_sod_violations():
    users_df, _ = load_data("data")
    # Mock systems list to trigger SoD
    users_df["systems_list"] = [["PROD_DB", "ADMIN_SYS"]] * len(users_df)
    violations, counts = detect_sod_violations(users_df)
    
    assert len(violations) == 300
    assert counts.iloc[0] > 0


def test_full_pipeline():
    results = run_pipeline()
    assert "users_df" in results
    assert "events_df" in results
    assert len(results["users_df"]) == 300
    assert len(results["events_df"]) == 900
