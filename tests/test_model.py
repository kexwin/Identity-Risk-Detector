"""
Tests for src.model — anomaly scoring.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import load_data
from src.graph import build_privilege_graph, compute_graph_metrics
from src.features import engineer_features
from src.model import score_users, get_model_stats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scored_data():
    users_df, events_df = load_data("data")
    G = build_privilege_graph(users_df)
    gm = compute_graph_metrics(G, users_df)
    feature_df = engineer_features(users_df, events_df, gm)
    scored_df = score_users(feature_df)
    return scored_df, feature_df


@pytest.fixture(scope="module")
def scored_df(scored_data):
    return scored_data[0]


@pytest.fixture(scope="module")
def feature_df(scored_data):
    return scored_data[1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestScoredOutput:
    """Verify scored DataFrame structure."""

    def test_not_empty(self, scored_df):
        assert len(scored_df) > 0

    def test_has_user_id(self, scored_df):
        assert "user_id" in scored_df.columns

    def test_has_anomaly_score(self, scored_df):
        assert "anomaly_score" in scored_df.columns

    def test_has_risk_level(self, scored_df):
        assert "risk_level" in scored_df.columns


class TestScoreRange:
    """Anomaly scores must be in [0, 100]."""

    def test_min_score(self, scored_df):
        assert scored_df["anomaly_score"].min() >= 0, "Scores should be >= 0"

    def test_max_score(self, scored_df):
        assert scored_df["anomaly_score"].max() <= 100, "Scores should be <= 100"

    def test_scores_are_numeric(self, scored_df):
        assert pd.api.types.is_numeric_dtype(scored_df["anomaly_score"])


class TestRiskLevels:
    """Risk level categories."""

    VALID_LEVELS = {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

    def test_valid_levels(self, scored_df):
        actual = set(scored_df["risk_level"].unique())
        assert actual.issubset(self.VALID_LEVELS), f"Unexpected levels: {actual - self.VALID_LEVELS}"

    def test_all_users_have_level(self, scored_df):
        assert scored_df["risk_level"].notna().all()

    def test_critical_has_high_score(self, scored_df):
        critical = scored_df[scored_df["risk_level"] == "CRITICAL"]
        if len(critical) > 0:
            assert critical["anomaly_score"].min() >= 50, (
                "CRITICAL users should have score >= 50"
            )


class TestModelStats:
    """Model metadata."""

    def test_returns_dict(self):
        stats = get_model_stats()
        assert isinstance(stats, dict)
