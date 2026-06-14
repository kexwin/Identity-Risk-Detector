"""
Tests for src.features — feature engineering.
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
from src.features import engineer_features, compute_percentile_thresholds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pipeline_data():
    users_df, events_df = load_data("data")
    G = build_privilege_graph(users_df)
    graph_metrics = compute_graph_metrics(G, users_df)
    feature_df = engineer_features(users_df, events_df, graph_metrics)
    return users_df, events_df, feature_df, graph_metrics


@pytest.fixture(scope="module")
def feature_df(pipeline_data):
    return pipeline_data[2]


@pytest.fixture(scope="module")
def users_df(pipeline_data):
    return pipeline_data[0]


@pytest.fixture(scope="module")
def events_df(pipeline_data):
    return pipeline_data[1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFeatureShape:
    """Feature DataFrame shape and structure."""

    def test_not_empty(self, feature_df):
        assert len(feature_df) > 0

    def test_has_user_id(self, feature_df):
        assert "user_id" in feature_df.columns

    def test_user_id_unique(self, feature_df):
        assert feature_df["user_id"].is_unique

    def test_multiple_features(self, feature_df):
        # Expect user_id + at least 5 feature columns
        assert len(feature_df.columns) >= 6, (
            f"Expected >=6 columns, got {len(feature_df.columns)}: {list(feature_df.columns)}"
        )


class TestFeatureTypes:
    """Feature types should be numeric (except metadata columns)."""

    def test_numeric_features(self, feature_df):
        non_numeric_allowed = {
            "user_id",
            "department",
            "privilege_level"
        }

        numeric_cols = [
            c for c in feature_df.columns
            if c not in non_numeric_allowed
        ]

        for col in numeric_cols:
            assert pd.api.types.is_numeric_dtype(feature_df[col]), (
                f"Column '{col}' should be numeric, got {feature_df[col].dtype}"
            )

    def test_no_all_nan_columns(self, feature_df):
        for col in feature_df.columns:
            assert not feature_df[col].isna().all(), (
                f"Column '{col}' is all NaN"
            )


class TestPercentileThresholds:
    """Percentile threshold computation."""

    def test_returns_dict(self, users_df, events_df):
        thresholds = compute_percentile_thresholds(users_df, events_df)
        assert isinstance(thresholds, dict)

    def test_threshold_values_numeric(self, users_df, events_df):
        thresholds = compute_percentile_thresholds(users_df, events_df)

        dict_thresholds = {
            "system_count_dept_medians",
            "dept_priv_mode"
        }

        for k, v in thresholds.items():
            if k in dict_thresholds:
                assert isinstance(v, dict), (
                    f"Threshold '{k}' should be dict, got {type(v)}"
                )
            else:
                assert isinstance(
                    v,
                    (int, float, np.integer, np.floating)
                ), (
                    f"Threshold '{k}' should be numeric, got {type(v)}"
                )