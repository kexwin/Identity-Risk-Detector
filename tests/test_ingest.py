"""
Tests for src.ingest — data loading and parsing.
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest import load_data, get_reference_date


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def raw_data():
    """Load users and events DataFrames once for the module."""
    users_df, events_df = load_data("data")
    return users_df, events_df


@pytest.fixture(scope="module")
def users_df(raw_data):
    return raw_data[0]


@pytest.fixture(scope="module")
def events_df(raw_data):
    return raw_data[1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestDataLoading:
    """Verify the CSVs can be loaded and have expected shape."""

    def test_users_not_empty(self, users_df):
        assert len(users_df) > 0, "Users dataframe should not be empty"

    def test_events_not_empty(self, events_df):
        assert len(events_df) > 0, "Events dataframe should not be empty"

    def test_users_row_count(self, users_df):
        # 301 data rows in the CSV (header + 301 users = 302 lines, with trailing newline)
        assert len(users_df) >= 100, f"Expected >=100 users, got {len(users_df)}"

    def test_events_row_count(self, events_df):
        assert len(events_df) >= 100, f"Expected >=100 events, got {len(events_df)}"


class TestUserColumns:
    """Verify required columns in users_df."""

    REQUIRED = [
        "user_id", "username", "email", "department", "job_title",
        "privilege_level", "systems_access", "last_login", "days_inactive", "is_active",
    ]

    @pytest.mark.parametrize("col", REQUIRED)
    def test_column_present(self, users_df, col):
        assert col in users_df.columns, f"Missing column: {col}"

    def test_user_id_unique(self, users_df):
        assert users_df["user_id"].is_unique, "user_id should be unique"

    def test_privilege_levels(self, users_df):
        valid = {"user", "power-user", "admin", "service-account"}
        actual = set(users_df["privilege_level"].unique())
        assert actual.issubset(valid), f"Unexpected privilege levels: {actual - valid}"


class TestEventColumns:
    """Verify required columns in events_df."""

    REQUIRED = [
        "timestamp", "user_id", "username", "action", "resource",
        "resource_sensitivity", "status", "source_ip", "time_classification",
    ]

    @pytest.mark.parametrize("col", REQUIRED)
    def test_column_present(self, events_df, col):
        assert col in events_df.columns, f"Missing column: {col}"

    def test_resource_sensitivity_values(self, events_df):
        valid = {"high", "medium", "low"}
        actual = set(events_df["resource_sensitivity"].str.lower().unique())
        assert actual.issubset(valid), f"Unexpected sensitivity: {actual - valid}"


class TestReferenceDate:
    """Verify reference date helper."""

    def test_returns_timestamp(self):
        ref = get_reference_date()
        assert isinstance(ref, pd.Timestamp)

    def test_date_value(self):
        ref = get_reference_date()
        assert ref == pd.Timestamp("2026-04-20")
