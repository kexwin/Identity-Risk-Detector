"""
Stage 1: Data Ingestion & Normalization
========================================
Loads identity_users.csv and identity_events.csv, parses fields,
adds derived columns, and reports data-quality stats.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_REFERENCE_DATE = pd.Timestamp("2026-04-20")


def get_reference_date() -> pd.Timestamp:
    """Return the canonical reference date used for tenure / staleness maths.

    Returns
    -------
    pd.Timestamp
        Fixed reference date ``2026-04-20``.
    """
    return _REFERENCE_DATE


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------
def load_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalise the identity-users and identity-events CSVs.

    Parameters
    ----------
    data_dir : str, default ``'data'``
        Directory (relative or absolute) that contains
        ``identity_users.csv`` and ``identity_events.csv``.

    Returns
    -------
    users_df : pd.DataFrame
        300 rows – one per user.  Extra columns added:
        * ``systems_list``  – ``list[str]`` parsed from pipe-delimited
          ``systems_access``.
        * ``system_count``  – ``int`` length of ``systems_list``.
        * ``tenure_days``   – ``int`` days between ``hire_date`` and
          the reference date (2026-04-20).
    events_df : pd.DataFrame
        900 rows – one per event.  ``timestamp`` parsed to
        ``datetime64[ns]``.

    Raises
    ------
    FileNotFoundError
        If either CSV is missing from *data_dir*.
    """
    users_path = os.path.join(data_dir, "identity_users.csv")
    events_path = os.path.join(data_dir, "identity_events.csv")

    for p in (users_path, events_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Required data file not found: {p}")

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------
    users_df = pd.read_csv(users_path)

    # Trim any stray whitespace from column names
    users_df.columns = users_df.columns.str.strip()

    # Parse dates
    users_df["last_login"] = pd.to_datetime(users_df["last_login"], errors="coerce")
    users_df["hire_date"] = pd.to_datetime(users_df["hire_date"], errors="coerce")

    # Normalise boolean-like is_active (CSV stores lowercase strings)
    users_df["is_active"] = (
        users_df["is_active"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
    )

    # Parse pipe-delimited systems_access → list
    users_df["systems_list"] = (
        users_df["systems_access"]
        .fillna("")
        .astype(str)
        .apply(lambda s: [x.strip() for x in s.split("|") if x.strip()])
    )
    users_df["system_count"] = users_df["systems_list"].apply(len)

    # Tenure in days
    users_df["tenure_days"] = (
        (_REFERENCE_DATE - users_df["hire_date"]).dt.days
    )
    # Guard against missing hire_date → NaT → NaN
    users_df["tenure_days"] = users_df["tenure_days"].fillna(0).astype(int)

    # Fill remaining missing values gracefully
    users_df["days_inactive"] = (
        users_df["days_inactive"].fillna(0).astype(int)
    )
    users_df["privilege_level"] = users_df["privilege_level"].fillna("user")
    users_df["department"] = users_df["department"].fillna("Unknown")

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    events_df = pd.read_csv(events_path)
    events_df.columns = events_df.columns.str.strip()

    events_df["timestamp"] = pd.to_datetime(
        events_df["timestamp"], errors="coerce"
    )

    # Fill missing categorical fields
    events_df["action"] = events_df["action"].fillna("unknown")
    events_df["resource_sensitivity"] = events_df["resource_sensitivity"].fillna(
        "low"
    )
    events_df["status"] = events_df["status"].fillna("unknown")
    events_df["time_classification"] = events_df["time_classification"].fillna(
        "business_hours"
    )

    # ------------------------------------------------------------------
    # Quality report
    # ------------------------------------------------------------------
    _print_quality_report(users_df, events_df)

    return users_df, events_df


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------
def _print_quality_report(users_df: pd.DataFrame, events_df: pd.DataFrame) -> None:
    """Print a compact data-quality summary to stdout.

    Parameters
    ----------
    users_df : pd.DataFrame
        Loaded and normalised user data.
    events_df : pd.DataFrame
        Loaded and normalised event data.
    """
    print("=" * 60)
    print("  DATA QUALITY REPORT  -  Stage 1: Ingest")
    print("=" * 60)

    # Users
    print(f"\n[Users]  rows={len(users_df)}")
    print(f"  Missing values per column:")
    missing = users_df.isnull().sum()
    for col, cnt in missing.items():
        if cnt > 0:
            print(f"    {col}: {cnt}")
    if missing.sum() == 0:
        print("    (none)")

    priv_counts = users_df["privilege_level"].value_counts()
    print(f"\n  Privilege-level distribution:")
    for level, cnt in priv_counts.items():
        print(f"    {level}: {cnt}")

    dept_counts = users_df["department"].value_counts()
    print(f"\n  Departments ({len(dept_counts)}): {', '.join(dept_counts.index[:8])}{'...' if len(dept_counts) > 8 else ''}")

    print(f"  days_inactive  range: {users_df['days_inactive'].min()} - {users_df['days_inactive'].max()}")
    print(f"  tenure_days    range: {users_df['tenure_days'].min()} - {users_df['tenure_days'].max()}")
    print(f"  system_count   range: {users_df['system_count'].min()} - {users_df['system_count'].max()}")

    # Events
    print(f"\n[Events]  rows={len(events_df)}")
    print(f"  Missing values per column:")
    emissing = events_df.isnull().sum()
    for col, cnt in emissing.items():
        if cnt > 0:
            print(f"    {col}: {cnt}")
    if emissing.sum() == 0:
        print("    (none)")

    unique_users_in_events = events_df["user_id"].nunique()
    total_users = len(users_df)
    users_without_events = total_users - unique_users_in_events
    print(f"\n  Unique users in events: {unique_users_in_events} / {total_users}")
    print(f"  Users with ZERO events: {users_without_events}")

    print(f"\n  Actions: {sorted(events_df['action'].unique())}")
    print(f"  Resources: {sorted(events_df['resource'].unique())}")

    tc = events_df["time_classification"].value_counts()
    print(f"  Time classification:")
    for k, v in tc.items():
        print(f"    {k}: {v}")

    status_counts = events_df["status"].value_counts()
    total_events = len(events_df)
    fail_count = status_counts.get("failure", 0)
    print(f"\n  Status: success={status_counts.get('success', 0)}  failure={fail_count}  ({fail_count / total_events * 100:.1f}%)")

    ts_min = events_df["timestamp"].min()
    ts_max = events_df["timestamp"].max()
    print(f"  Timestamp range: {ts_min} -> {ts_max}")

    print("=" * 60)
