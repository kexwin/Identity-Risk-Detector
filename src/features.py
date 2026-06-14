"""
Stage 3: Feature Engineering
==============================
Builds 15 user-level features – every threshold is derived from data
percentiles at runtime (no hard-coded magic numbers).  Also provides
cross-department access detection and department→resource mappings.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .ingest import get_reference_date

# ---------------------------------------------------------------------------
# Privilege encoding map
# ---------------------------------------------------------------------------
_PRIV_ENCODE = {
    "user": 0,
    "power-user": 1,
    "admin": 2,
    "service-account": 1,
}

_SENSITIVE_SYSTEMS = {"PROD_DB", "ADMIN_SYS", "SIEM", "Customer_Vault"}


# ===================================================================
# 1. Percentile thresholds
# ===================================================================
def compute_percentile_thresholds(
    users_df: pd.DataFrame, events_df: pd.DataFrame
) -> Dict[str, Any]:
    """Compute all percentile cut-offs used by the feature pipeline.

    Parameters
    ----------
    users_df : pd.DataFrame
        Normalised user table (from ``ingest.load_data``).
    events_df : pd.DataFrame
        Normalised event table.

    Returns
    -------
    dict
        Keys describe each threshold; values are the numeric cut-offs.
    """
    thresholds: Dict[str, Any] = {}

    # days_inactive P75 for admins
    admin_mask = users_df["privilege_level"] == "admin"
    thresholds["P75_days_inactive_admin"] = (
        users_df.loc[admin_mask, "days_inactive"].quantile(0.75)
        if admin_mask.any()
        else 0
    )

    # days_inactive P75 for power-users
    pu_mask = users_df["privilege_level"] == "power-user"
    thresholds["P75_days_inactive_power_user"] = (
        users_df.loc[pu_mask, "days_inactive"].quantile(0.75)
        if pu_mask.any()
        else 0
    )

    # tenure_days P10 (new-hire cut-off)
    thresholds["P10_tenure_days"] = users_df["tenure_days"].quantile(0.10)

    # system_count median per department
    dept_medians = users_df.groupby("department")["system_count"].median()
    thresholds["system_count_dept_medians"] = dept_medians.to_dict()

    # department privilege-level mode
    dept_priv_mode = (
        users_df.groupby("department")["privilege_level"]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) > 0 else "user")
    )
    thresholds["dept_priv_mode"] = dept_priv_mode.to_dict()

    # Global failure rate (reference)
    if len(events_df) > 0:
        thresholds["global_failure_rate"] = (
            (events_df["status"] == "failure").sum() / len(events_df)
        )
    else:
        thresholds["global_failure_rate"] = 0.0

    return thresholds


# ===================================================================
# 2. Department → resource mapping
# ===================================================================
def _build_dept_resource_map(
    events_df: pd.DataFrame, users_df: pd.DataFrame
) -> Dict[str, List[str]]:
    """Build a mapping of department → most-accessed resources.

    Parameters
    ----------
    events_df : pd.DataFrame
        Event data.
    users_df : pd.DataFrame
        User data (for dept lookup).

    Returns
    -------
    dict
        ``{ dept: [resource, …] }`` sorted by frequency desc.
    """
    merged = events_df.merge(
        users_df[["user_id", "department"]], on="user_id", how="left"
    )
    dept_res = (
        merged.groupby(["department", "resource"])
        .size()
        .reset_index(name="count")
    )
    mapping: Dict[str, List[str]] = {}
    for dept, grp in dept_res.groupby("department"):
        ordered = grp.sort_values("count", ascending=False)["resource"].tolist()
        mapping[dept] = ordered
    return mapping


# ===================================================================
# 3. Cross-department access detection
# ===================================================================
def detect_cross_dept_access(
    events_df: pd.DataFrame, users_df: pd.DataFrame
) -> pd.DataFrame:
    """Flag users who access resources unusual for their department.

    A resource is considered "unusual" for a department if fewer than
    5 % of that department's total events involve it.

    Parameters
    ----------
    events_df : pd.DataFrame
        Event data.
    users_df : pd.DataFrame
        User data (for department lookup).

    Returns
    -------
    pd.DataFrame
        Columns: ``user_id``, ``cross_dept_access_count``,
        ``cross_dept_resources`` (pipe-delimited string).
    """
    merged = events_df.merge(
        users_df[["user_id", "department"]], on="user_id", how="left"
    )
    # Compute resource frequency per department
    dept_totals = merged.groupby("department").size()
    dept_res_counts = merged.groupby(["department", "resource"]).size()
    dept_res_pct = dept_res_counts.div(dept_totals, level="department").fillna(0)

    # Resources where pct < 0.05 are "unusual" for that dept
    unusual = set()
    for (dept, res), pct in dept_res_pct.items():
        if pct < 0.05:
            unusual.add((dept, res))

    # Flag each event
    merged["is_cross_dept"] = merged.apply(
        lambda r: (r.get("department", ""), r.get("resource", "")) in unusual,
        axis=1,
    )

    cross_counts = (
        merged[merged["is_cross_dept"]]
        .groupby("user_id")
        .agg(
            cross_dept_access_count=("is_cross_dept", "sum"),
            cross_dept_resources=("resource", lambda s: "|".join(sorted(set(s)))),
        )
        .reset_index()
    )

    # Ensure every user has a row
    all_users = users_df[["user_id"]].copy()
    result = all_users.merge(cross_counts, on="user_id", how="left")
    result["cross_dept_access_count"] = result["cross_dept_access_count"].fillna(0).astype(int)
    result["cross_dept_resources"] = result["cross_dept_resources"].fillna("")

    n_flagged = (result["cross_dept_access_count"] > 0).sum()
    print(f"[Features]  Cross-dept access: {n_flagged} users flagged")
    return result


# ===================================================================
# 4. Main feature engineering
# ===================================================================
def engineer_features(
    users_df: pd.DataFrame,
    events_df: pd.DataFrame,
    graph_metrics: Dict[str, Any],
) -> pd.DataFrame:
    """Build a 15-feature DataFrame for every user.

    Parameters
    ----------
    users_df : pd.DataFrame
        Normalised user table from ``ingest.load_data``.
    events_df : pd.DataFrame
        Normalised event table.
    graph_metrics : dict
        Output of ``graph.compute_graph_metrics`` – must contain
        ``blast_radius`` and ``shared_system_users`` sub-dicts.

    Returns
    -------
    pd.DataFrame
        One row per ``user_id`` with columns:

        1. ``days_inactive``
        2. ``is_stale_admin``
        3. ``is_stale_power_user``
        4. ``system_count``
        5. ``system_count_vs_dept_median``
        6. ``has_sensitive_system``
        7. ``privilege_vs_dept_norm``
        8. ``is_service_account_no_owner``
        9. ``recent_event_count``
        10. ``after_hours_event_ratio``
        11. ``high_sensitivity_export_count``
        12. ``admin_op_off_hours_count``
        13. ``failure_rate``
        14. ``tenure_days``
        15. ``is_new_hire``

        Plus helper columns: ``user_id``, ``blast_radius``,
        ``shared_system_users``, ``privilege_level``,
        ``department``.
    """
    ref_date = get_reference_date()
    thresholds = compute_percentile_thresholds(users_df, events_df)

    # Start from user skeleton
    feat = users_df[["user_id", "department", "privilege_level"]].copy()

    # ---- 1. days_inactive (direct) --------------------------------
    feat["days_inactive"] = users_df["days_inactive"].values

    # ---- 2. is_stale_admin ----------------------------------------
    p75_admin = thresholds["P75_days_inactive_admin"]
    feat["is_stale_admin"] = (
        (users_df["privilege_level"] == "admin")
        & (users_df["days_inactive"] > p75_admin)
    ).astype(int)

    # ---- 3. is_stale_power_user -----------------------------------
    p75_pu = thresholds["P75_days_inactive_power_user"]
    feat["is_stale_power_user"] = (
        (users_df["privilege_level"] == "power-user")
        & (users_df["days_inactive"] > p75_pu)
    ).astype(int)

    # ---- 4. system_count (from graph / user table) ----------------
    feat["system_count"] = users_df["system_count"].values

    # ---- 5. system_count_vs_dept_median ---------------------------
    dept_med = thresholds["system_count_dept_medians"]
    feat["system_count_vs_dept_median"] = feat.apply(
        lambda r: r["system_count"] - dept_med.get(r["department"], 1.0),
        axis=1,
    )

    # ---- 6. has_sensitive_system ----------------------------------
    feat["has_sensitive_system"] = users_df["systems_list"].apply(
        lambda sl: int(bool(_SENSITIVE_SYSTEMS & set(sl)))
    )

    # ---- 7. privilege_vs_dept_norm --------------------------------
    dept_mode = thresholds["dept_priv_mode"]
    feat["privilege_vs_dept_norm"] = feat.apply(
        lambda r: (
            _PRIV_ENCODE.get(r["privilege_level"], 0)
            - _PRIV_ENCODE.get(dept_mode.get(r["department"], "user"), 0)
        ),
        axis=1,
    )

    # ---- 8. is_service_account_no_owner ---------------------------
    users_with_events = set(events_df["user_id"].unique())
    feat["is_service_account_no_owner"] = (
        (users_df["privilege_level"] == "service-account")
        & (~users_df["user_id"].isin(users_with_events))
    ).astype(int)

    # ----------------------------------------------------------------
    # Event-level aggregates (features 9-13)
    # ----------------------------------------------------------------
    trailing_90 = ref_date - pd.Timedelta(days=90)

    if len(events_df) > 0:
        # 9. recent_event_count
        recent_mask = events_df["timestamp"] >= trailing_90
        recent_counts = (
            events_df.loc[recent_mask]
            .groupby("user_id")
            .size()
            .rename("recent_event_count")
        )

        # 10. after_hours_event_ratio
        off_hours_mask = events_df["time_classification"].isin(
            ["night", "unusual_hours", "weekend"]
        )
        off_hours_counts = (
            events_df.loc[off_hours_mask]
            .groupby("user_id")
            .size()
            .rename("off_hours_count")
        )
        total_counts = events_df.groupby("user_id").size().rename("total_count")

        # 11. high_sensitivity_export_count
        hs_export_mask = (events_df["action"] == "export_data") & (
            events_df["resource_sensitivity"] == "high"
        )
        hs_export = (
            events_df.loc[hs_export_mask]
            .groupby("user_id")
            .size()
            .rename("high_sensitivity_export_count")
        )

        # 12. admin_op_off_hours_count
        admin_op_off = (events_df["action"] == "admin_operation") & (
            events_df["time_classification"] != "business_hours"
        )
        admin_op_off_count = (
            events_df.loc[admin_op_off]
            .groupby("user_id")
            .size()
            .rename("admin_op_off_hours_count")
        )

        # 13. failure_rate
        failure_mask = events_df["status"] == "failure"
        failure_counts = (
            events_df.loc[failure_mask]
            .groupby("user_id")
            .size()
            .rename("failure_count")
        )

        # Merge all event aggregates
        event_agg = pd.DataFrame({"user_id": users_df["user_id"]})
        event_agg = event_agg.merge(recent_counts, on="user_id", how="left")
        event_agg = event_agg.merge(off_hours_counts, on="user_id", how="left")
        event_agg = event_agg.merge(total_counts, on="user_id", how="left")
        event_agg = event_agg.merge(hs_export, on="user_id", how="left")
        event_agg = event_agg.merge(admin_op_off_count, on="user_id", how="left")
        event_agg = event_agg.merge(failure_counts, on="user_id", how="left")

        event_agg = event_agg.fillna(0)

        feat["recent_event_count"] = event_agg["recent_event_count"].values.astype(int)

        feat["after_hours_event_ratio"] = np.where(
            event_agg["total_count"] > 0,
            event_agg["off_hours_count"] / event_agg["total_count"],
            0.0,
        )

        feat["high_sensitivity_export_count"] = (
            event_agg["high_sensitivity_export_count"].values.astype(int)
        )

        feat["admin_op_off_hours_count"] = (
            event_agg["admin_op_off_hours_count"].values.astype(int)
        )

        feat["failure_rate"] = np.where(
            event_agg["total_count"] > 0,
            event_agg["failure_count"] / event_agg["total_count"],
            0.0,
        )
    else:
        # No events at all → zero everything
        for col in [
            "recent_event_count",
            "after_hours_event_ratio",
            "high_sensitivity_export_count",
            "admin_op_off_hours_count",
            "failure_rate",
        ]:
            feat[col] = 0

    # ---- 14. tenure_days ------------------------------------------
    feat["tenure_days"] = users_df["tenure_days"].values

    # ---- 15. is_new_hire ------------------------------------------
    p10_tenure = thresholds["P10_tenure_days"]
    feat["is_new_hire"] = (users_df["tenure_days"] < p10_tenure).astype(int)

    # ----------------------------------------------------------------
    # Graph metrics (bonus columns carried forward)
    # ----------------------------------------------------------------
    blast = graph_metrics.get("blast_radius", {})
    shared = graph_metrics.get("shared_system_users", {})
    feat["blast_radius"] = feat["user_id"].map(blast).fillna(0).astype(int)
    feat["shared_system_users"] = feat["user_id"].map(shared).fillna(0).astype(int)

    # ----------------------------------------------------------------
    # Print summary
    # ----------------------------------------------------------------
    _print_feature_summary(feat, thresholds)

    return feat


# ---------------------------------------------------------------------------
# Internal summary printer
# ---------------------------------------------------------------------------
def _print_feature_summary(
    feat: pd.DataFrame, thresholds: Dict[str, Any]
) -> None:
    """Log a compact feature-engineering summary.

    Parameters
    ----------
    feat : pd.DataFrame
        Completed feature matrix.
    thresholds : dict
        Percentile thresholds from :func:`compute_percentile_thresholds`.
    """
    print("=" * 60)
    print("  FEATURE ENGINEERING SUMMARY  -  Stage 3")
    print("=" * 60)
    print(f"  Total users:                {len(feat)}")
    print(f"  Stale admins:               {feat['is_stale_admin'].sum()}")
    print(f"  Stale power-users:          {feat['is_stale_power_user'].sum()}")
    print(f"  Sensitive-system holders:   {feat['has_sensitive_system'].sum()}")
    print(f"  Service-acct no events:     {feat['is_service_account_no_owner'].sum()}")
    print(f"  New hires:                  {feat['is_new_hire'].sum()}")
    print(f"  Users with >0 events:       {(feat['recent_event_count'] > 0).sum()}")
    print(f"\n  Key thresholds:")
    print(f"    P75 days_inactive (admin):      {thresholds['P75_days_inactive_admin']:.0f}")
    print(f"    P75 days_inactive (power-user): {thresholds['P75_days_inactive_power_user']:.0f}")
    print(f"    P10 tenure_days (new-hire):      {thresholds['P10_tenure_days']:.0f}")
    print(f"    Global failure rate:             {thresholds['global_failure_rate']:.3f}")
    print("=" * 60)
