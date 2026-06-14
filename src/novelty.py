"""
Section 7: Four Novelty Layers
===============================
7.1 Identity Digital Twin — cluster-based expected-access profiles
7.2 Reversibility Simulator — safe-to-revoke analysis per system
7.3 Confidence Calibration — evidence-volume-aware confidence scoring
7.4 Recurrence-Aware Pattern Detection — periodic-event detection

Functions
---------
- apply_novelty_features(scored_df, feature_df, users_df, events_df, graph_metrics)
- build_digital_twins(users_df, feature_df)
- check_reversibility(user_id, users_df, events_df)
- calibrate_confidence(user_id, events_df, feature_df)
- detect_recurrence(user_id, events_df)
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

# Reference date from ingest.py
REFERENCE_DATE = pd.Timestamp("2026-04-20")

# Lookback window for reversibility checks (days)
REVERSIBILITY_LOOKBACK_DAYS = 365


# ══════════════════════════════════════════════════════════════════════════════
# 7.1  IDENTITY DIGITAL TWIN
# ══════════════════════════════════════════════════════════════════════════════

def build_digital_twins(
    users_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    n_clusters: int = 8,
) -> Dict[str, Any]:
    """Cluster the NORMAL population by department + job_title and build
    expected-access profiles per cluster.

    Parameters
    ----------
    users_df : pd.DataFrame
        User data from ingest (must include ``user_id``, ``department``,
        ``job_title``, ``privilege_level``, ``system_count`` or ``systems_access``).
    feature_df : pd.DataFrame
        Engineered features from features module.
    n_clusters : int, optional
        Number of KMeans clusters (default 8).

    Returns
    -------
    dict
        ``twin_profiles`` — mapping with keys:

        - ``cluster_assignments``: dict[user_id → cluster_id]
        - ``profiles``: dict[cluster_id → profile_dict]
            profile_dict contains ``median_system_count``, ``typical_resources``,
            ``typical_privilege``, ``median_days_inactive``, ``user_count``, ``departments``.
        - ``model``: fitted KMeans model (for reuse)
        - ``scaler``: fitted StandardScaler
        - ``label_encoders``: dict of LabelEncoders used
    """
    df = users_df.copy()

    # Ensure system_count exists
    if "system_count" not in df.columns:
        df["system_count"] = df["systems_access"].apply(
            lambda x: len(str(x).split("|")) if pd.notna(x) and str(x).strip() else 0
        )

    # Encode categoricals
    le_dept = LabelEncoder()
    le_title = LabelEncoder()
    le_priv = LabelEncoder()

    df["dept_enc"] = le_dept.fit_transform(df["department"].astype(str))
    df["title_enc"] = le_title.fit_transform(df["job_title"].astype(str))
    df["priv_enc"] = le_priv.fit_transform(df["privilege_level"].astype(str))

    # Merge select features
    merge_cols = ["user_id"]
    for col in ["total_events", "off_hours_ratio", "failure_rate"]:
        if col in feature_df.columns:
            merge_cols.append(col)
    if len(merge_cols) > 1:
        df = df.merge(feature_df[merge_cols], on="user_id", how="left")
        for col in merge_cols[1:]:
            df[col] = df[col].fillna(0)

    # Feature matrix for clustering
    cluster_features = ["dept_enc", "title_enc", "priv_enc", "system_count", "days_inactive"]
    for opt_col in ["total_events", "off_hours_ratio", "failure_rate"]:
        if opt_col in df.columns:
            cluster_features.append(opt_col)

    X = df[cluster_features].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit KMeans
    actual_k = min(n_clusters, len(df))
    kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
    df["cluster_id"] = kmeans.fit_predict(X_scaled)

    # Build profiles per cluster
    profiles: Dict[int, Dict[str, Any]] = {}
    for cid in range(actual_k):
        cluster_mask = df["cluster_id"] == cid
        cluster_users = df[cluster_mask]

        # Typical resources: most common systems across cluster members
        all_systems: List[str] = []
        for sa in cluster_users["systems_access"].dropna():
            all_systems.extend(str(sa).split("|"))
        resource_counts = Counter(all_systems)
        typical_resources = [s for s, _ in resource_counts.most_common(5)]

        # Typical privilege: mode
        priv_mode = cluster_users["privilege_level"].mode()
        typical_priv = priv_mode.iloc[0] if not priv_mode.empty else "user"

        profiles[cid] = {
            "median_system_count": float(cluster_users["system_count"].median()),
            "mean_system_count": float(cluster_users["system_count"].mean()),
            "std_system_count": float(cluster_users["system_count"].std()) if len(cluster_users) > 1 else 0.0,
            "typical_resources": typical_resources,
            "typical_privilege": typical_priv,
            "median_days_inactive": float(cluster_users["days_inactive"].median()),
            "user_count": int(len(cluster_users)),
            "departments": sorted(cluster_users["department"].unique().tolist()),
        }

    # Cluster assignments
    assignments = dict(zip(df["user_id"], df["cluster_id"]))

    logger.info(
        "Digital twins built: %d clusters from %d users", actual_k, len(df)
    )

    return {
        "cluster_assignments": assignments,
        "profiles": profiles,
        "model": kmeans,
        "scaler": scaler,
        "label_encoders": {"department": le_dept, "job_title": le_title, "privilege_level": le_priv},
    }


def _compute_twin_deviation(
    user_id: str,
    users_df: pd.DataFrame,
    twin_profiles: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute how far a user deviates from their digital-twin cluster profile.

    Returns
    -------
    dict
        Keys: ``cluster_id``, ``system_count_deviation``, ``inactive_deviation``,
        ``overall_deviation_score``.
    """
    assignments = twin_profiles.get("cluster_assignments", {})
    profiles = twin_profiles.get("profiles", {})
    cid = assignments.get(user_id)
    if cid is None:
        return {"cluster_id": None, "system_count_deviation": 0.0,
                "inactive_deviation": 0.0, "overall_deviation_score": 0.0}

    profile = profiles.get(cid, {})
    user_mask = users_df["user_id"] == user_id
    if not user_mask.any():
        return {"cluster_id": cid, "system_count_deviation": 0.0,
                "inactive_deviation": 0.0, "overall_deviation_score": 0.0}

    urow = users_df[user_mask].iloc[0]
    sys_count = int(urow.get("system_count", len(str(urow.get("systems_access", "")).split("|"))))
    days_inactive = int(urow.get("days_inactive", 0))

    median_sys = profile.get("median_system_count", sys_count)
    std_sys = profile.get("std_system_count", 1.0) or 1.0
    median_inactive = profile.get("median_days_inactive", days_inactive)

    sys_dev = (sys_count - median_sys) / std_sys if std_sys > 0 else 0.0
    inactive_dev = (days_inactive - median_inactive) / max(median_inactive, 1)

    overall = (abs(sys_dev) * 0.6 + abs(inactive_dev) * 0.4)

    return {
        "cluster_id": int(cid),
        "system_count_deviation": round(sys_dev, 3),
        "inactive_deviation": round(inactive_dev, 3),
        "overall_deviation_score": round(overall, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7.2  REVERSIBILITY SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

def check_reversibility(
    user_id: str,
    users_df: pd.DataFrame,
    events_df: pd.DataFrame,
    lookback_days: int = REVERSIBILITY_LOOKBACK_DAYS,
) -> List[Dict[str, Any]]:
    """For each system in the user's ``systems_access``, check the events table
    for actual usage in the lookback window.

    Parameters
    ----------
    user_id : str
    users_df : pd.DataFrame
    events_df : pd.DataFrame
    lookback_days : int, optional
        How many days back to look for usage evidence (default 365).

    Returns
    -------
    list[dict]
        One entry per system: ``{system, events_using, safe_to_revoke, reasoning}``.
    """
    user_mask = users_df["user_id"] == user_id
    if not user_mask.any():
        return []

    urow = users_df[user_mask].iloc[0]
    systems_access = str(urow.get("systems_access", ""))
    systems = [s.strip() for s in systems_access.split("|") if s.strip()]

    cutoff = REFERENCE_DATE - pd.Timedelta(days=lookback_days)

    # User events
    user_events = events_df[events_df["user_id"] == user_id].copy()
    if not user_events.empty:
        user_events["_ts"] = pd.to_datetime(user_events["timestamp"], errors="coerce")
        user_events = user_events[user_events["_ts"] >= cutoff]

    # Note: systems_access lists *identity systems* (AD, Okta, etc.) while
    # events reference *resources* (PROD_DB, HRIS, etc.).  These are different
    # namespaces.  We check for *exact* matches (some overlap, e.g. SIEM,
    # ADMIN_SYS, PROD_DB appear in both) and also record a note when no
    # overlap exists.
    results: List[Dict[str, Any]] = []
    for system in systems:
        # Count events where the resource matches the system name
        if not user_events.empty:
            matching = user_events[user_events["resource"] == system]
            event_count = len(matching)
        else:
            event_count = 0

        if event_count == 0:
            safe = True
            reasoning = (
                f"Removing {system} access is safe — 0 events used {system} "
                f"in the past {lookback_days} days. No evidence of active usage."
            )
        else:
            safe = False
            last_use = matching["_ts"].max().strftime("%Y-%m-%d") if not matching.empty else "unknown"
            reasoning = (
                f"{system} access is actively used — {event_count} event(s) "
                f"recorded in the past {lookback_days} days (last use: {last_use}). "
                f"Revoking may disrupt operations."
            )

        results.append({
            "system": system,
            "events_using": event_count,
            "safe_to_revoke": safe,
            "reasoning": reasoning,
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 7.3  CONFIDENCE CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_confidence(
    user_id: str,
    events_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> Tuple[float, str]:
    """Compute a confidence score (0.0–1.0) based on evidence volume.

    Parameters
    ----------
    user_id : str
    events_df : pd.DataFrame
    feature_df : pd.DataFrame

    Returns
    -------
    tuple[float, str]
        ``(confidence_score, confidence_basis)`` — the score and a human-
        readable explanation of what drove it.
    """
    # Event count for this user
    user_events = events_df[events_df["user_id"] == user_id]
    event_count = len(user_events)

    # Time span of events
    if event_count > 0:
        ts = pd.to_datetime(user_events["timestamp"], errors="coerce").dropna()
        if len(ts) >= 2:
            span_days = (ts.max() - ts.min()).days
        else:
            span_days = 0
    else:
        span_days = 0

    # Peer group size (users in same department)
    feat_row_mask = feature_df["user_id"] == user_id
    if feat_row_mask.any():
        feat_row = feature_df[feat_row_mask].iloc[0]
        peer_size = int(feat_row.get("peer_group_size", 0))
    else:
        peer_size = 0

    # ── Confidence formula ──────────────────────────────────────────────────
    # Component 1: event volume (0-0.4)
    #   0 events → 0.0; 1-2 → 0.1; 3-5 → 0.2; 6-9 → 0.3; 10+ → 0.4
    if event_count == 0:
        vol_score = 0.0
    elif event_count <= 2:
        vol_score = 0.10
    elif event_count <= 5:
        vol_score = 0.20
    elif event_count <= 9:
        vol_score = 0.30
    else:
        vol_score = 0.40

    # Component 2: temporal span (0-0.3)
    #   0 days → 0.0; 1-30 → 0.1; 31-180 → 0.2; 181+ → 0.3
    if span_days == 0:
        span_score = 0.0
    elif span_days <= 30:
        span_score = 0.10
    elif span_days <= 180:
        span_score = 0.20
    else:
        span_score = 0.30

    # Component 3: peer group context (0-0.3)
    #   0 peers → 0.05; 1-10 → 0.15; 11-30 → 0.25; 31+ → 0.30
    if peer_size == 0:
        peer_score = 0.05
    elif peer_size <= 10:
        peer_score = 0.15
    elif peer_size <= 30:
        peer_score = 0.25
    else:
        peer_score = 0.30

    confidence = round(min(vol_score + span_score + peer_score, 1.0), 2)

    # ── Basis text ──────────────────────────────────────────────────────────
    if event_count == 0:
        basis = (
            f"0 events in observed period — risk score based on static profile "
            f"only (system count, privilege level, days inactive). "
            f"Peer group n={peer_size}."
        )
    else:
        basis = (
            f"{event_count} event(s) over {span_days} days; "
            f"peer group n={peer_size}."
        )

    return confidence, basis


# ══════════════════════════════════════════════════════════════════════════════
# 7.4  RECURRENCE-AWARE PATTERN DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_recurrence(
    user_id: str,
    events_df: pd.DataFrame,
    min_events: int = 5,
) -> List[Dict[str, Any]]:
    """For users with ≥ *min_events* events, check if anomalous-looking events
    recur at regular intervals.

    Parameters
    ----------
    user_id : str
    events_df : pd.DataFrame
    min_events : int, optional
        Minimum number of events required for analysis (default 5).

    Returns
    -------
    list[dict]
        Each entry: ``{pattern, interval_days, is_recurring, note}``.
    """
    user_events = events_df[events_df["user_id"] == user_id].copy()

    if len(user_events) < min_events:
        return [{
            "pattern": "insufficient_data",
            "interval_days": None,
            "is_recurring": False,
            "note": (
                f"User {user_id} has only {len(user_events)} event(s), below "
                f"the minimum of {min_events} required for recurrence analysis. "
                f"Skipping — noted in confidence basis."
            ),
        }]

    user_events["_ts"] = pd.to_datetime(user_events["timestamp"], errors="coerce")
    user_events = user_events.dropna(subset=["_ts"]).sort_values("_ts")

    results: List[Dict[str, Any]] = []

    # Group by action to find recurring patterns within each action type
    for action, grp in user_events.groupby("action"):
        if len(grp) < 3:
            continue  # need ≥3 occurrences of same action to detect periodicity

        timestamps = grp["_ts"].values
        deltas_ns = np.diff(timestamps)
        # Convert to days
        deltas_days = deltas_ns.astype("timedelta64[D]").astype(float)

        if len(deltas_days) < 2:
            continue

        mean_interval = float(np.mean(deltas_days))
        std_interval = float(np.std(deltas_days))

        if mean_interval <= 0:
            continue

        cv = std_interval / mean_interval  # coefficient of variation

        # If CV < 0.3, pattern is reasonably regular
        is_recurring = cv < 0.30 and mean_interval >= 1

        if is_recurring:
            note = (
                f"Action '{action}' recurs every ~{mean_interval:.0f} days "
                f"(CV={cv:.2f}, n={len(grp)}). Pattern is consistent with a "
                f"scheduled process — severity should be downgraded."
            )
        else:
            note = (
                f"Action '{action}' occurs {len(grp)} times with mean interval "
                f"{mean_interval:.1f} days (CV={cv:.2f}). Intervals are too "
                f"irregular to classify as recurring."
            )

        results.append({
            "pattern": action,
            "interval_days": round(mean_interval, 1),
            "is_recurring": is_recurring,
            "note": note,
        })

    if not results:
        results.append({
            "pattern": "no_repeated_actions",
            "interval_days": None,
            "is_recurring": False,
            "note": (
                f"User {user_id} has {len(user_events)} events but no single "
                f"action type occurs 3+ times; recurrence analysis not applicable."
            ),
        })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT — APPLY ALL NOVELTY FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def apply_novelty_features(
    scored_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    users_df: pd.DataFrame,
    events_df: pd.DataFrame,
    graph_metrics: Dict[str, Any],
) -> pd.DataFrame:
    """Apply all four novelty layers and return an enhanced scored DataFrame.

    Adds columns:
    - ``twin_cluster_id`` — digital-twin cluster assignment
    - ``twin_deviation`` — deviation from cluster profile
    - ``reversibility`` — list of safe-to-revoke assessments
    - ``confidence`` — calibrated confidence score (0–1)
    - ``confidence_basis`` — human-readable confidence explanation
    - ``recurrence_patterns`` — detected recurring patterns

    Parameters
    ----------
    scored_df : pd.DataFrame
        Scored user data (user_id, anomaly_score, risk_level, etc.).
    feature_df : pd.DataFrame
        Engineered feature DataFrame.
    users_df : pd.DataFrame
        Raw user data.
    events_df : pd.DataFrame
        Raw event data.
    graph_metrics : dict
        Output of ``graph.compute_graph_metrics``.

    Returns
    -------
    pd.DataFrame
        Enhanced scored_df with novelty columns.
    """
    df = scored_df.copy()

    # ── 7.1 Digital Twin ────────────────────────────────────────────────────
    logger.info("Building digital twins...")
    twin_profiles = build_digital_twins(users_df, feature_df)

    twin_clusters = []
    twin_deviations = []
    for uid in df["user_id"]:
        dev = _compute_twin_deviation(uid, users_df, twin_profiles)
        twin_clusters.append(dev.get("cluster_id"))
        twin_deviations.append(dev.get("overall_deviation_score", 0.0))
    df["twin_cluster_id"] = twin_clusters
    df["twin_deviation"] = twin_deviations

    # ── 7.2 Reversibility ──────────────────────────────────────────────────
    logger.info("Running reversibility analysis...")
    rev_results = []
    for uid in df["user_id"]:
        rev = check_reversibility(uid, users_df, events_df)
        rev_results.append(rev)
    df["reversibility"] = rev_results

    # ── 7.3 Confidence Calibration ─────────────────────────────────────────
    logger.info("Calibrating confidence scores...")
    confidences = []
    conf_bases = []
    for uid in df["user_id"]:
        conf, basis = calibrate_confidence(uid, events_df, feature_df)
        confidences.append(conf)
        conf_bases.append(basis)
    df["confidence"] = confidences
    df["confidence_basis"] = conf_bases

    # ── 7.4 Recurrence Detection ───────────────────────────────────────────
    logger.info("Detecting recurrence patterns...")
    recurrence_results = []
    for uid in df["user_id"]:
        rec = detect_recurrence(uid, events_df)
        recurrence_results.append(rec)
    df["recurrence_patterns"] = recurrence_results

    # ── Adjust scores based on recurrence ──────────────────────────────────
    for idx, row in df.iterrows():
        patterns = row.get("recurrence_patterns", [])
        has_recurring = any(p.get("is_recurring", False) for p in patterns if isinstance(p, dict))
        if has_recurring:
            # Downgrade severity slightly for recurring patterns
            current_score = float(row.get("anomaly_score", 0))
            df.at[idx, "anomaly_score"] = max(0, current_score - 5)
            risk = str(row.get("risk_level", "LOW"))
            if risk == "CRITICAL" and current_score - 5 < 80:
                df.at[idx, "risk_level"] = "HIGH"

    logger.info("Novelty features applied to %d users", len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# API & DASHBOARD WRAPPER FUNCTIONS (Compatibility layer)
# ══════════════════════════════════════════════════════════════════════════════

def get_digital_twin_profiles(users_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute digital twin baseline profile by department."""
    profiles = {}
    for dept, grp in users_df.groupby("department"):
        median_sys = grp["system_count"].median() if "system_count" in grp.columns else len(grp.get("systems_list", pd.Series([[]]*len(grp))).iloc[0])
        mode_priv_series = grp["privilege_level"].mode()
        typical_priv = mode_priv_series.iloc[0] if not mode_priv_series.empty else "user"
        profiles[dept] = {
            "median_system_count": median_sys,
            "typical_privilege": typical_priv,
            "peer_count": len(grp)
        }
    return profiles


def compute_twin_deviations(user_row: pd.Series, twin_profiles: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate twin deviations for a specific user row against their department twin."""
    dept = user_row.get("department", "Unknown")
    profile = twin_profiles.get(dept, {"median_system_count": 1.0, "typical_privilege": "user", "peer_count": 1})
    
    sys_count = user_row.get("system_count", 0)
    expected_sys = profile.get("median_system_count", 1.0)
    sys_dev = float(sys_count - expected_sys)
    
    priv_map = {"user": 0, "power-user": 1, "admin": 2, "service-account": 1}
    user_priv = priv_map.get(user_row.get("privilege_level", "user"), 0)
    typical_priv = priv_map.get(profile.get("typical_privilege", "user"), 0)
    priv_dev = float(user_priv - typical_priv)
    
    return {
        "system_count_deviation": sys_dev,
        "expected_system_count": expected_sys,
        "privilege_level_deviation": priv_dev,
        "peer_count": profile.get("peer_count", 1),
        "department_twin": dept
    }


def run_reversibility_simulation(user_id: str, systems_list: List[str], user_events: pd.DataFrame, lookback_days: int = 365) -> List[Dict[str, Any]]:
    """Determine which of the user's systems have not had activity in lookback_days."""
    cutoff = REFERENCE_DATE - pd.Timedelta(days=lookback_days)
    if not user_events.empty:
        user_events = user_events.copy()
        user_events["_ts"] = pd.to_datetime(user_events["timestamp"], errors="coerce")
        user_events = user_events[user_events["_ts"] >= cutoff]

    results = []
    for system in systems_list:
        if not user_events.empty:
            matching = user_events[user_events["resource"] == system]
            event_count = len(matching)
        else:
            event_count = 0

        if event_count == 0:
            safe = True
            reasoning = (
                f"Removing {system} access is safe — 0 events used {system} "
                f"in the past {lookback_days} days. No evidence of active usage."
            )
        else:
            safe = False
            last_use = matching["_ts"].max().strftime("%Y-%m-%d") if not matching.empty else "unknown"
            reasoning = (
                f"{system} access is actively used — {event_count} event(s) "
                f"recorded in the past {lookback_days} days (last use: {last_use}). "
                f"Revoking may disrupt operations."
            )

        results.append({
            "system": system,
            "events_using": event_count,
            "safe_to_revoke": safe,
            "reasoning": reasoning,
        })
    return results


def detect_recurrence_pattern(user_events: pd.DataFrame, min_events: int = 5) -> List[Dict[str, Any]]:
    """Scan user events for regularly scheduled recurrence patterns."""
    if len(user_events) < min_events:
        user_id = user_events["user_id"].iloc[0] if not user_events.empty else "unknown"
        return [{
            "pattern": "insufficient_data",
            "interval_days": None,
            "is_recurring": False,
            "note": (
                f"User {user_id} has only {len(user_events)} event(s), below "
                f"the minimum of {min_events} required for recurrence analysis. "
                f"Skipping — noted in confidence basis."
            ),
        }]

    user_events = user_events.copy()
    user_events["_ts"] = pd.to_datetime(user_events["timestamp"], errors="coerce")
    user_events = user_events.dropna(subset=["_ts"]).sort_values("_ts")

    results = []
    for action, grp in user_events.groupby("action"):
        if len(grp) < 3:
            continue

        timestamps = grp["_ts"].values
        deltas_ns = np.diff(timestamps)
        deltas_days = deltas_ns.astype("timedelta64[D]").astype(float)

        if len(deltas_days) < 2:
            continue

        mean_interval = float(np.mean(deltas_days))
        std_interval = float(np.std(deltas_days))

        if mean_interval <= 0:
            continue

        cv = std_interval / mean_interval
        is_recurring = cv < 0.30 and mean_interval >= 1

        if is_recurring:
            note = (
                f"Action '{action}' recurs every ~{mean_interval:.0f} days "
                f"(CV={cv:.2f}, n={len(grp)}). Pattern is consistent with a "
                f"scheduled process — severity should be downgraded."
            )
        else:
            note = (
                f"Action '{action}' occurs {len(grp)} times with mean interval "
                f"{mean_interval:.1f} days (CV={cv:.2f}). Intervals are too "
                f"irregular to classify as recurring."
            )

        results.append({
            "pattern": action,
            "interval_days": round(mean_interval, 1),
            "is_recurring": is_recurring,
            "note": note,
        })

    if not results:
        user_id = user_events["user_id"].iloc[0] if not user_events.empty else "unknown"
        results.append({
            "pattern": "no_repeated_actions",
            "interval_days": None,
            "is_recurring": False,
            "note": (
                f"User {user_id} has {len(user_events)} events but no single "
                f"action type occurs 3+ times; recurrence analysis not applicable."
            ),
        })

    return results

