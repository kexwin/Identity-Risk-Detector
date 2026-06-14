"""
Bonus S3: Behavioral Clustering
================================
KMeans clustering on the full feature vector to group users into
behavioural archetypes.  Builds on top of the Digital Twin (Section 7.1)
by clustering the normal population first.

Functions
---------
- cluster_users(feature_df, scored_df, n_clusters=5) → cluster_df
- get_cluster_summary() → dict
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Module-level cache for the last computed cluster summary
_CLUSTER_SUMMARY_CACHE: Dict[str, Any] = {}


# ──────────────────────────────────────────────────────────────────────────────
# Cluster label heuristics
# ──────────────────────────────────────────────────────────────────────────────

def _infer_cluster_label(profile: Dict[str, Any]) -> str:
    """Infer a human-readable cluster label from its statistical profile.

    Parameters
    ----------
    profile : dict
        Must contain ``mean_anomaly_score``, ``mean_system_count``,
        ``dominant_privilege``, ``mean_days_inactive``, ``mean_total_events``.

    Returns
    -------
    str
        Descriptive label like "Low-activity admins" or "Active power-users".
    """
    priv = str(profile.get("dominant_privilege", "user"))
    score = float(profile.get("mean_anomaly_score", 0))
    sys_count = float(profile.get("mean_system_count", 0))
    events = float(profile.get("mean_total_events", 0))
    inactive = float(profile.get("mean_days_inactive", 0))

    # Activity level
    if events < 1:
        activity = "Zero-event"
    elif events < 3:
        activity = "Low-activity"
    elif events < 6:
        activity = "Moderate-activity"
    else:
        activity = "Active"

    # Risk modifier
    if score >= 70:
        risk_mod = "high-risk"
    elif score >= 40:
        risk_mod = "elevated-risk"
    else:
        risk_mod = ""

    # Privilege label
    priv_label_map = {
        "admin": "admins",
        "power-user": "power-users",
        "service-account": "service accounts",
        "user": "standard users",
    }
    priv_label = priv_label_map.get(priv, "users")

    # Staleness modifier
    stale_mod = "stale " if inactive > 40 else ""

    # Build label
    parts = [activity, stale_mod + priv_label]
    if risk_mod:
        parts.insert(1, risk_mod)

    label = " ".join(p for p in parts if p).strip()
    return label.capitalize() if label else "General users"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CLUSTERING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def cluster_users(
    feature_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    n_clusters: int = 5,
    users_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Cluster users based on their behavioural feature vectors.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Engineered features from ``features.engineer_features``.
        Must include ``user_id`` and numeric feature columns.
    scored_df : pd.DataFrame
        Output of ``model.score_users`` — must contain ``user_id``,
        ``anomaly_score``, ``risk_level``.
    n_clusters : int, optional
        Number of KMeans clusters (default 5).
    users_df : pd.DataFrame, optional
        User data for enriching profiles with department/privilege info.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``user_id``, ``cluster_id``, ``cluster_label``,
        ``cluster_profile`` (dict).
    """
    global _CLUSTER_SUMMARY_CACHE

    # Ensure user_id is a column in feature_df
    if "user_id" not in feature_df.columns:
        if feature_df.index.name == "user_id":
            feature_df = feature_df.reset_index()
        else:
            feature_df = feature_df.copy()
            feature_df["user_id"] = feature_df.index

    # Ensure user_id is a column in scored_df
    if "user_id" not in scored_df.columns:
        if scored_df.index.name == "user_id":
            scored_df = scored_df.reset_index()
        else:
            scored_df = scored_df.copy()
            scored_df["user_id"] = scored_df.index

    # Merge features and scores
    merged = feature_df.merge(scored_df[["user_id", "anomaly_score", "risk_level"]],
                              on="user_id", how="left")

    # Select numeric columns for clustering (exclude user_id)
    numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    # Remove columns that aren't meaningful features
    exclude = {"user_id"}
    feature_cols = [c for c in numeric_cols if c not in exclude]

    if not feature_cols:
        logger.warning("No numeric feature columns found for clustering.")
        result = pd.DataFrame({
            "user_id": merged["user_id"],
            "cluster_id": 0,
            "cluster_label": "Unclustered",
            "cluster_profile": [{}] * len(merged),
        })
        return result

    X = merged[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine actual k (cannot exceed sample count)
    actual_k = min(n_clusters, len(merged))
    if actual_k < 2:
        actual_k = 2  # minimum for meaningful clustering

    kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
    merged["cluster_id"] = kmeans.fit_predict(X_scaled)

    # If users_df is provided, merge for enrichment
    if users_df is not None:
        enrich_cols = ["user_id"]
        for col in ["department", "job_title", "privilege_level",
                     "system_count", "days_inactive", "systems_access"]:
            if col in users_df.columns and col not in merged.columns:
                enrich_cols.append(col)
        if len(enrich_cols) > 1:
            merged = merged.merge(users_df[enrich_cols], on="user_id", how="left")

    # Build profiles per cluster
    profiles: Dict[int, Dict[str, Any]] = {}
    for cid in range(actual_k):
        cmask = merged["cluster_id"] == cid
        cluster = merged[cmask]

        # Dominant privilege level
        if "privilege_level" in cluster.columns:
            priv_mode = cluster["privilege_level"].mode()
            dominant_priv = priv_mode.iloc[0] if not priv_mode.empty else "user"
        else:
            dominant_priv = "user"

        # Departments
        if "department" in cluster.columns:
            departments = sorted(cluster["department"].dropna().unique().tolist())
        else:
            departments = []

        profile = {
            "cluster_id": int(cid),
            "user_count": int(len(cluster)),
            "mean_anomaly_score": round(float(cluster["anomaly_score"].mean()), 1) if "anomaly_score" in cluster.columns else 0,
            "mean_system_count": round(float(cluster.get("system_count", pd.Series([0])).mean()), 1),
            "mean_days_inactive": round(float(cluster.get("days_inactive", pd.Series([0])).mean()), 1),
            "mean_total_events": round(float(cluster.get("total_events", pd.Series([0])).mean()), 1),
            "dominant_privilege": dominant_priv,
            "departments": departments,
            "risk_distribution": (
                cluster["risk_level"].value_counts().to_dict()
                if "risk_level" in cluster.columns else {}
            ),
        }
        profile["cluster_label"] = _infer_cluster_label(profile)
        profiles[cid] = profile

    # Assign labels and profiles to DataFrame
    merged["cluster_label"] = merged["cluster_id"].map(
        lambda cid: profiles.get(cid, {}).get("cluster_label", "Unknown")
    )
    merged["cluster_profile"] = merged["cluster_id"].map(
        lambda cid: profiles.get(cid, {})
    )

    # Cache the summary
    _CLUSTER_SUMMARY_CACHE = {
        "n_clusters": actual_k,
        "total_users": len(merged),
        "profiles": profiles,
        "feature_columns_used": feature_cols,
        "inertia": float(kmeans.inertia_),
    }

    result = merged[["user_id", "cluster_id", "cluster_label", "cluster_profile"]].copy()

    logger.info(
        "Behavioural clustering complete: %d clusters, %d users",
        actual_k, len(result),
    )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# CLUSTER SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def get_cluster_summary() -> Dict[str, Any]:
    """Return a summary dict describing each cluster from the last run.

    Returns
    -------
    dict
        Keys: ``n_clusters``, ``total_users``, ``profiles`` (dict of
        cluster_id → profile), ``feature_columns_used``, ``inertia``.
        Returns empty dict if ``cluster_users`` has not been called yet.
    """
    if not _CLUSTER_SUMMARY_CACHE:
        logger.warning("No cluster summary available — run cluster_users() first.")
    return _CLUSTER_SUMMARY_CACHE.copy()
