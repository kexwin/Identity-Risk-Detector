"""
Stage 4: ML Anomaly Scoring
=============================
Two Isolation-Forest models:
  * **User-level** (contamination=0.16) – scores each user 0-100.
  * **Event-level** (contamination=0.41) – scores individual events.

Risk classification (CRITICAL / HIGH / MEDIUM / LOW) is
percentile-derived from the user anomaly scores.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Module-level state (populated after fitting)
# ---------------------------------------------------------------------------
_model_stats: Dict[str, Any] = {}


# ===================================================================
# 1. User-level scoring
# ===================================================================

# Features used by the user-level Isolation Forest
_USER_ML_FEATURES = [
    "days_inactive",
    "is_stale_admin",
    "is_stale_power_user",
    "system_count",
    "system_count_vs_dept_median",
    "has_sensitive_system",
    "privilege_vs_dept_norm",
    "is_service_account_no_owner",
    "recent_event_count",
    "after_hours_event_ratio",
    "high_sensitivity_export_count",
    "admin_op_off_hours_count",
    "failure_rate",
    "tenure_days",
    "is_new_hire",
    "blast_radius",
    "shared_system_users",
    "flagged_event_count",
]


def score_users(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Score every user with an Isolation-Forest anomaly model.

    Parameters
    ----------
    feature_df : pd.DataFrame
        Output of ``features.engineer_features`` merged with `aggregate_event_flags`.
        Must contain every column listed in ``_USER_ML_FEATURES`` plus ``user_id``,
        ``privilege_level``, ``has_sensitive_system``.

    Returns
    -------
    pd.DataFrame
        Copy of *feature_df* with added columns:

        * ``anomaly_score`` – float in [0, 100].
        * ``risk_level`` – one of CRITICAL / HIGH / MEDIUM / LOW.
        * ``ml_features`` – pipe-delimited list of features used.
    """
    global _model_stats

    result = feature_df.copy()

    # Build feature matrix
    available = [c for c in _USER_ML_FEATURES if c in result.columns]
    X = result[available].copy().fillna(0).astype(float)

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Isolation Forest
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.16,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    # Raw scores: sklearn returns negative-anomaly-score (lower = more anomalous)
    raw_scores = iso.decision_function(X_scaled)

    # Scale to 0-100 (higher = more anomalous)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    if max_s - min_s > 0:
        scaled = (1 - (raw_scores - min_s) / (max_s - min_s)) * 100
    else:
        scaled = np.full_like(raw_scores, 50.0)

    result["anomaly_score"] = np.round(scaled, 2)

    # Classify risk levels
    result["risk_level"] = classify_risk_level(result["anomaly_score"], result)

    # Record which features were used
    result["ml_features"] = "|".join(available)

    # Compute and store model stats
    _model_stats["user_model"] = {
        "features_used": available,
        "n_estimators": 200,
        "contamination": 0.16,
        "P95_score": float(np.percentile(result["anomaly_score"], 95)),
        "P84_score": float(np.percentile(result["anomaly_score"], 84)),
        "P70_score": float(np.percentile(result["anomaly_score"], 70)),
        "score_min": float(result["anomaly_score"].min()),
        "score_max": float(result["anomaly_score"].max()),
        "score_mean": float(result["anomaly_score"].mean()),
    }

    # Feature importances (approximated via isolation depth)
    importances = _approximate_feature_importance(iso, X_scaled, available)
    _model_stats["user_model"]["feature_importances"] = importances

    _print_user_scoring_summary(result)

    return result


def _approximate_feature_importance(
    model: IsolationForest,
    X: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """Estimate feature importance by measuring score shift on permutation."""
    base_scores = model.decision_function(X)
    importances: Dict[str, float] = {}
    rng = np.random.RandomState(42)

    for i, fname in enumerate(feature_names):
        X_perm = X.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        perm_scores = model.decision_function(X_perm)
        importances[fname] = float(np.mean(np.abs(base_scores - perm_scores)))

    # Normalise so they sum to 1
    total = sum(importances.values())
    if total > 0:
        importances = {k: round(v / total, 4) for k, v in importances.items()}

    return importances


# ===================================================================
# 2. Event-level scoring
# ===================================================================
def score_events(
    events_df: pd.DataFrame, users_df: pd.DataFrame
) -> pd.DataFrame:
    """Score each event with a separate Isolation Forest.

    Parameters
    ----------
    events_df : pd.DataFrame
        Normalised event data.
    users_df : pd.DataFrame
        User data (for privilege-level lookup).

    Returns
    -------
    pd.DataFrame
        Copy of *events_df* with added column
        ``event_anomaly_score`` (float 0-100) and ``weighted_anomaly``.
    """
    global _model_stats

    result = events_df.copy()

    if len(result) == 0:
        result["event_anomaly_score"] = np.nan
        result["weighted_anomaly"] = np.nan
        return result

    # Encode features
    result["hour_of_day"] = result["timestamp"].dt.hour
    result["day_of_week"] = result["timestamp"].dt.dayofweek

    # Resource sensitivity encoding
    sens_map = {"low": 0, "medium": 1, "high": 2}
    result["resource_sensitivity_encoded"] = (
        result["resource_sensitivity"].map(sens_map).fillna(0).astype(int)
    )

    # Action encoding
    action_le = LabelEncoder()
    result["action_encoded"] = action_le.fit_transform(
        result["action"].fillna("unknown")
    )

    # Status encoding
    result["status_encoded"] = (result["status"] == "failure").astype(int)

    # High-value resource flag
    high_value = {"PROD_DB", "ADMIN_SYS", "SIEM", "Customer_Vault", "HRIS"}
    result["is_high_value_resource"] = (
        result["resource"].isin(high_value).astype(int)
    )

    event_features = [
        "hour_of_day",
        "day_of_week",
        "resource_sensitivity_encoded",
        "action_encoded",
        "status_encoded",
        "is_high_value_resource",
    ]

    X = result[event_features].fillna(0).astype(float).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(
        n_estimators=150,
        contamination=0.41,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_scaled)

    raw = iso.decision_function(X_scaled)
    min_s, max_s = raw.min(), raw.max()
    if max_s - min_s > 0:
        scaled = (1 - (raw - min_s) / (max_s - min_s)) * 100
    else:
        scaled = np.full_like(raw, 50.0)

    result["event_anomaly_score"] = np.round(scaled, 2)

    # ------------------------------------------------------------------
    # Finance Seasonal exception: Reduce event-anomaly weight 30% during month-end window
    # ------------------------------------------------------------------
    users_lookup = users_df[["user_id", "department"]].drop_duplicates()
    events_with_dept = result.merge(users_lookup, on="user_id", how="left")
    
    # Month-end / close window: day is in [28, 29, 30, 31, 1, 2, 3]
    events_with_dept["is_month_end"] = events_with_dept["timestamp"].dt.day.isin([28, 29, 30, 31, 1, 2, 3])
    
    events_with_dept["is_anomaly"] = (events_with_dept["event_anomaly_score"] >= 50.0).astype(int)
    
    def get_event_anomaly_weight(row):
        if row["department"] == "Finance" and row["is_month_end"] and row["is_anomaly"] == 1:
            return 0.7
        return float(row["is_anomaly"])

    result["weighted_anomaly"] = events_with_dept.apply(get_event_anomaly_weight, axis=1)

    # Store stats
    _model_stats["event_model"] = {
        "features_used": event_features,
        "n_estimators": 150,
        "contamination": 0.41,
        "score_min": float(result["event_anomaly_score"].min()),
        "score_max": float(result["event_anomaly_score"].max()),
        "score_mean": float(result["event_anomaly_score"].mean()),
        "flagged_events": int((result["event_anomaly_score"] >= 50).sum()),
    }

    flagged = (result["event_anomaly_score"] >= 50).sum()
    print(f"[Model]  Event scoring: {len(result)} events, "
          f"{flagged} flagged (score>=50)")

    return result


# ===================================================================
# 3. Risk classification
# ===================================================================
def classify_risk_level(
    anomaly_scores: pd.Series, feature_df: pd.DataFrame
) -> pd.Series:
    """Assign a risk level to each user based on anomaly score + context."""
    p95 = np.percentile(anomaly_scores, 95)
    p84 = np.percentile(anomaly_scores, 84)
    p70 = np.percentile(anomaly_scores, 70)

    elevated_priv = feature_df["privilege_level"].isin(["admin", "power-user"])
    sensitive = feature_df.get("has_sensitive_system", pd.Series(0, index=feature_df.index)).astype(bool)

    # Individual statistical rules that push to MEDIUM
    rule_fires = (
        feature_df.get("is_stale_admin", pd.Series(0, index=feature_df.index)).astype(bool)
        | feature_df.get("is_stale_power_user", pd.Series(0, index=feature_df.index)).astype(bool)
        | feature_df.get("is_service_account_no_owner", pd.Series(0, index=feature_df.index)).astype(bool)
    )

    levels = pd.Series("LOW", index=anomaly_scores.index)

    # MEDIUM
    levels[(anomaly_scores >= p70) | rule_fires] = "MEDIUM"

    # HIGH
    levels[anomaly_scores >= p84] = "HIGH"

    # CRITICAL
    levels[(anomaly_scores >= p95) & (elevated_priv | sensitive)] = "CRITICAL"

    return levels


# ===================================================================
# 4. Aggregate flagged events back to user level
# ===================================================================
def aggregate_event_flags(
    scored_events: pd.DataFrame,
    user_scores: pd.DataFrame,
    threshold: float = 50.0,
) -> pd.DataFrame:
    """Merge per-user weighted flagged-event counts back into user scores.

    Parameters
    ----------
    scored_events : pd.DataFrame
        Events with ``event_anomaly_score``.
    user_scores : pd.DataFrame
        User-level scored data.
    threshold : float, default 50.0
        Events at or above this score are considered "flagged".

    Returns
    -------
    pd.DataFrame
        *user_scores* with an added ``flagged_event_count`` column.
    """
    if "weighted_anomaly" in scored_events.columns:
        counts = scored_events.groupby("user_id")["weighted_anomaly"].sum().rename("flagged_event_count")
    else:
        flagged = scored_events[scored_events["event_anomaly_score"] >= threshold]
        counts = flagged.groupby("user_id").size().rename("flagged_event_count")

    result = user_scores.merge(counts, on="user_id", how="left")
    result["flagged_event_count"] = result["flagged_event_count"].fillna(0.0)

    print(f"[Model]  Users with flagged events (weighted): "
          f"{(result['flagged_event_count'] > 0).sum()} / {len(result)}")
    return result


# ===================================================================
# 5. Model stats accessor
# ===================================================================
def get_model_stats() -> Dict[str, Any]:
    """Return diagnostics from the most recent model fits."""
    return _model_stats.copy()


# ---------------------------------------------------------------------------
# Internal summary printer
# ---------------------------------------------------------------------------
def _print_user_scoring_summary(result: pd.DataFrame) -> None:
    """Log a compact summary of user-level scoring."""
    print("=" * 60)
    print("  ANOMALY SCORING SUMMARY  -  Stage 4")
    print("=" * 60)

    risk_counts = result["risk_level"].value_counts()
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        cnt = risk_counts.get(level, 0)
        print(f"  {level:10s}: {cnt:4d}  ({cnt / len(result) * 100:5.1f}%)")

    print(f"\n  Score statistics:")
    print(f"    min   = {result['anomaly_score'].min():.1f}")
    print(f"    P50   = {result['anomaly_score'].median():.1f}")
    print(f"    P84   = {np.percentile(result['anomaly_score'], 84):.1f}")
    print(f"    P95   = {np.percentile(result['anomaly_score'], 95):.1f}")
    print(f"    max   = {result['anomaly_score'].max():.1f}")
    print(f"    mean  = {result['anomaly_score'].mean():.1f}")
    print(f"    stdev = {result['anomaly_score'].std():.1f}")

    # Top-5 riskiest users
    top5 = result.nlargest(5, "anomaly_score")[
        ["user_id", "anomaly_score", "risk_level", "privilege_level"]
    ]
    print(f"\n  Top 5 riskiest users:")
    for _, row in top5.iterrows():
        print(
            f"    {row['user_id']}  score={row['anomaly_score']:.1f}  "
            f"risk={row['risk_level']}  priv={row['privilege_level']}"
        )
    print("=" * 60)
