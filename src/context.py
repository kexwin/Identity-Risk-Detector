"""
Stage 5: Context & Exception Layer
===================================
Two categories of logic kept explicitly separate:
  1. Statistical Rules — data-derived percentile-based rules
  2. Domain-Policy Exception Rules — 6 hardcoded business rules

Functions
---------
- apply_statistical_rules(scored_df, feature_df, percentile_thresholds)
- apply_exceptions(scored_df, users_df, events_df)
- generate_findings(user_row, feature_row, events_for_user, exceptions,
                    graph_metrics, sod_violations)
"""

from __future__ import annotations

import logging
import os
import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Reference date used throughout the project (matches ingest.py)
# ──────────────────────────────────────────────────────────────────────────────
REFERENCE_DATE = pd.Timestamp("2026-04-20")

# ──────────────────────────────────────────────────────────────────────────────
# Executive identifiers
# ──────────────────────────────────────────────────────────────────────────────
EXEC_TITLES = {"Director", "Executive", "Chief", "VP", "CTO", "CISO", "CFO"}
EXEC_DEPARTMENTS = {"Executive"}

# ──────────────────────────────────────────────────────────────────────────────
# On-call indicative job titles (partial match)
# ──────────────────────────────────────────────────────────────────────────────
ONCALL_TITLE_KEYWORDS = {"Engineer", "DevOps", "SRE"}

# ──────────────────────────────────────────────────────────────────────────────
# Month-end / Quarter-end windows (days before end-of-month considered "window")
# ──────────────────────────────────────────────────────────────────────────────
MONTH_END_WINDOW_DAYS = 5
QUARTER_END_MONTHS = {3, 6, 9, 12}

# ──────────────────────────────────────────────────────────────────────────────
# Sensitivity weights for resources (used in findings)
# ──────────────────────────────────────────────────────────────────────────────
RESOURCE_SENSITIVITY_WEIGHTS: Dict[str, int] = {
    "PROD_DB": 10, "Customer_Vault": 10, "SIEM": 8, "HRIS": 8,
    "ADMIN_SYS": 7, "GL_System": 7, "Admin_Console": 5,
    "Email_Archive": 4, "Data_Lake": 3, "BI_Tool": 2, "File_Share": 1,
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. STATISTICAL RULES (data-derived, percentile-based)
# ══════════════════════════════════════════════════════════════════════════════

def apply_statistical_rules(
    scored_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    percentile_thresholds: Dict[str, float],
) -> Dict[str, List[Dict[str, Any]]]:
    """Apply percentile-based statistical rules and return findings per user.

    Parameters
    ----------
    scored_df : pd.DataFrame
        Output of model.score_users — must contain ``user_id``, ``anomaly_score``,
        ``risk_level``.
    feature_df : pd.DataFrame
        Output of features.engineer_features — 15 feature columns + ``user_id``.
    percentile_thresholds : dict
        Output of features.compute_percentile_thresholds — e.g.
        ``{"system_count_p90": 5, "days_inactive_p90": 52, ...}``.

    Returns
    -------
    dict
        Mapping of ``user_id`` → list of statistical-rule finding dicts.
    """
    findings_map: Dict[str, List[Dict[str, Any]]] = {}
    merged = scored_df.merge(feature_df, on="user_id", how="left", suffixes=("", "_feat"))

    for _, row in merged.iterrows():
        uid = row["user_id"]
        findings: List[Dict[str, Any]] = []

        # --- P90 system count rule ------------------------------------------
        sys_count_p90 = percentile_thresholds.get("system_count_p90", 5)
        sys_count = row.get("system_count", 0)
        if sys_count >= sys_count_p90:
            findings.append({
                "finding": "PRIVILEGE_OUTLIER",
                "details": (
                    f"User {uid} has access to {int(sys_count)} systems, which "
                    f"exceeds the 90th-percentile threshold of {int(sys_count_p90)} "
                    f"systems across the organisation. This breadth of access is "
                    f"statistically anomalous and warrants review."
                ),
                "severity": "MEDIUM",
                "recommendation": (
                    "Conduct a privilege review to determine whether all "
                    f"{int(sys_count)} system entitlements are still business-justified."
                ),
            })

        # --- P90 days-inactive rule ------------------------------------------
        inactive_p90 = percentile_thresholds.get("days_inactive_p90", 50)
        days_inactive = row.get("days_inactive", 0)
        if days_inactive >= inactive_p90:
            priv = row.get("privilege_level", "user")
            sev = "HIGH" if priv in ("admin", "power-user") else "MEDIUM"
            findings.append({
                "finding": "STALE_PRIVILEGED_ACCOUNT",
                "details": (
                    f"User {uid} (privilege_level={priv}) has been inactive for "
                    f"{int(days_inactive)} days, exceeding the 90th-percentile "
                    f"inactivity threshold of {int(inactive_p90)} days. Stale "
                    f"privileged accounts are high-value targets for credential theft."
                ),
                "severity": sev,
                "recommendation": (
                    f"Disable the account if no activity is confirmed within 7 days "
                    f"and escalate to the user's manager for verification."
                ),
            })

        # --- P90 failure rate rule -------------------------------------------
        failure_rate_p90 = percentile_thresholds.get("failure_rate_p90", 0.15)
        failure_rate = row.get("failure_rate", 0.0)
        total_events = row.get("total_events", 0)
        if failure_rate >= failure_rate_p90 and total_events > 0:
            findings.append({
                "finding": "HIGH_FAILURE_RATE",
                "details": (
                    f"User {uid} has a login/action failure rate of "
                    f"{failure_rate:.1%}, exceeding the 90th-percentile threshold "
                    f"of {failure_rate_p90:.1%}. Out of {int(total_events)} total "
                    f"events, approximately {int(total_events * failure_rate)} "
                    f"resulted in failure. This may indicate credential stuffing "
                    f"or brute-force attempts."
                ),
                "severity": "HIGH",
                "recommendation": (
                    "Investigate source IPs, enforce MFA if not already active, "
                    "and correlate with SOC alerts for the user."
                ),
            })

        # --- P90 off-hours activity rule ------------------------------------
        off_hours_p90 = percentile_thresholds.get("off_hours_ratio_p90", 0.5)
        off_hours_ratio = row.get("off_hours_ratio", 0.0)
        if off_hours_ratio >= off_hours_p90 and total_events > 0:
            findings.append({
                "finding": "AFTER_HOURS_ACCESS",
                "details": (
                    f"User {uid} performs {off_hours_ratio:.0%} of their activity "
                    f"outside business hours, exceeding the 90th-percentile "
                    f"threshold of {off_hours_p90:.0%}. This level of off-hours "
                    f"activity is unusual compared to the broader user population."
                ),
                "severity": "MEDIUM",
                "recommendation": (
                    "Verify whether the user has a legitimate on-call or shift "
                    "schedule; if not, investigate the off-hours access pattern."
                ),
            })

        # --- P90 export count rule ------------------------------------------
        export_p90 = percentile_thresholds.get("export_count_p90", 2)
        export_count = row.get("export_count", 0)
        if export_count >= export_p90:
            findings.append({
                "finding": "EXCESSIVE_EXPORTS",
                "details": (
                    f"User {uid} executed {int(export_count)} data export "
                    f"operations, reaching or exceeding the 90th-percentile "
                    f"threshold of {int(export_p90)}. Bulk exports may indicate "
                    f"data exfiltration or policy violations."
                ),
                "severity": "HIGH",
                "recommendation": (
                    "Review exported datasets for sensitive content, restrict "
                    "export permissions if not business-critical, and alert DLP team."
                ),
            })

        # --- P90 cross-department resource access ----------------------------
        cross_dept = row.get("cross_dept_resource_ratio", 0.0)
        cross_dept_p90 = percentile_thresholds.get("cross_dept_resource_ratio_p90", 0.4)
        if cross_dept >= cross_dept_p90 and total_events > 0:
            findings.append({
                "finding": "CROSS_DEPT_ACCESS",
                "details": (
                    f"User {uid} accesses resources outside their department at a "
                    f"rate of {cross_dept:.0%}, above the 90th-percentile threshold "
                    f"of {cross_dept_p90:.0%}. Cross-departmental access may "
                    f"indicate role creep or inappropriate entitlements."
                ),
                "severity": "MEDIUM",
                "recommendation": (
                    "Review the user's resource access list with their department "
                    "head to confirm business justification for cross-departmental "
                    "entitlements."
                ),
            })

        if findings:
            findings_map[uid] = findings

    logger.info(
        "Statistical rules produced findings for %d / %d users",
        len(findings_map), len(merged),
    )
    return findings_map


# ══════════════════════════════════════════════════════════════════════════════
# 2. DOMAIN-POLICY EXCEPTION RULES
# ══════════════════════════════════════════════════════════════════════════════

def _is_exec(row: pd.Series) -> bool:
    """Return True if the user's job_title or department suggests an executive role."""
    title = str(row.get("job_title", ""))
    dept = str(row.get("department", ""))
    if dept in EXEC_DEPARTMENTS:
        return True
    for kw in EXEC_TITLES:
        if kw.lower() in title.lower():
            return True
    return False


def _is_oncall_candidate(row: pd.Series) -> bool:
    """Return True if job_title hints at on-call responsibilities."""
    title = str(row.get("job_title", ""))
    return any(kw.lower() in title.lower() for kw in ONCALL_TITLE_KEYWORDS)


def _in_month_end_window(ts: pd.Timestamp) -> bool:
    """Check whether *ts* falls within the month-end window."""
    try:
        month_end = ts + pd.offsets.MonthEnd(0)
        return (month_end - ts).days <= MONTH_END_WINDOW_DAYS
    except Exception:
        return False


def _in_quarter_end_window(ts: pd.Timestamp) -> bool:
    """Check whether *ts* falls within a quarter-end window."""
    return ts.month in QUARTER_END_MONTHS and _in_month_end_window(ts)


def apply_exceptions(
    scored_df: pd.DataFrame,
    users_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Apply 6 domain-policy exception rules.

    Modifies ``anomaly_score`` and ``risk_level`` where appropriate and
    populates an ``exception_tags`` column (list of applied exception names).

    Parameters
    ----------
    scored_df : pd.DataFrame
        Contains ``user_id``, ``anomaly_score``, ``risk_level``.
    users_df : pd.DataFrame
        Raw user data from ingest (with parsed fields).
    events_df : pd.DataFrame
        Raw event data from ingest.

    Returns
    -------
    pd.DataFrame
        *scored_df* with adjusted ``anomaly_score``, ``risk_level``, added
        ``exception_tags``, and ``exception_findings`` columns.
    """
    df = scored_df.copy()
    df["exception_tags"] = [[] for _ in range(len(df))]
    df["exception_findings"] = [[] for _ in range(len(df))]

    # Load False Positive Feedback records
    feedback_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "feedback.json")
    known_fps = []
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, "r") as f:
                known_fps = json.load(f)
        except Exception as e:
            logger.error("Failed to load feedback.json: %s", e)

    # Pre-compute helpers
    users_lookup = users_df.set_index("user_id") if "user_id" in users_df.columns else users_df

    # Tenure P10
    if "tenure_days" in users_df.columns:
        tenure_p10 = float(np.percentile(users_df["tenure_days"].dropna(), 10))
    elif "hire_date" in users_df.columns:
        tenures = (REFERENCE_DATE - pd.to_datetime(users_df["hire_date"])).dt.days
        tenure_p10 = float(np.percentile(tenures.dropna(), 10))
    else:
        tenure_p10 = 120  # fallback

    # Events grouped by user
    events_by_user = dict(list(events_df.groupby("user_id"))) if not events_df.empty else {}

    # Trailing-90-day cutoff
    trailing_90 = REFERENCE_DATE - pd.Timedelta(days=90)

    for idx, row in df.iterrows():
        uid = row["user_id"]
        tags: List[str] = []
        ex_findings: List[Dict[str, Any]] = []

        # Lookup user row
        try:
            urow = users_lookup.loc[uid]
        except KeyError:
            continue

        user_events = events_by_user.get(uid, pd.DataFrame())
        score = float(row["anomaly_score"])
        risk = str(row["risk_level"])

        # ── RULE 1: EXEC_ROLE_EXCEPTION ─────────────────────────────────────
        if _is_exec(urow):
            score = max(0, score - 20)
            if risk == "CRITICAL":
                risk = "HIGH"
            tags.append("EXEC_ROLE_EXCEPTION")
            ex_findings.append({
                "finding": "EXEC_ROLE_EXCEPTION",
                "details": (
                    f"User {uid} holds an executive-level role "
                    f"(job_title={urow.get('job_title', 'N/A')}, "
                    f"department={urow.get('department', 'N/A')}). "
                    f"Broad access is expected for executive positions. "
                    f"Score reduced by 20 points and CRITICAL downgraded to HIGH "
                    f"if applicable. Manual review recommended to confirm "
                    f"executive scope is current."
                ),
                "severity": "INFO",
                "recommendation": (
                    "Confirm the user's executive role is current via HR records "
                    "and ensure access aligns with their responsibilities."
                ),
            })

        # ── RULE 2: NEW_HIRE_EXCEPTION ──────────────────────────────────────
        if "tenure_days" in urow.index:
            user_tenure = float(urow["tenure_days"])
        elif "hire_date" in urow.index:
            user_tenure = (REFERENCE_DATE - pd.to_datetime(urow["hire_date"])).days
        else:
            user_tenure = 9999

        if user_tenure < tenure_p10:
            score = max(0, score - 15)
            tags.append("NEW_HIRE_EXCEPTION")
            ex_findings.append({
                "finding": "NEW_HIRE_EXCEPTION",
                "details": (
                    f"User {uid} has a tenure of {int(user_tenure)} days, which "
                    f"is below the 10th-percentile threshold of "
                    f"{int(tenure_p10)} days. Elevated activity during "
                    f"onboarding is considered normal. Score reduced by 15 points."
                ),
                "severity": "INFO",
                "recommendation": (
                    "Monitor the account through the standard onboarding review "
                    "period; re-evaluate after 90 days."
                ),
            })

        # ── RULE 3: ONCALL_POSSIBLE ─────────────────────────────────────────
        if _is_oncall_candidate(urow):
            # Check if there's any admin op outside business hours
            if not user_events.empty:
                admin_off = user_events[
                    (user_events["action"] == "admin_operation")
                    & (user_events["time_classification"].isin(["unusual_hours", "night", "weekend"]))
                ]
                if len(admin_off) > 0:
                    tags.append("ONCALL_POSSIBLE")
                    ex_findings.append({
                        "finding": "ONCALL_POSSIBLE",
                        "details": (
                            f"User {uid} (job_title={urow.get('job_title', 'N/A')}) "
                            f"performed {len(admin_off)} admin operation(s) outside "
                            f"business hours. Given their engineering/operations role, "
                            f"this may reflect legitimate on-call duties rather than "
                            f"suspicious activity."
                        ),
                        "severity": "INFO",
                        "recommendation": (
                            "Verify the user's on-call schedule with their team lead; "
                            "if confirmed, whitelist the pattern for future analysis."
                        ),
                    })

        # ── RULE 4: SEASONAL_FINANCE ────────────────────────────────────────
        dept = str(urow.get("department", ""))
        if dept == "Finance" and not user_events.empty:
            ts_col = pd.to_datetime(user_events["timestamp"], errors="coerce")
            in_window = ts_col.apply(
                lambda t: _in_month_end_window(t) if pd.notna(t) else False
            )
            if in_window.any():
                tags.append("SEASONAL_FINANCE")
                # We note it; the 30 % weight reduction is applied at event-scoring
                # level — here we record the tag so downstream can act on it.
                window_count = int(in_window.sum())
                ex_findings.append({
                    "finding": "SEASONAL_FINANCE",
                    "details": (
                        f"User {uid} is in the Finance department and "
                        f"{window_count} of their events fall within a "
                        f"month-end / quarter-end window. Elevated activity "
                        f"during close periods is expected. Event-anomaly "
                        f"weight reduced by 30 %."
                    ),
                    "severity": "INFO",
                    "recommendation": (
                        "No immediate action; continue monitoring outside "
                        "close windows for anomalous behaviour."
                    ),
                })

        # ── RULE 5: SVC_ACTIVE_BASELINE ─────────────────────────────────────
        priv_level = str(urow.get("privilege_level", ""))
        if priv_level == "service-account":
            if not user_events.empty:
                ts_col = pd.to_datetime(user_events["timestamp"], errors="coerce")
                recent = ts_col[ts_col >= trailing_90]
                if len(recent) > 0:
                    tags.append("SVC_ACTIVE_BASELINE")
                    ex_findings.append({
                        "finding": "SVC_ACTIVE_BASELINE",
                        "details": (
                            f"Service account {uid} has {len(recent)} event(s) in "
                            f"the trailing 90-day window (since "
                            f"{trailing_90.strftime('%Y-%m-%d')}), confirming it "
                            f"is actively used. It should not be flagged as "
                            f"orphaned."
                        ),
                        "severity": "INFO",
                        "recommendation": (
                            "Maintain current monitoring; schedule next service-"
                            "account review in 90 days."
                        ),
                    })

        # ── RULE 6: CONTRACTOR_NORM ─────────────────────────────────────────
        email = str(urow.get("email", ""))
        title = str(urow.get("job_title", ""))
        is_contractor = (
            not email.endswith("@company.com")
            or "contractor" in title.lower()
            or "vendor" in title.lower()
        )
        if is_contractor:
            # Apply a more aggressive stale-account threshold for contractors
            contractor_stale_threshold = 21  # days (stricter than default)
            user_inactive = int(urow.get("days_inactive", 0))
            tags.append("CONTRACTOR_NORM")
            if user_inactive > contractor_stale_threshold:
                ex_findings.append({
                    "finding": "CONTRACTOR_NORM",
                    "details": (
                        f"User {uid} (email={email}, job_title={title}) is "
                        f"identified as a contractor/vendor. They have been "
                        f"inactive for {user_inactive} days, exceeding the "
                        f"contractor stale threshold of "
                        f"{contractor_stale_threshold} days."
                    ),
                    "severity": "MEDIUM",
                    "recommendation": (
                        "Verify contractor engagement status with procurement; "
                        "disable account if engagement has ended."
                    ),
                })
            else:
                ex_findings.append({
                    "finding": "CONTRACTOR_NORM",
                    "details": (
                        f"User {uid} is identified as a contractor/vendor. "
                        f"A separate, stricter stale threshold of "
                        f"{contractor_stale_threshold} days applies. "
                        f"Current inactivity: {user_inactive} days — within "
                        f"acceptable range."
                    ),
                    "severity": "INFO",
                    "recommendation": (
                        "No action required; re-evaluate at next contractor "
                        "review cycle."
                    ),
                })

        # ── RULE 7: SIMILAR_TO_KNOWN_FP (Feedback Loop) ─────────────────────
        if known_fps:
            fp_feature_cols = [
                "days_inactive", "system_count", "recent_event_count", 
                "after_hours_event_ratio", "high_sensitivity_export_count", 
                "admin_op_off_hours_count", "failure_rate"
            ]
            # Use precalculated dataset means and stds to standardize features
            # to prevent larger columns like days_inactive from dominating similarity
            means = scored_df[fp_feature_cols].mean()
            stds = scored_df[fp_feature_cols].std().replace(0, 1.0)
            
            u_vals = []
            for col in fp_feature_cols:
                val = 0.0
                if col in row:
                    val = float(row[col])
                elif col in urow:
                    val = float(urow[col])
                u_vals.append((val - means[col]) / stds[col])
            u_vec = np.array(u_vals)
            
            is_fp_match = False
            max_sim = 0.0
            for fp in known_fps:
                fp_features = fp.get("features", {})
                fp_vals = []
                for col in fp_feature_cols:
                    val = float(fp_features.get(col, 0.0))
                    fp_vals.append((val - means[col]) / stds[col])
                fp_vec = np.array(fp_vals)
                
                dot_product = np.dot(u_vec, fp_vec)
                norm_u = np.linalg.norm(u_vec)
                norm_fp = np.linalg.norm(fp_vec)
                if norm_u > 0 and norm_fp > 0:
                    sim = dot_product / (norm_u * norm_fp)
                    if sim > max_sim:
                        max_sim = sim
                    if sim >= 0.90:
                        is_fp_match = True
            
            if is_fp_match:
                score = max(0.0, score - 20)
                if risk == "CRITICAL":
                    risk = "HIGH"
                elif risk == "HIGH":
                    risk = "MEDIUM"
                elif risk == "MEDIUM":
                    risk = "LOW"
                tags.append("SIMILAR_TO_KNOWN_FP")
                ex_findings.append({
                    "finding": "SIMILAR_TO_KNOWN_FP",
                    "details": (
                        f"User {uid} exhibits a behavior vector similar to a known "
                        f"False Positive in the feedback database (max cosine similarity = {max_sim:.2f}). "
                        f"Anomaly score reduced by 20 points and risk level downgraded."
                    ),
                    "severity": "INFO",
                    "recommendation": "Monitor for any new/distinct access patterns outside of this baseline.",
                })

        # ── Write back ──────────────────────────────────────────────────────
        df.at[idx, "anomaly_score"] = max(0.0, score)
        df.at[idx, "risk_level"] = risk
        df.at[idx, "exception_tags"] = tags
        df.at[idx, "exception_findings"] = ex_findings

    logger.info(
        "Exception rules applied. Tags distribution: %s",
        df["exception_tags"].explode().value_counts().to_dict(),
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. FINDINGS GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_findings(
    user_row: pd.Series,
    feature_row: pd.Series,
    events_for_user: pd.DataFrame,
    exceptions: List[str],
    graph_metrics: Dict[str, Any],
    sod_violations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Generate a comprehensive list of finding dicts for a single user.

    Each finding follows the schema::

        {
            "finding": "FINDING_TYPE",
            "details": ">=100 chars, citing specific numbers",
            "severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
            "recommendation": "Actionable remediation step"
        }

    Parameters
    ----------
    user_row : pd.Series
        Single row from ``users_df`` (indexed by user_id or with user_id field).
    feature_row : pd.Series
        Corresponding row from ``feature_df``.
    events_for_user : pd.DataFrame
        Subset of ``events_df`` for this user.
    exceptions : list[str]
        Exception tags already applied to this user.
    graph_metrics : dict
        Output of ``graph.compute_graph_metrics`` — contains ``blast_radius``,
        ``shared_system_users``.
    sod_violations : list[dict]
        Output of ``graph.detect_sod_violations``.

    Returns
    -------
    list[dict]
        List of finding dicts.
    """
    uid = user_row.get("user_id", user_row.name if hasattr(user_row, "name") else "UNKNOWN")
    findings: List[Dict[str, Any]] = []

    priv = str(user_row.get("privilege_level", "user"))
    days_inactive = int(user_row.get("days_inactive", 0))
    sys_count = int(user_row.get("system_count", len(str(user_row.get("systems_access", "")).split("|"))))
    total_events = int(feature_row.get("total_events", 0))

    # ── STALE PRIVILEGED ACCOUNT ────────────────────────────────────────────
    stale_threshold = 21 if "CONTRACTOR_NORM" in exceptions else 45
    is_svc = priv == "service-account"
    skip_stale_svc = is_svc and "SVC_ACTIVE_BASELINE" in exceptions

    if days_inactive >= stale_threshold and not skip_stale_svc:
        sev = "CRITICAL" if priv in ("admin", "service-account") else "HIGH"
        if "EXEC_ROLE_EXCEPTION" in exceptions and sev == "CRITICAL":
            sev = "HIGH"
        findings.append({
            "finding": "STALE_PRIVILEGED_ACCOUNT",
            "details": (
                f"User {uid} (privilege_level={priv}) has not logged in for "
                f"{days_inactive} days (last_login={user_row.get('last_login', 'N/A')}). "
                f"The applicable stale threshold is {stale_threshold} days. "
                f"The account retains access to {sys_count} system(s). "
                f"Stale accounts with elevated privileges are prime targets for "
                f"credential compromise and lateral movement."
            ),
            "severity": sev,
            "recommendation": (
                f"Immediately disable the account and revoke all {sys_count} "
                f"system entitlements. Verify employment/engagement status with HR."
            ),
        })

    # ── BROAD PRIVILEGE SCOPE ───────────────────────────────────────────────
    if sys_count >= 5:
        sev = "HIGH" if priv in ("admin", "power-user") else "MEDIUM"
        peer_median = float(feature_row.get("peer_median_system_count", 2))
        findings.append({
            "finding": "BROAD_PRIVILEGE_SCOPE",
            "details": (
                f"User {uid} has access to {sys_count} systems, compared to a "
                f"peer-group median of {peer_median:.1f} systems. This represents "
                f"a {((sys_count / max(peer_median, 1)) - 1) * 100:.0f}% excess "
                f"over the peer baseline. Excessive breadth increases the blast "
                f"radius in a credential compromise scenario."
            ),
            "severity": sev,
            "recommendation": (
                f"Review all {sys_count} system entitlements with the user's "
                f"manager; remove access not justified by current role."
            ),
        })

    # ── ORPHANED SERVICE ACCOUNT ────────────────────────────────────────────
    if is_svc and not skip_stale_svc:
        if total_events == 0 or days_inactive >= 30:
            findings.append({
                "finding": "ORPHANED_SERVICE_ACCOUNT",
                "details": (
                    f"Service account {uid} shows {total_events} event(s) in the "
                    f"observed period and has been inactive for {days_inactive} "
                    f"days. With access to {sys_count} system(s), an unmonitored "
                    f"service account represents a significant unmanaged risk vector."
                ),
                "severity": "CRITICAL",
                "recommendation": (
                    "Identify the service account owner, rotate credentials "
                    "immediately, and disable the account if no owner is found "
                    "within 48 hours."
                ),
            })

    # ── SOD VIOLATIONS ──────────────────────────────────────────────────────
    user_sod = [v for v in sod_violations if v.get("user_id") == uid]
    for violation in user_sod:
        findings.append({
            "finding": "SOD_VIOLATION",
            "details": (
                f"User {uid} holds conflicting entitlements: "
                f"{violation.get('system_a', 'N/A')} and "
                f"{violation.get('system_b', 'N/A')}. "
                f"Rule: {violation.get('rule', 'N/A')}. "
                f"Separation-of-duties requires these capabilities be held by "
                f"different individuals to prevent fraud and errors. "
                f"This violation was detected via graph-based entitlement analysis."
            ),
            "severity": "HIGH",
            "recommendation": (
                f"Remove one of the conflicting entitlements or implement a "
                f"compensating control (dual approval) for operations on "
                f"{violation.get('system_a', 'N/A')} and "
                f"{violation.get('system_b', 'N/A')}."
            ),
        })

    # ── AFTER-HOURS ACCESS ──────────────────────────────────────────────────
    if not events_for_user.empty:
        off_hours = events_for_user[
            events_for_user["time_classification"].isin(["unusual_hours", "night", "weekend"])
        ]
        if len(off_hours) >= 2:
            oh_ratio = len(off_hours) / len(events_for_user)
            sev = "HIGH" if oh_ratio > 0.5 else "MEDIUM"
            if "ONCALL_POSSIBLE" in exceptions:
                sev = "LOW"
            findings.append({
                "finding": "AFTER_HOURS_ACCESS",
                "details": (
                    f"User {uid} has {len(off_hours)} out of {len(events_for_user)} "
                    f"events ({oh_ratio:.0%}) occurring outside business hours. "
                    f"Off-hours sessions included actions: "
                    f"{', '.join(off_hours['action'].unique())}. "
                    f"Resources accessed off-hours: "
                    f"{', '.join(off_hours['resource'].unique())}."
                    + (" [ONCALL_POSSIBLE exception applied — severity reduced.]"
                       if "ONCALL_POSSIBLE" in exceptions else "")
                ),
                "severity": sev,
                "recommendation": (
                    "Cross-reference with on-call schedules and VPN logs; "
                    "if the pattern is unexplained, escalate to SOC."
                ),
            })

    # ── EXCESSIVE EXPORTS ───────────────────────────────────────────────────
    if not events_for_user.empty:
        exports = events_for_user[events_for_user["action"] == "export_data"]
        if len(exports) >= 2:
            resources_exported = ", ".join(exports["resource"].unique())
            findings.append({
                "finding": "EXCESSIVE_EXPORTS",
                "details": (
                    f"User {uid} performed {len(exports)} data export operations "
                    f"across resources: {resources_exported}. Repeated bulk exports "
                    f"may indicate data exfiltration, especially when targeting "
                    f"high-sensitivity systems."
                ),
                "severity": "HIGH",
                "recommendation": (
                    f"Inspect export payloads for sensitive data via DLP; "
                    f"restrict export capabilities to business-justified users."
                ),
            })

    # ── BLAST RADIUS (from graph) ───────────────────────────────────────────
    blast = graph_metrics.get("blast_radius", {}).get(uid, 0)
    if blast >= 10:
        findings.append({
            "finding": "HIGH_BLAST_RADIUS",
            "details": (
                f"User {uid} has a blast radius of {blast} in the privilege "
                f"graph, meaning a compromise of this account could directly "
                f"or indirectly expose {blast} other users/systems. This is "
                f"significantly above the median blast radius."
            ),
            "severity": "HIGH",
            "recommendation": (
                "Implement just-in-time access controls, require MFA for "
                "all privileged operations, and reduce standing access."
            ),
        })

    return findings
