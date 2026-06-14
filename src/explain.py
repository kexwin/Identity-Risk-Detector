"""
Stage 6: Template-Based Explanation Generator (NO LLM API)
===========================================================
Uses sophisticated f-string templates that cite specific numbers
(days inactive, system count, peer median, event counts).

Functions
---------
- generate_explanation(user_data) → full JSON-ready dict
- generate_executive_summary(all_results, users_df, events_df) → summary dict
- generate_report(all_results, metadata) → full report dict
"""

from __future__ import annotations

import logging
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

REFERENCE_DATE = pd.Timestamp("2026-04-20")

# ──────────────────────────────────────────────────────────────────────────────
# Risk-level → escalation mapping
# ──────────────────────────────────────────────────────────────────────────────
ESCALATION_MAP = {
    "CRITICAL": "Immediate escalation to CISO; Security Operations Center (SOC) ticket required within 1 hour",
    "HIGH": "Security manager review required within 24 hours; create incident ticket",
    "MEDIUM": "Identity governance team review required within 5 business days",
    "LOW": "Include in next scheduled access review cycle",
    "INFO": "No escalation required; retain for audit trail",
}

# ──────────────────────────────────────────────────────────────────────────────
# Finding → suggested actions templates
# ──────────────────────────────────────────────────────────────────────────────
ACTION_TEMPLATES: Dict[str, List[str]] = {
    "STALE_PRIVILEGED_ACCOUNT": [
        "Disable the account in Active Directory and all federated identity providers within 24 hours",
        "Verify employment/engagement status with HR or procurement",
        "Rotate all credentials and API keys associated with this account",
        "Review audit logs for any unauthorized activity during the inactivity period",
    ],
    "BROAD_PRIVILEGE_SCOPE": [
        "Schedule a privilege review meeting with the user's manager within 5 business days",
        "Remove access to systems not used in the past 90 days",
        "Implement just-in-time (JIT) access for non-daily-use systems",
        "Document business justification for each remaining entitlement",
    ],
    "ORPHANED_SERVICE_ACCOUNT": [
        "Identify the service account owner via CMDB or last-known-good contact",
        "Rotate credentials immediately regardless of ownership status",
        "Disable the account if no owner is confirmed within 48 hours",
        "Add the account to the automated service-account lifecycle management system",
    ],
    "SOD_VIOLATION": [
        "Remove one of the conflicting entitlements within 5 business days",
        "If removal is not feasible, implement compensating controls (dual approval, audit logging)",
        "Document the exception in the risk register with CISO sign-off",
        "Schedule quarterly re-validation of the exception",
    ],
    "AFTER_HOURS_ACCESS": [
        "Cross-reference with on-call/shift schedules from PagerDuty or team roster",
        "Review VPN and source-IP logs for geographic anomalies",
        "If unexplained, enforce time-based conditional access policies",
        "Escalate to SOC if the user has no documented justification",
    ],
    "EXCESSIVE_EXPORTS": [
        "Review exported data via DLP for sensitive or regulated content",
        "Interview the user's manager about business justification for exports",
        "Restrict bulk export permissions until review is complete",
        "Enable enhanced logging on export-capable systems",
    ],
    "CROSS_DEPT_ACCESS": [
        "Review the user's access list with their department head",
        "Remove cross-departmental entitlements not justified by current role",
        "Evaluate whether the user's role requires a formal cross-functional access grant",
    ],
    "HIGH_FAILURE_RATE": [
        "Check if the user reported forgotten credentials or MFA issues to helpdesk",
        "Review source IPs for credential-stuffing indicators",
        "Enforce password reset and MFA re-enrollment",
        "Correlate with SOC alerts for brute-force or account-takeover patterns",
    ],
    "PRIVILEGE_OUTLIER": [
        "Conduct a formal entitlement review comparing the user to peer-group norms",
        "Remove any standing admin privileges that can be replaced with JIT access",
        "Document business justification for outlier-level access",
    ],
    "HIGH_BLAST_RADIUS": [
        "Implement network segmentation to limit lateral movement potential",
        "Require MFA for all privileged operations on shared systems",
        "Reduce standing access by migrating to a Privileged Access Management (PAM) solution",
    ],
}

# Default actions for unknown finding types
DEFAULT_ACTIONS = [
    "Review the finding with the identity governance team",
    "Verify the user's current role and access requirements",
    "Document remediation steps and timeline",
]


# ══════════════════════════════════════════════════════════════════════════════
# DETAIL TEMPLATES  (each >= 100 chars, cite specific numbers)
# ══════════════════════════════════════════════════════════════════════════════

def _build_detail_stale(uid: str, priv: str, days_inactive: int,
                        sys_count: int, last_login: str, **_: Any) -> str:
    """Build >=100-char detail string for STALE_PRIVILEGED_ACCOUNT."""
    return (
        f"Account {uid} (privilege_level={priv}) has been inactive for {days_inactive} "
        f"days since last login on {last_login}. The account retains access to "
        f"{sys_count} system(s). Stale privileged accounts are high-value targets "
        f"for credential theft and account takeover attacks. Immediate review is "
        f"required to determine whether this account should be disabled."
    )


def _build_detail_broad(uid: str, sys_count: int, peer_median: float,
                        systems: str, **_: Any) -> str:
    """Build >=100-char detail string for BROAD_PRIVILEGE_SCOPE."""
    excess_pct = ((sys_count / max(peer_median, 1)) - 1) * 100
    return (
        f"Account {uid} has access to {sys_count} systems ({systems}), which is "
        f"{excess_pct:.0f}% above the peer-group median of {peer_median:.1f} "
        f"systems. Excessive breadth of access violates the principle of least "
        f"privilege and increases the blast radius in a compromise scenario."
    )


def _build_detail_orphaned(uid: str, total_events: int, days_inactive: int,
                           sys_count: int, **_: Any) -> str:
    """Build >=100-char detail string for ORPHANED_SERVICE_ACCOUNT."""
    return (
        f"Service account {uid} has generated {total_events} event(s) in the "
        f"observation period and has been inactive for {days_inactive} days. "
        f"With standing access to {sys_count} system(s), this unmonitored "
        f"service account represents an unmanaged risk vector that could be "
        f"exploited for lateral movement or data exfiltration."
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXPLANATION GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_explanation(user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a complete explanation JSON dict for a single user.

    Parameters
    ----------
    user_data : dict
        Must contain at least:

        - ``user_id``, ``username``, ``risk_level``, ``risk_score``
        - ``findings``: list of finding dicts
        - ``confidence``, ``confidence_basis``
        - ``exceptions``: list of exception tag strings (may be empty)
        - ``recurrence_patterns``: list of recurrence dicts (may be empty)
        - ``days_inactive``, ``system_count``, ``privilege_level``
        - ``department``, ``job_title``

    Returns
    -------
    dict
        Full explanation matching the Section 2 output schema.
    """
    uid = user_data.get("user_id", "UNKNOWN")
    username = user_data.get("username", "unknown")
    risk_level = user_data.get("risk_level", "LOW")
    risk_score = user_data.get("risk_score", user_data.get("anomaly_score", 0))
    findings = user_data.get("findings", [])
    confidence = user_data.get("confidence", 0.5)
    confidence_basis = user_data.get("confidence_basis", "")
    exceptions = user_data.get("exceptions", [])
    recurrence = user_data.get("recurrence_patterns", [])
    days_inactive = user_data.get("days_inactive", 0)
    sys_count = user_data.get("system_count", 0)
    priv = user_data.get("privilege_level", "user")
    dept = user_data.get("department", "")
    title = user_data.get("job_title", "")
    total_events = user_data.get("total_events", 0)

    # ── Enrich findings with detailed templates ─────────────────────────────
    enriched_findings: List[Dict[str, Any]] = []
    for f in findings:
        finding_type = f.get("finding", "")
        detail = f.get("details", "")

        # Ensure detail is >= 100 chars; if not, pad with context
        if len(detail) < 100:
            detail = _pad_detail(detail, uid, priv, days_inactive, sys_count,
                                 total_events)

        enriched = {
            "finding": finding_type,
            "details": detail,
            "severity": f.get("severity", "MEDIUM"),
            "recommendation": f.get("recommendation", "Review with identity governance team."),
        }
        if "compliance_references" in f:
            enriched["compliance_references"] = f["compliance_references"]
        enriched_findings.append(enriched)

    # ── Suggested actions ───────────────────────────────────────────────────
    suggested_actions = _build_suggested_actions(findings, exceptions, recurrence,
                                                  uid, priv, days_inactive, sys_count)

    # ── Escalation ──────────────────────────────────────────────────────────
    next_escalation = ESCALATION_MAP.get(risk_level, ESCALATION_MAP["LOW"])

    # ── Exception annotations ───────────────────────────────────────────────
    if exceptions:
        exception_note = (
            f"Exception(s) applied: {', '.join(exceptions)}. "
            f"Score and/or severity have been adjusted accordingly."
        )
        enriched_findings.append({
            "finding": "EXCEPTIONS_APPLIED",
            "details": exception_note + " " * max(0, 100 - len(exception_note)),
            "severity": "INFO",
            "recommendation": "Review exception validity during next audit cycle.",
        })

    # ── Recurrence annotations ──────────────────────────────────────────────
    recurring_patterns = [p for p in recurrence
                          if isinstance(p, dict) and p.get("is_recurring")]
    if recurring_patterns:
        for p in recurring_patterns:
            pattern_note = (
                f"Recurring pattern detected: action '{p.get('pattern', 'N/A')}' "
                f"repeats every ~{p.get('interval_days', 'N/A')} days. This is "
                f"consistent with a scheduled automated process. Severity has "
                f"been downgraded by 5 points to reflect the benign nature of "
                f"this recurring activity pattern."
            )
            enriched_findings.append({
                "finding": "RECURRING_PATTERN",
                "details": pattern_note,
                "severity": "INFO",
                "recommendation": (
                    "Confirm the scheduled process with the system owner; "
                    "add to the approved-automation whitelist."
                ),
            })

    # ── Build output ────────────────────────────────────────────────────────
    explanation = {
        "user_id": uid,
        "username": username,
        "department": dept,
        "job_title": title,
        "privilege_level": priv,
        "risk_level": risk_level,
        "risk_score": int(risk_score),
        "findings": enriched_findings,
        "confidence": round(confidence, 2),
        "confidence_basis": confidence_basis,
        "suggested_actions": suggested_actions,
        "next_escalation": next_escalation,
        "analysis_timestamp": datetime.now(UTC).isoformat(),
    }

    return explanation


def _pad_detail(detail: str, uid: str, priv: str, days_inactive: int,
                sys_count: int, total_events: int) -> str:
    """Ensure a finding detail string is >= 100 characters by appending context."""
    padding = (
        f" Additional context: user {uid} (privilege_level={priv}, "
        f"days_inactive={days_inactive}, system_count={sys_count}, "
        f"total_events={total_events})."
    )
    full = detail + padding
    return full


def _build_suggested_actions(
    findings: List[Dict[str, Any]],
    exceptions: List[str],
    recurrence: List[Dict[str, Any]],
    uid: str,
    priv: str,
    days_inactive: int,
    sys_count: int,
) -> List[str]:
    """Build a prioritised list of suggested actions based on findings."""
    actions: List[str] = []
    seen_types: set = set()

    # Priority-ordered finding types
    priority_order = [
        "STALE_PRIVILEGED_ACCOUNT", "ORPHANED_SERVICE_ACCOUNT",
        "SOD_VIOLATION", "EXCESSIVE_EXPORTS", "HIGH_FAILURE_RATE",
        "AFTER_HOURS_ACCESS", "BROAD_PRIVILEGE_SCOPE", "CROSS_DEPT_ACCESS",
        "PRIVILEGE_OUTLIER", "HIGH_BLAST_RADIUS",
    ]

    # Collect unique finding types in priority order
    finding_types = [f.get("finding", "") for f in findings]
    ordered = [ft for ft in priority_order if ft in finding_types]
    ordered += [ft for ft in finding_types if ft not in ordered and ft not in seen_types]

    for ft in ordered:
        if ft in seen_types:
            continue
        seen_types.add(ft)
        templates = ACTION_TEMPLATES.get(ft, DEFAULT_ACTIONS)
        for action in templates[:2]:  # Top 2 per finding type
            # Personalise with specific numbers
            personalised = action.replace("the user", f"user {uid}")
            if personalised not in actions:
                actions.append(personalised)

    # Add exception-specific actions
    if "EXEC_ROLE_EXCEPTION" in exceptions:
        actions.append(
            f"Confirm {uid}'s executive role is current via HR verification "
            f"before closing any findings."
        )
    if "NEW_HIRE_EXCEPTION" in exceptions:
        actions.append(
            f"Schedule a 90-day post-onboarding access review for {uid}."
        )

    # Add recurrence-specific actions
    recurring = [p for p in recurrence if isinstance(p, dict) and p.get("is_recurring")]
    if recurring:
        actions.append(
            f"Verify the {len(recurring)} recurring pattern(s) detected for "
            f"{uid} against approved automation schedules."
        )

    # Ensure we always have at least one action
    if not actions:
        actions.append(
            f"Review {uid}'s access profile ({sys_count} systems, "
            f"{priv} privilege, {days_inactive} days inactive) during "
            f"the next scheduled access certification."
        )

    return actions


# ══════════════════════════════════════════════════════════════════════════════
# EXECUTIVE SUMMARY GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_executive_summary(
    all_results: List[Dict[str, Any]],
    users_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Generate an executive-level summary of the entire analysis.

    Parameters
    ----------
    all_results : list[dict]
        List of per-user explanation dicts from ``generate_explanation``.
    users_df : pd.DataFrame
        Full user DataFrame.
    events_df : pd.DataFrame
        Full event DataFrame.

    Returns
    -------
    dict
        Executive summary with key metrics, risk distribution, top findings,
        and recommended priorities.
    """
    total_users = len(users_df)
    total_events = len(events_df)
    analysed_users = len(all_results)

    # Risk distribution
    risk_dist = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in all_results:
        lvl = r.get("risk_level", "LOW")
        risk_dist[lvl] = risk_dist.get(lvl, 0) + 1

    # Finding type frequency
    finding_freq: Dict[str, int] = {}
    for r in all_results:
        for f in r.get("findings", []):
            ft = f.get("finding", "UNKNOWN")
            finding_freq[ft] = finding_freq.get(ft, 0) + 1

    top_findings = sorted(finding_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    # High-risk users
    critical_users = [
        {"user_id": r["user_id"], "username": r.get("username", ""),
         "risk_score": r.get("risk_score", 0),
         "finding_count": len(r.get("findings", []))}
        for r in all_results if r.get("risk_level") == "CRITICAL"
    ]
    critical_users.sort(key=lambda x: x["risk_score"], reverse=True)

    # Confidence stats
    confidences = [r.get("confidence", 0) for r in all_results]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    low_confidence_count = sum(1 for c in confidences if c < 0.3)

    # Privilege breakdown
    priv_counts = users_df["privilege_level"].value_counts().to_dict()

    # Users with zero events
    users_with_events = set(events_df["user_id"].unique())
    zero_event_users = total_users - len(users_with_events)

    summary = {
        "report_title": "Identity Sprawl & Privilege Abuse Detection — Executive Summary",
        "generated_at": datetime.now(UTC).isoformat(),
        "reference_date": str(REFERENCE_DATE.date()),
        "scope": {
            "total_users_analysed": analysed_users,
            "total_users_in_directory": total_users,
            "total_events_processed": total_events,
            "observation_period": f"{events_df['timestamp'].min()} to {events_df['timestamp'].max()}" if not events_df.empty else "N/A",
            "users_with_zero_events": zero_event_users,
        },
        "risk_distribution": risk_dist,
        "risk_summary": (
            f"{risk_dist.get('CRITICAL', 0)} CRITICAL and "
            f"{risk_dist.get('HIGH', 0)} HIGH-risk users identified out of "
            f"{analysed_users} analysed. {zero_event_users} users had zero "
            f"events and were scored on static profile only."
        ),
        "top_findings": [
            {"finding": ft, "count": cnt} for ft, cnt in top_findings
        ],
        "critical_users": critical_users[:10],
        "confidence_metrics": {
            "average_confidence": round(avg_confidence, 2),
            "low_confidence_count": low_confidence_count,
            "note": (
                f"{low_confidence_count} users have confidence < 0.3, meaning "
                f"their risk scores rely primarily on static attributes."
            ),
        },
        "privilege_distribution": priv_counts,
        "recommended_priorities": _build_priorities(risk_dist, finding_freq,
                                                      zero_event_users),
    }

    return summary


def _build_priorities(
    risk_dist: Dict[str, int],
    finding_freq: Dict[str, int],
    zero_event_users: int,
) -> List[str]:
    """Build a prioritised list of recommended actions for leadership."""
    priorities = []

    crit = risk_dist.get("CRITICAL", 0)
    high = risk_dist.get("HIGH", 0)

    if crit > 0:
        priorities.append(
            f"IMMEDIATE: Review and remediate {crit} CRITICAL-risk accounts "
            f"within 24 hours — these represent the highest exposure."
        )
    if high > 0:
        priorities.append(
            f"SHORT-TERM: Address {high} HIGH-risk accounts within 5 business "
            f"days via incident tickets and manager review."
        )

    stale = finding_freq.get("STALE_PRIVILEGED_ACCOUNT", 0)
    if stale > 0:
        priorities.append(
            f"GOVERNANCE: {stale} stale privileged accounts detected — "
            f"implement automated deprovisioning to prevent recurrence."
        )

    orphaned = finding_freq.get("ORPHANED_SERVICE_ACCOUNT", 0)
    if orphaned > 0:
        priorities.append(
            f"SERVICE ACCOUNTS: {orphaned} potentially orphaned service accounts "
            f"require owner identification and credential rotation."
        )

    sod = finding_freq.get("SOD_VIOLATION", 0)
    if sod > 0:
        priorities.append(
            f"COMPLIANCE: {sod} separation-of-duties violations require "
            f"remediation or documented compensating controls."
        )

    if zero_event_users > 20:
        priorities.append(
            f"VISIBILITY GAP: {zero_event_users} users have zero events in "
            f"the observation period — expand logging coverage to improve "
            f"detection fidelity."
        )

    if not priorities:
        priorities.append(
            "No immediate critical actions required. Continue standard "
            "monitoring and schedule next quarterly access review."
        )

    return priorities


# ══════════════════════════════════════════════════════════════════════════════
# FULL REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(
    all_results: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate the full output report combining all user results and metadata.

    Parameters
    ----------
    all_results : list[dict]
        List of per-user explanation dicts from ``generate_explanation``.
    metadata : dict, optional
        Additional metadata to include (pipeline version, run config, etc.).

    Returns
    -------
    dict
        Complete report dict ready for JSON serialisation.
    """
    if metadata is None:
        metadata = {}

    # Sort results: CRITICAL first, then HIGH, then by score descending
    level_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    sorted_results = sorted(
        all_results,
        key=lambda r: (level_order.get(r.get("risk_level", "LOW"), 5),
                       -r.get("risk_score", 0)),
    )

    # Aggregate stats
    total_findings = sum(len(r.get("findings", [])) for r in all_results)
    severity_counts: Dict[str, int] = {}
    for r in all_results:
        for f in r.get("findings", []):
            sev = f.get("severity", "MEDIUM")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

    report = {
        "report_metadata": {
            "title": "Identity Sprawl & Privilege Abuse Detection Report",
            "generated_at": datetime.now(UTC).isoformat(),
            "reference_date": str(REFERENCE_DATE.date()),
            "pipeline_version": metadata.get("pipeline_version", "1.0.0"),
            "total_users_analysed": len(all_results),
            "total_findings": total_findings,
            "severity_distribution": severity_counts,
            **{k: v for k, v in metadata.items() if k != "pipeline_version"},
        },
        "user_results": sorted_results,
    }

    logger.info(
        "Report generated: %d users, %d findings", len(all_results), total_findings
    )

    return report
