"""
EXTRA: Compliance Framework Mapper
===================================
Maps each finding type to specific compliance framework references
(NIST 800-53, GDPR, SOX, ISO 27001) and generates compliance-oriented
reports with remediation timeline recommendations.

Functions
---------
- map_compliance(findings) → enhanced findings with compliance_references
- generate_compliance_report(all_results) → compliance summary by framework
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, UTC
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# COMPLIANCE MAPPING TABLE
# ══════════════════════════════════════════════════════════════════════════════

COMPLIANCE_MAP: Dict[str, Dict[str, Any]] = {
    "STALE_PRIVILEGED_ACCOUNT": {
        "frameworks": {
            "NIST_800-53": {
                "controls": ["AC-2(3)"],
                "description": "Account Management — Disable Inactive Accounts",
                "requirement": (
                    "The information system automatically disables inactive "
                    "accounts after a defined time period."
                ),
            },
            "GDPR": {
                "controls": ["Art 32"],
                "description": "Security of Processing",
                "requirement": (
                    "Implement appropriate technical measures to ensure "
                    "ongoing confidentiality, including timely deprovisioning "
                    "of unused accounts that may access personal data."
                ),
            },
            "ISO_27001": {
                "controls": ["A.9.2.6"],
                "description": "Removal or Adjustment of Access Rights",
                "requirement": (
                    "Access rights shall be removed upon termination or "
                    "adjusted upon change of employment."
                ),
            },
        },
        "remediation_timeline": "24 hours for CRITICAL; 5 business days for HIGH",
        "audit_evidence": [
            "Last login timestamp",
            "Account status in identity provider",
            "Manager acknowledgment of deprovisioning",
        ],
    },
    "BROAD_PRIVILEGE_SCOPE": {
        "frameworks": {
            "NIST_800-53": {
                "controls": ["AC-6"],
                "description": "Least Privilege",
                "requirement": (
                    "The organisation employs the principle of least privilege, "
                    "allowing only authorised accesses necessary to accomplish "
                    "assigned tasks."
                ),
            },
            "GDPR": {
                "controls": ["Art 5(1)(f)"],
                "description": "Integrity and Confidentiality Principle",
                "requirement": (
                    "Personal data shall be processed in a manner that ensures "
                    "appropriate security, including protection against "
                    "unauthorised access."
                ),
            },
            "SOX": {
                "controls": ["Section 404"],
                "description": "Internal Controls Assessment",
                "requirement": (
                    "Management must assess and report on the effectiveness "
                    "of internal controls over financial reporting, including "
                    "access controls."
                ),
            },
        },
        "remediation_timeline": "5 business days for privilege review; 30 days for full remediation",
        "audit_evidence": [
            "Current entitlement list",
            "Peer group comparison report",
            "Manager-approved access justification",
        ],
    },
    "ORPHANED_SERVICE_ACCOUNT": {
        "frameworks": {
            "NIST_800-53": {
                "controls": ["AC-2(4)"],
                "description": "Account Management — Automated Audit Actions",
                "requirement": (
                    "The information system automatically audits account "
                    "creation, modification, enabling, disabling, and "
                    "removal actions."
                ),
            },
            "ISO_27001": {
                "controls": ["A.9.2.5"],
                "description": "Review of User Access Rights",
                "requirement": (
                    "Asset owners shall review user access rights at "
                    "regular intervals."
                ),
            },
        },
        "remediation_timeline": "48 hours to identify owner; disable within 72 hours if no owner found",
        "audit_evidence": [
            "Service account inventory",
            "CMDB owner assignment",
            "Credential rotation records",
        ],
    },
    "SOD_VIOLATION": {
        "frameworks": {
            "NIST_800-53": {
                "controls": ["AC-5"],
                "description": "Separation of Duties",
                "requirement": (
                    "The organisation separates duties of individuals to "
                    "reduce risk of malevolent activity without collusion."
                ),
            },
            "SOX": {
                "controls": ["Section 302", "Section 404"],
                "description": "Corporate Responsibility for Financial Reports / Internal Controls",
                "requirement": (
                    "Officers must certify that internal controls, including "
                    "duty segregation, are effective and operating."
                ),
            },
        },
        "remediation_timeline": "5 business days to remediate or document compensating control",
        "audit_evidence": [
            "SOD violation report",
            "Compensating control documentation",
            "CISO exception approval (if applicable)",
        ],
    },
    "AFTER_HOURS_ACCESS": {
        "frameworks": {
            "NIST_800-53": {
                "controls": ["AU-6"],
                "description": "Audit Review, Analysis, and Reporting",
                "requirement": (
                    "The organisation reviews and analyses information "
                    "system audit records for indications of inappropriate "
                    "or unusual activity."
                ),
            },
            "ISO_27001": {
                "controls": ["A.12.4"],
                "description": "Logging and Monitoring",
                "requirement": (
                    "Event logs recording user activities, exceptions, "
                    "and information security events shall be produced "
                    "and kept."
                ),
            },
        },
        "remediation_timeline": "3 business days for investigation; immediate action if no justification found",
        "audit_evidence": [
            "Event timestamps and source IPs",
            "On-call schedule verification",
            "Manager acknowledgment of legitimate access",
        ],
    },
    "EXCESSIVE_EXPORTS": {
        "frameworks": {
            "GDPR": {
                "controls": ["Art 32"],
                "description": "Security of Processing",
                "requirement": (
                    "Implement technical measures to prevent unauthorised "
                    "data exfiltration and ensure data integrity."
                ),
            },
            "NIST_800-53": {
                "controls": ["SI-4"],
                "description": "Information System Monitoring",
                "requirement": (
                    "The organisation monitors the information system "
                    "to detect attacks and indicators of potential attacks."
                ),
            },
        },
        "remediation_timeline": "24 hours for DLP review; 48 hours for export restriction if unverified",
        "audit_evidence": [
            "Export operation logs",
            "DLP scan results",
            "Data classification of exported content",
        ],
    },
    "CROSS_DEPT_ACCESS": {
        "frameworks": {
            "NIST_800-53": {
                "controls": ["AC-3"],
                "description": "Access Enforcement",
                "requirement": (
                    "The information system enforces approved authorisations "
                    "for logical access to information and system resources."
                ),
            },
            "ISO_27001": {
                "controls": ["A.9.4.1"],
                "description": "Information Access Restriction",
                "requirement": (
                    "Access to information and application system functions "
                    "shall be restricted in accordance with the access "
                    "control policy."
                ),
            },
        },
        "remediation_timeline": "5 business days for access review; 15 days for remediation",
        "audit_evidence": [
            "Cross-departmental resource access log",
            "Department-head approval for cross-functional access",
            "Role definition documentation",
        ],
    },
    "HIGH_FAILURE_RATE": {
        "frameworks": {
            "NIST_800-53": {
                "controls": ["AC-7"],
                "description": "Unsuccessful Logon Attempts",
                "requirement": (
                    "The information system enforces a limit of consecutive "
                    "invalid logon attempts and automatically locks the "
                    "account when the limit is exceeded."
                ),
            },
            "ISO_27001": {
                "controls": ["A.9.4.2"],
                "description": "Secure Log-on Procedures",
                "requirement": (
                    "Access to systems shall be controlled by a secure "
                    "log-on procedure."
                ),
            },
        },
        "remediation_timeline": "Immediate lockout for brute-force indicators; 24 hours for investigation",
        "audit_evidence": [
            "Authentication failure logs",
            "Source IP analysis",
            "MFA status verification",
        ],
    },
    "PRIVILEGE_OUTLIER": {
        "frameworks": {
            "NIST_800-53": {
                "controls": ["AC-6(1)"],
                "description": "Least Privilege — Authorize Access to Security Functions",
                "requirement": (
                    "The organisation explicitly authorises access to "
                    "security-relevant information and functions."
                ),
            },
            "ISO_27001": {
                "controls": ["A.9.2.3"],
                "description": "Management of Privileged Access Rights",
                "requirement": (
                    "The allocation and use of privileged access rights "
                    "shall be restricted and controlled."
                ),
            },
        },
        "remediation_timeline": "5 business days for peer-group comparison review; 15 days for remediation",
        "audit_evidence": [
            "Peer-group privilege comparison",
            "Entitlement justification records",
            "Manager sign-off on outlier access",
        ],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# MAP COMPLIANCE TO FINDINGS
# ══════════════════════════════════════════════════════════════════════════════

def map_compliance(
    findings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Enhance a list of findings with compliance framework references.

    Parameters
    ----------
    findings : list[dict]
        List of finding dicts, each with at least a ``finding`` key.

    Returns
    -------
    list[dict]
        Same findings with added ``compliance_references`` key containing
        the framework mapping, remediation timeline, and audit evidence.
    """
    enhanced: List[Dict[str, Any]] = []

    for finding in findings:
        f = finding.copy()
        finding_type = f.get("finding", "")
        mapping = COMPLIANCE_MAP.get(finding_type)

        if mapping:
            f["compliance_references"] = {
                "frameworks": mapping["frameworks"],
                "remediation_timeline": mapping["remediation_timeline"],
                "audit_evidence_required": mapping["audit_evidence"],
            }
        else:
            f["compliance_references"] = {
                "frameworks": {},
                "remediation_timeline": "Follow organisational incident response SLA",
                "audit_evidence_required": ["Finding details", "Remediation actions taken"],
            }

        enhanced.append(f)

    logger.info(
        "Compliance mapping applied to %d findings (%d with framework refs)",
        len(enhanced),
        sum(1 for f in enhanced if f["compliance_references"]["frameworks"]),
    )

    return enhanced


# ══════════════════════════════════════════════════════════════════════════════
# COMPLIANCE REPORT GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_compliance_report(
    all_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate a compliance summary grouped by framework.

    Parameters
    ----------
    all_results : list[dict]
        List of per-user explanation dicts. Each user's ``findings`` must
        have been enhanced via ``map_compliance`` (i.e., contain
        ``compliance_references``).

    Returns
    -------
    dict
        Compliance report with per-framework breakdowns, control coverage,
        and recommended timelines.
    """
    # Collect all findings across users
    all_findings: List[Dict[str, Any]] = []
    for result in all_results:
        for finding in result.get("findings", []):
            finding_with_user = finding.copy()
            finding_with_user["user_id"] = result.get("user_id", "UNKNOWN")
            finding_with_user["risk_level"] = result.get("risk_level", "LOW")
            all_findings.append(finding_with_user)

    # Group by framework
    framework_groups: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "controls_triggered": set(),
        "finding_count": 0,
        "finding_types": defaultdict(int),
        "affected_users": set(),
        "severity_distribution": defaultdict(int),
        "remediation_actions": [],
    })

    for f in all_findings:
        comp_ref = f.get("compliance_references", {})
        frameworks = comp_ref.get("frameworks", {})
        timeline = comp_ref.get("remediation_timeline", "")
        finding_type = f.get("finding", "UNKNOWN")
        uid = f.get("user_id", "UNKNOWN")
        severity = f.get("severity", "MEDIUM")

        for fw_name, fw_data in frameworks.items():
            grp = framework_groups[fw_name]
            controls = fw_data.get("controls", [])
            grp["controls_triggered"].update(controls)
            grp["finding_count"] += 1
            grp["finding_types"][finding_type] += 1
            grp["affected_users"].add(uid)
            grp["severity_distribution"][severity] += 1
            if timeline:
                grp["remediation_actions"].append({
                    "finding_type": finding_type,
                    "timeline": timeline,
                    "controls": controls,
                })

    # Serialise for JSON
    framework_summaries = {}
    for fw_name, grp in framework_groups.items():
        # Deduplicate remediation actions
        seen_actions = set()
        unique_actions = []
        for action in grp["remediation_actions"]:
            key = (action["finding_type"], action["timeline"])
            if key not in seen_actions:
                seen_actions.add(key)
                unique_actions.append(action)

        framework_summaries[fw_name] = {
            "framework_name": _friendly_name(fw_name),
            "controls_triggered": sorted(grp["controls_triggered"]),
            "total_findings": grp["finding_count"],
            "unique_finding_types": dict(grp["finding_types"]),
            "affected_user_count": len(grp["affected_users"]),
            "severity_distribution": dict(grp["severity_distribution"]),
            "remediation_actions": unique_actions,
        }

    # Overall compliance posture
    total_findings = len(all_findings)
    frameworks_triggered = len(framework_summaries)

    # Control coverage analysis
    all_controls_triggered: set = set()
    for fw in framework_summaries.values():
        all_controls_triggered.update(fw["controls_triggered"])

    report = {
        "report_title": "Compliance Framework Impact Analysis",
        "generated_at": datetime.now(UTC).isoformat(),
        "compliance_posture": {
            "total_findings_with_compliance_impact": total_findings,
            "frameworks_triggered": frameworks_triggered,
            "total_controls_triggered": len(all_controls_triggered),
            "controls_list": sorted(all_controls_triggered),
        },
        "framework_summaries": framework_summaries,
        "remediation_priority": _build_remediation_priority(framework_summaries),
    }

    logger.info(
        "Compliance report generated: %d frameworks, %d controls triggered",
        frameworks_triggered, len(all_controls_triggered),
    )

    return report


def _friendly_name(fw_key: str) -> str:
    """Convert framework key to human-readable name."""
    names = {
        "NIST_800-53": "NIST SP 800-53 Rev. 5",
        "GDPR": "EU General Data Protection Regulation (GDPR)",
        "SOX": "Sarbanes-Oxley Act (SOX)",
        "ISO_27001": "ISO/IEC 27001:2022",
    }
    return names.get(fw_key, fw_key)


def _build_remediation_priority(
    framework_summaries: Dict[str, Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build a prioritised remediation plan across frameworks.

    Returns
    -------
    list[dict]
        Priority-ordered remediation items with ``priority``, ``framework``,
        ``action``, ``deadline``.
    """
    priorities = []

    # Sort frameworks by total findings (highest first)
    sorted_fw = sorted(
        framework_summaries.items(),
        key=lambda x: x[1]["total_findings"],
        reverse=True,
    )

    priority_rank = 1
    for fw_key, fw_data in sorted_fw:
        sev_dist = fw_data.get("severity_distribution", {})
        critical_count = sev_dist.get("CRITICAL", 0) + sev_dist.get("HIGH", 0)

        if critical_count > 0:
            urgency = "IMMEDIATE"
            deadline = "Within 24-48 hours"
        elif sev_dist.get("MEDIUM", 0) > 0:
            urgency = "SHORT-TERM"
            deadline = "Within 5-15 business days"
        else:
            urgency = "SCHEDULED"
            deadline = "Next quarterly review cycle"

        priorities.append({
            "priority": priority_rank,
            "urgency": urgency,
            "framework": fw_data["framework_name"],
            "controls": fw_data["controls_triggered"],
            "total_findings": fw_data["total_findings"],
            "action": (
                f"Address {fw_data['total_findings']} finding(s) affecting "
                f"{fw_data['affected_user_count']} user(s) across controls: "
                f"{', '.join(fw_data['controls_triggered'])}."
            ),
            "deadline": deadline,
        })
        priority_rank += 1

    return priorities
