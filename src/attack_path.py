"""
EXTRA: Attack Path & Lateral Movement Analysis
================================================
Traces potential lateral movement through the privilege graph for
CRITICAL/HIGH-risk users.  Uses sensitivity-weighted scoring to
assess exposure.

Functions
---------
- analyze_attack_paths(G, scored_df, users_df) → attack_paths list
- compute_blast_radius(G, user_id) → blast radius dict
- find_critical_paths(G, scored_df) → list of highest-risk paths
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Sensitivity-weighted scoring for resources/systems
# ──────────────────────────────────────────────────────────────────────────────
SENSITIVITY_WEIGHTS: Dict[str, int] = {
    "PROD_DB": 10,
    "Customer_Vault": 10,
    "SIEM": 8,
    "HRIS": 8,
    "ADMIN_SYS": 7,
    "GL_System": 7,
    "Admin_Console": 5,
    "Email_Archive": 4,
    "Data_Lake": 3,
    "BI_Tool": 2,
    "File_Share": 1,
    # Identity systems (from systems_access column)
    "AD": 6,
    "Azure_AD": 6,
    "AWS_IAM": 7,
    "GCP": 6,
    "Okta": 7,
    "EMAIL": 2,
    "VPN": 4,
    "Salesforce": 3,
    "ServiceNow": 3,
}

DEFAULT_WEIGHT = 3  # for unknown systems


def _get_weight(system: str) -> int:
    """Return the sensitivity weight for a system/resource name."""
    return SENSITIVITY_WEIGHTS.get(system, DEFAULT_WEIGHT)


# ══════════════════════════════════════════════════════════════════════════════
# BLAST RADIUS COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_blast_radius(
    G: nx.Graph,
    user_id: str,
) -> Dict[str, Any]:
    """Compute the blast radius for a single user in the privilege graph.

    The blast radius traces: ``user → systems → other users sharing those
    systems → their other systems``, building a chain of exposure.

    Parameters
    ----------
    G : nx.Graph
        Privilege graph from ``graph.build_privilege_graph``.
        Nodes can be users (prefix ``USR``) or systems.  Edges connect
        users to systems they have access to.
    user_id : str
        The user to analyse.

    Returns
    -------
    dict
        Keys:

        - ``user_id``: the analysed user
        - ``direct_systems``: list of systems the user has direct access to
        - ``indirect_users``: list of other users reachable through shared systems
        - ``indirect_systems``: list of systems reachable through indirect users
        - ``total_exposure_score``: sensitivity-weighted total exposure score
        - ``blast_chain``: list of dicts describing the chain steps
    """
    if user_id not in G:
        return {
            "user_id": user_id,
            "direct_systems": [],
            "indirect_users": [],
            "indirect_systems": [],
            "total_exposure_score": 0,
            "blast_chain": [],
        }

    # Step 1: Direct systems (neighbors of the user node)
    direct_systems: List[str] = []
    for neighbor in G.neighbors(user_id):
        node_data = G.nodes.get(neighbor, {})
        # System nodes are those that don't start with USR (or have type="system")
        if not str(neighbor).startswith("USR"):
            direct_systems.append(str(neighbor))
        elif node_data.get("type") == "system":
            direct_systems.append(str(neighbor))

    # Step 2: Indirect users (other users who share the same systems)
    indirect_users: Set[str] = set()
    for system in direct_systems:
        if system in G:
            for neighbor in G.neighbors(system):
                if str(neighbor).startswith("USR") and neighbor != user_id:
                    indirect_users.add(str(neighbor))

    # Step 3: Indirect systems (systems accessible through indirect users)
    indirect_systems: Set[str] = set()
    for ind_user in indirect_users:
        if ind_user in G:
            for neighbor in G.neighbors(ind_user):
                if not str(neighbor).startswith("USR") and neighbor not in direct_systems:
                    indirect_systems.add(str(neighbor))

    # Step 4: Total exposure score (sensitivity-weighted)
    direct_score = sum(_get_weight(s) for s in direct_systems)
    indirect_score = sum(_get_weight(s) for s in indirect_systems) * 0.5  # discounted
    total_exposure = direct_score + indirect_score

    # Step 5: Blast chain
    blast_chain: List[Dict[str, Any]] = []

    # Level 0: compromised user
    blast_chain.append({
        "level": 0,
        "type": "compromise_origin",
        "entity": user_id,
        "description": f"Initial compromise of account {user_id}",
    })

    # Level 1: direct systems
    for system in sorted(direct_systems):
        blast_chain.append({
            "level": 1,
            "type": "direct_system",
            "entity": system,
            "sensitivity_weight": _get_weight(system),
            "description": f"Attacker gains access to {system} (weight={_get_weight(system)})",
        })

    # Level 2: lateral movement to users sharing systems
    for ind_user in sorted(indirect_users):
        # Find which system(s) are shared
        shared = []
        for system in direct_systems:
            if system in G and ind_user in G.neighbors(system):
                shared.append(system)
        blast_chain.append({
            "level": 2,
            "type": "lateral_user",
            "entity": ind_user,
            "shared_systems": shared,
            "description": (
                f"Lateral movement to {ind_user} via shared system(s): "
                f"{', '.join(shared)}"
            ),
        })

    # Level 3: additional systems reachable through lateral users
    for ind_sys in sorted(indirect_systems):
        owners = [u for u in indirect_users
                  if ind_sys in G and u in G.neighbors(ind_sys)]
        blast_chain.append({
            "level": 3,
            "type": "indirect_system",
            "entity": ind_sys,
            "sensitivity_weight": _get_weight(ind_sys),
            "reachable_via": owners[:5],  # cap for readability
            "description": (
                f"Indirect access to {ind_sys} (weight={_get_weight(ind_sys)}) "
                f"via {len(owners)} lateral user(s)"
            ),
        })

    return {
        "user_id": user_id,
        "direct_systems": sorted(direct_systems),
        "direct_system_count": len(direct_systems),
        "indirect_users": sorted(indirect_users),
        "indirect_user_count": len(indirect_users),
        "indirect_systems": sorted(indirect_systems),
        "indirect_system_count": len(indirect_systems),
        "total_exposure_score": round(total_exposure, 1),
        "blast_chain": blast_chain,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ATTACK PATH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_attack_paths(
    G: nx.Graph,
    scored_df: pd.DataFrame,
    users_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Analyse potential attack paths for CRITICAL and HIGH-risk users.

    For each high-risk user, traces lateral movement potential through
    the privilege graph and computes exposure metrics.

    Parameters
    ----------
    G : nx.Graph
        Privilege graph from ``graph.build_privilege_graph``.
    scored_df : pd.DataFrame
        Must contain ``user_id``, ``anomaly_score``, ``risk_level``.
    users_df : pd.DataFrame
        User data for enrichment.

    Returns
    -------
    list[dict]
        One entry per CRITICAL/HIGH user with blast radius, attack chain,
        and recommendations.
    """
    # Filter to CRITICAL and HIGH risk users
    high_risk = scored_df[scored_df["risk_level"].isin(["CRITICAL", "HIGH"])].copy()

    if high_risk.empty:
        logger.info("No CRITICAL/HIGH-risk users found; attack path analysis skipped.")
        return []

    # Build user lookup for enrichment
    user_lookup = {}
    if "user_id" in users_df.columns:
        for _, urow in users_df.iterrows():
            user_lookup[urow["user_id"]] = urow

    attack_paths: List[Dict[str, Any]] = []

    for _, row in high_risk.iterrows():
        uid = str(row["user_id"])
        score = float(row.get("anomaly_score", 0))
        risk = str(row.get("risk_level", ""))

        # Compute blast radius
        blast = compute_blast_radius(G, uid)

        # Enrich with user details
        udata = user_lookup.get(uid, {})
        if isinstance(udata, pd.Series):
            username = str(udata.get("username", ""))
            priv = str(udata.get("privilege_level", ""))
            dept = str(udata.get("department", ""))
            sys_access = str(udata.get("systems_access", ""))
        else:
            username = ""
            priv = ""
            dept = ""
            sys_access = ""

        # Risk assessment
        exposure = blast["total_exposure_score"]
        if exposure >= 50:
            path_risk = "CRITICAL"
            recommendation = (
                f"CRITICAL exposure: {uid} has a total exposure score of "
                f"{exposure}. Immediately implement network segmentation, "
                f"revoke unnecessary entitlements, and enable continuous "
                f"monitoring on all {blast['direct_system_count']} direct systems."
            )
        elif exposure >= 25:
            path_risk = "HIGH"
            recommendation = (
                f"HIGH exposure: {uid} can reach "
                f"{blast['indirect_user_count']} other users and "
                f"{blast['indirect_system_count']} indirect systems. "
                f"Prioritise privilege reduction and implement MFA for "
                f"all shared-system access."
            )
        elif exposure >= 10:
            path_risk = "MEDIUM"
            recommendation = (
                f"MODERATE exposure: {uid} has access to "
                f"{blast['direct_system_count']} systems with an exposure "
                f"score of {exposure}. Review entitlements in next "
                f"access certification cycle."
            )
        else:
            path_risk = "LOW"
            recommendation = (
                f"LOW exposure: {uid}'s blast radius is contained. "
                f"Maintain standard monitoring."
            )

        attack_paths.append({
            "user_id": uid,
            "username": username,
            "department": dept,
            "privilege_level": priv,
            "anomaly_score": score,
            "risk_level": risk,
            "path_risk_level": path_risk,
            "blast_radius": blast,
            "recommendation": recommendation,
        })

    # Sort by exposure score descending
    attack_paths.sort(
        key=lambda x: x["blast_radius"]["total_exposure_score"],
        reverse=True,
    )

    logger.info(
        "Attack path analysis complete for %d CRITICAL/HIGH-risk users",
        len(attack_paths),
    )

    return attack_paths


# ══════════════════════════════════════════════════════════════════════════════
# CRITICAL PATH FINDER
# ══════════════════════════════════════════════════════════════════════════════

def find_critical_paths(
    G: nx.Graph,
    scored_df: pd.DataFrame,
) -> List[Dict[str, Any]]:
    """Identify the highest-risk paths through the privilege graph.

    A "critical path" connects a high-risk user to the most sensitive
    systems through the fewest hops.  This function uses BFS to enumerate
    paths from high-risk users to high-sensitivity systems and ranks them.

    Parameters
    ----------
    G : nx.Graph
        Privilege graph.
    scored_df : pd.DataFrame
        Scored user data with ``user_id``, ``anomaly_score``, ``risk_level``.

    Returns
    -------
    list[dict]
        Top critical paths, each with ``source_user``, ``target_system``,
        ``path``, ``hops``, ``path_risk_score``.
    """
    # High-value target systems (weight >= 7)
    high_value_systems = {s for s, w in SENSITIVITY_WEIGHTS.items() if w >= 7}

    # High-risk users
    high_risk = scored_df[scored_df["risk_level"].isin(["CRITICAL", "HIGH"])]

    critical_paths: List[Dict[str, Any]] = []

    for _, row in high_risk.iterrows():
        uid = str(row["user_id"])
        user_score = float(row.get("anomaly_score", 0))

        if uid not in G:
            continue

        # BFS from user to find paths to high-value systems
        for target_system in high_value_systems:
            if target_system not in G:
                continue

            try:
                path = nx.shortest_path(G, source=uid, target=target_system)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            hops = len(path) - 1
            # Path risk = user anomaly score × system weight / hops (penalise indirect)
            sys_weight = _get_weight(target_system)
            path_risk = (user_score * sys_weight) / max(hops, 1)

            # Describe the path
            path_description = []
            for i, node in enumerate(path):
                if str(node).startswith("USR"):
                    path_description.append({"hop": i, "type": "user", "entity": str(node)})
                else:
                    path_description.append({
                        "hop": i, "type": "system", "entity": str(node),
                        "sensitivity_weight": _get_weight(str(node)),
                    })

            critical_paths.append({
                "source_user": uid,
                "source_anomaly_score": user_score,
                "target_system": target_system,
                "target_sensitivity": sys_weight,
                "path": path_description,
                "hops": hops,
                "path_risk_score": round(path_risk, 1),
                "is_direct": hops == 1,
                "description": (
                    f"{'Direct' if hops == 1 else f'{hops}-hop'} path from "
                    f"{uid} (score={user_score}) to {target_system} "
                    f"(sensitivity={sys_weight}). Path risk: {path_risk:.1f}."
                ),
            })

    # Sort by path_risk_score descending and take top results
    critical_paths.sort(key=lambda x: x["path_risk_score"], reverse=True)

    # Deduplicate: keep top path per (user, system) pair
    seen: Set[Tuple[str, str]] = set()
    unique_paths: List[Dict[str, Any]] = []
    for p in critical_paths:
        key = (p["source_user"], p["target_system"])
        if key not in seen:
            seen.add(key)
            unique_paths.append(p)

    logger.info(
        "Critical path analysis found %d unique paths (%d total before dedup)",
        len(unique_paths), len(critical_paths),
    )

    return unique_paths[:50]  # Cap at top 50
