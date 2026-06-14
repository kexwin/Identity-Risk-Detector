"""
Stage 2: Graph Construction
============================
Builds a NetworkX bipartite graph (users ↔ systems), computes
blast-radius and shared-system metrics, detects Separation-of-Duty
violations, and exports an interactive Pyvis HTML.
"""

import os
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Sensitive / conflict systems
# ---------------------------------------------------------------------------
_SENSITIVE_SYSTEMS = {"PROD_DB", "ADMIN_SYS", "SIEM", "Customer_Vault"}
_ADMIN_SYSTEMS = {"ADMIN_SYS"}


# ===================================================================
# 1. Build the bipartite privilege graph
# ===================================================================
def build_privilege_graph(users_df: pd.DataFrame) -> nx.Graph:
    """Create a bipartite graph with user-nodes and system-nodes.

    Parameters
    ----------
    users_df : pd.DataFrame
        Must contain ``user_id``, ``systems_list``, ``department``,
        ``privilege_level``, ``job_title``, ``days_inactive``.

    Returns
    -------
    nx.Graph
        Bipartite graph.  User nodes have ``bipartite=0`` and
        attributes ``department``, ``privilege_level``, ``job_title``,
        ``days_inactive``.  System nodes have ``bipartite=1`` and
        ``node_type='system'``.
    """
    G = nx.Graph()

    all_systems = set()
    for systems in users_df["systems_list"]:
        all_systems.update(systems)

    # Add system nodes first
    for sys_name in sorted(all_systems):
        G.add_node(
            sys_name,
            bipartite=1,
            node_type="system",
            label=sys_name,
        )

    # Add user nodes and edges
    for _, row in users_df.iterrows():
        uid = row["user_id"]
        G.add_node(
            uid,
            bipartite=0,
            node_type="user",
            department=row.get("department", "Unknown"),
            privilege_level=row.get("privilege_level", "user"),
            job_title=row.get("job_title", ""),
            days_inactive=int(row.get("days_inactive", 0)),
            label=row.get("username", uid),
        )
        for sys_name in row["systems_list"]:
            G.add_edge(uid, sys_name)

    user_count = sum(1 for _, d in G.nodes(data=True) if d.get("bipartite") == 0)
    sys_count = sum(1 for _, d in G.nodes(data=True) if d.get("bipartite") == 1)
    print(f"[Graph]  Built bipartite graph: {user_count} users, {sys_count} systems, {G.number_of_edges()} edges")
    return G


# ===================================================================
# 2. Compute graph metrics
# ===================================================================
def compute_graph_metrics(
    G: nx.Graph, users_df: pd.DataFrame
) -> Dict[str, Any]:
    """Derive blast-radius and shared-system-users for every user.

    Parameters
    ----------
    G : nx.Graph
        Bipartite privilege graph from :func:`build_privilege_graph`.
    users_df : pd.DataFrame
        User table (used only for the user-id list).

    Returns
    -------
    dict
        ``{ 'blast_radius': {user_id: int, …},
            'shared_system_users': {user_id: int, …} }``

        * **blast_radius** – number of *distinct other users* reachable
          within 2 hops (user → system → other_user).
        * **shared_system_users** – total count of other users who share
          at least one system with this user (same as blast_radius but
          kept explicit for downstream features).
    """
    user_nodes = [
        n for n, d in G.nodes(data=True) if d.get("bipartite") == 0
    ]

    blast_radius: Dict[str, int] = {}
    shared_system_users: Dict[str, int] = {}

    for uid in user_nodes:
        if uid not in G:
            blast_radius[uid] = 0
            shared_system_users[uid] = 0
            continue

        # Systems this user touches
        neighbour_systems = set(G.neighbors(uid))

        # Users reachable through those systems (excluding self)
        reachable_users = set()
        for sys_node in neighbour_systems:
            for other_user in G.neighbors(sys_node):
                if other_user != uid:
                    reachable_users.add(other_user)

        blast_radius[uid] = len(reachable_users)
        shared_system_users[uid] = len(reachable_users)

    # Also cover users that may not be in the graph at all
    all_user_ids = set(users_df["user_id"])
    for uid in all_user_ids - set(user_nodes):
        blast_radius[uid] = 0
        shared_system_users[uid] = 0

    print(f"[Graph]  Blast-radius  mean={np.mean(list(blast_radius.values())):.1f}  "
          f"max={max(blast_radius.values())}")
    return {
        "blast_radius": blast_radius,
        "shared_system_users": shared_system_users,
    }


# ===================================================================
# 3. Separation-of-Duty violation detection
# ===================================================================
def detect_sod_violations(users_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Detect Separation-of-Duty violations from the user table.

    Conflict rules
    ~~~~~~~~~~~~~~
    1. **PROD_DB + ADMIN_SYS** access together.
    2. **Finance dept + GL_System + admin/power-user** privilege.
    3. **HRIS + admin** privilege *outside* the HR department.
    4. **Customer_Vault + export_data capability + non-Support** dept.
       (Approximated as: Customer_Vault access + privilege >= power-user
       + dept not in {Support, Customer Support}).

    Parameters
    ----------
    users_df : pd.DataFrame
        Must contain ``user_id``, ``username``, ``systems_list``,
        ``department``, ``privilege_level``.

    Returns
    -------
    list[dict]
        Each dict: ``{'user_id', 'username', 'rule', 'details'}``.
    """
    violations: List[Dict[str, Any]] = []

    for _, row in users_df.iterrows():
        uid = row["user_id"]
        uname = row.get("username", uid)
        systems = set(row["systems_list"])
        dept = str(row.get("department", "")).strip()
        priv = str(row.get("privilege_level", "user")).strip()

        # Rule 1: PROD_DB + ADMIN_SYS
        if "PROD_DB" in systems and "ADMIN_SYS" in systems:
            violations.append({
                "user_id": uid,
                "username": uname,
                "rule": "PROD_DB + ADMIN_SYS",
                "details": (
                    f"User has access to both PROD_DB and ADMIN_SYS "
                    f"(dept={dept}, priv={priv})"
                ),
            })

        # Rule 2: Finance + GL_System + elevated privilege
        if (
            dept == "Finance"
            and "GL_System" in systems
            and priv in ("admin", "power-user")
        ):
            violations.append({
                "user_id": uid,
                "username": uname,
                "rule": "Finance + GL_System + elevated",
                "details": (
                    f"Finance user with GL_System access and "
                    f"privilege_level={priv}"
                ),
            })

        # Rule 3: HRIS + admin privilege outside HR
        if (
            "HRIS" in systems
            and priv == "admin"
            and dept not in ("HR", "Human Resources")
        ):
            violations.append({
                "user_id": uid,
                "username": uname,
                "rule": "HRIS + admin (non-HR)",
                "details": (
                    f"Non-HR user (dept={dept}) has HRIS access with "
                    f"admin privilege"
                ),
            })

        # Rule 4: Customer_Vault + export potential + non-Support
        if (
            "Customer_Vault" in systems
            and priv in ("admin", "power-user")
            and dept not in ("Support", "Customer Support")
        ):
            violations.append({
                "user_id": uid,
                "username": uname,
                "rule": "Customer_Vault + export + non-Support",
                "details": (
                    f"Non-Support user (dept={dept}) with Customer_Vault "
                    f"access and elevated privilege ({priv}), may export data"
                ),
            })

    print(f"[SOD]   Detected {len(violations)} separation-of-duty violations "
          f"across {len(set(v['user_id'] for v in violations))} users")
    return violations


# ===================================================================
# 4. Interactive HTML export (Pyvis)
# ===================================================================
def export_graph_html(
    G: nx.Graph,
    risk_results: Optional[pd.DataFrame] = None,
    output_path: str = "output/privilege_graph.html",
) -> str:
    """Export the bipartite privilege graph as an interactive Pyvis HTML.

    Parameters
    ----------
    G : nx.Graph
        Bipartite privilege graph.
    risk_results : pd.DataFrame, optional
        If provided, must contain ``user_id`` and ``risk_level``
        columns.  Nodes are colour-coded by risk level.
    output_path : str, default ``'output/privilege_graph.html'``
        Destination file path.

    Returns
    -------
    str
        Absolute path to the written HTML file.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        print("[Graph]  pyvis not installed - skipping HTML export. "
              "Install with: pip install pyvis")
        return ""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=False,
        notebook=False,
    )
    net.barnes_hut(gravity=-8000, spring_length=150)

    # Build a risk lookup
    risk_map: Dict[str, str] = {}
    if risk_results is not None and "risk_level" in risk_results.columns:
        risk_map = dict(
            zip(risk_results["user_id"], risk_results["risk_level"])
        )

    _RISK_COLORS = {
        "CRITICAL": "#e63946",
        "HIGH": "#f4a261",
        "MEDIUM": "#e9c46a",
        "LOW": "#2a9d8f",
    }
    _SYSTEM_COLOR = "#457b9d"
    _DEFAULT_USER_COLOR = "#a8dadc"

    for node, attrs in G.nodes(data=True):
        node_type = attrs.get("node_type", "user")
        label = attrs.get("label", str(node))

        if node_type == "system":
            net.add_node(
                node,
                label=label,
                color=_SYSTEM_COLOR,
                shape="box",
                size=25,
                title=f"System: {label}",
            )
        else:
            risk = risk_map.get(node, "LOW")
            color = _RISK_COLORS.get(risk, _DEFAULT_USER_COLOR)
            priv = attrs.get("privilege_level", "user")
            dept = attrs.get("department", "")
            size = {"admin": 22, "power-user": 18, "service-account": 16}.get(priv, 12)
            title = (
                f"User: {label}<br>"
                f"Dept: {dept}<br>"
                f"Privilege: {priv}<br>"
                f"Risk: {risk}<br>"
                f"Inactive days: {attrs.get('days_inactive', '?')}"
            )
            net.add_node(
                node,
                label=label,
                color=color,
                shape="dot",
                size=size,
                title=title,
            )

    for u, v in G.edges():
        net.add_edge(u, v, color="#555555")

    net.write_html(output_path)
    abs_path = os.path.abspath(output_path)
    print(f"[Graph]  Exported interactive graph -> {abs_path}")
    return abs_path


# ══════════════════════════════════════════════════════════════════════════════
# COMPATIBILITY & TEST HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def export_graph_to_html(G: nx.Graph, output_path: str, users_dict: Dict[str, Any]) -> str:
    """Helper to convert users_dict to DataFrame and call export_graph_html."""
    risk_results = pd.DataFrame.from_dict(users_dict, orient='index')
    if not risk_results.empty and 'user_id' not in risk_results.columns:
        risk_results = risk_results.rename_axis('user_id').reset_index()
    return export_graph_html(G, risk_results, output_path)


def build_access_graph(users_df: pd.DataFrame):
    """Bipartite privilege graph builder alias for backward compatibility/tests."""
    G = build_privilege_graph(users_df)
    metrics = compute_graph_metrics(G, users_df)
    return G, metrics


def get_system_weight(system_name: str) -> int:
    """System weight boundary helper for backward compatibility/tests."""
    weights = {"PROD_DB": 3, "Azure_AD": 2, "File_Share": 1}
    return weights.get(system_name, 1)

