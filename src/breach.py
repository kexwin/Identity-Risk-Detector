"""
Bonus S4: Breach Impact Simulation
Simulates the business and compliance impact if an account is compromised.
Incorporates blast radius, systems exposed, reversibility, and compliance risks.
"""

import pandas as pd
import networkx as nx
from src.novelty import run_reversibility_simulation


def simulate_user_breach(
    user_id: str,
    users_df: pd.DataFrame,
    events_df: pd.DataFrame,
    G: nx.DiGraph,
    user_metrics: dict
) -> dict:
    """
    Simulates the impact of a breach for a specific user.
    """
    if user_id not in users_df.index:
        return {"error": f"User {user_id} not found."}

    user_row = users_df.loc[user_id]
    username = user_row["username"]
    department = user_row["department"]
    privilege = user_row["privilege_level"]
    job_title = user_row["job_title"]
    
    # Get systems accessed
    systems_list = user_row["systems_list"] if isinstance(user_row["systems_list"], list) else []
    
    # Calculate blast radius from user_row
    blast_radius = int(user_row.get("blast_radius", 0))
    system_count = int(user_row.get("system_count", 0))
    
    # Run reversibility to see which accesses are unused and safe to revoke (minimization potential)
    user_events = events_df[events_df["user_id"] == user_id]
    reversibility = run_reversibility_simulation(user_id, systems_list, user_events)
    
    # Calculate potential blast radius reduction if unused access is revoked

    revocable_systems = [
        item["system"]
        for item in reversibility
        if item.get("safe_to_revoke", False)
]

    revocable_weight = 0

    for sys in revocable_systems:
        if G.has_node(sys):
            revocable_weight += G.nodes[sys].get("weight", 1)
            
    minimized_blast_radius = blast_radius - revocable_weight
    
    # Identify critical / sensitive systems exposed
    high_sens_systems = [
        "PROD_DB", "PROD-DB", "ADMIN_SYS", "SIEM",
        "CUSTOMER_VAULT", "CUSTOMER_PII", "ADMIN_CONSOLE", "GL_SYSTEM"
    ]
    exposed_sensitive = [sys for sys in systems_list if any(hs in sys.upper() for hs in high_sens_systems)]
    
    # Determine GDPR / Compliance impact
    compliance_risks = []
    if "CUSTOMER_VAULT" in [s.upper() for s in systems_list] or "CUSTOMER_PII" in [s.upper() for s in systems_list]:
        compliance_risks.append({
            "framework": "GDPR Art. 32 / CCPA",
            "risk_type": "Exposure of Customer PII / Vault",
            "description": "User has direct access to raw customer records, posing high data leak liability and regulatory fines."
        })
    if "PROD_DB" in [s.upper() for s in systems_list] or "PROD-DB" in [s.upper() for s in systems_list]:
        compliance_risks.append({
            "framework": "SOX / GDPR Art. 25",
            "risk_type": "Production Database Access",
            "description": "Direct read/write access to production database, bypassing standard deployment pipelines."
        })
    if "GL_SYSTEM" in [s.upper() for s in systems_list]:
        compliance_risks.append({
            "framework": "SOX / Internal Financial Controls",
            "risk_type": "General Ledger Access",
            "description": "Ability to modify financial records or alter balances in the General Ledger system."
        })
    if "SIEM" in [s.upper() for s in systems_list] or "ADMIN_CONSOLE" in [s.upper() for s in systems_list]:
        compliance_risks.append({
            "framework": "ISO 27001 / SOC 2",
            "risk_type": "Security Administration Access",
            "description": "Compromise of the SIEM or Admin Console enables an attacker to disable logging or cover their tracks."
        })

    # Compare with department peer median
    dept_users = users_df[users_df["department"] == department]
    dept_blast_radii = dept_users.get("blast_radius", pd.Series(dtype=float)).dropna().tolist()
    dept_median_blast = float(pd.Series(dept_blast_radii).median()) if len(dept_blast_radii) > 0 else 0.0
    
    deviation_from_peer = blast_radius - dept_median_blast

    return {
        "user_id": user_id,
        "username": username,
        "department": department,
        "job_title": job_title,
        "privilege_level": privilege,
        "blast_radius_score": blast_radius,
        "system_count": system_count,
        "systems_exposed": systems_list,
        "exposed_sensitive_systems": exposed_sensitive,
        "compliance_risks": compliance_risks,
        "peer_median_blast_radius": dept_median_blast,
        "blast_radius_deviation": deviation_from_peer,
        "reversibility_analysis": reversibility,
        "safe_to_revoke_systems": revocable_systems,
        "potential_blast_radius_reduction": revocable_weight,
        "minimized_blast_radius": minimized_blast_radius
    }
