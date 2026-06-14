"""
Bonus S5: Separation of Duty (SoD) Violation Detection
Scans all 300 users for compliance and security conflicts, mapped to GDPR Article 5(1)(f) (Integrity and Least Privilege).
"""

import pandas as pd


def get_sod_rules() -> list[dict]:
    """
    Returns the defined Separation of Duty conflict rules.
    """
    return [
        {
            "id": "SOD_DB_ADMIN",
            "name": "Production Database & System Admin Conflict",
            "systems_required": ["PROD_DB", "ADMIN_SYS"],
            "severity": "CRITICAL",
            "gdpr_ref": "GDPR Art. 5(1)(f) - Integrity & Confidentiality",
            "description": "User holds both PROD_DB and ADMIN_SYS access. This allows direct modifications to production databases alongside control of administrative infrastructure, allowing logs to be cleared."
        },
        {
            "id": "SOD_FINANCE_LEDGER",
            "name": "Finance Ledger & Admin Privilege Conflict",
            "systems_required": ["GL_SYSTEM"],
            "required_privileges": ["admin", "power-user"],
            "required_department": "Finance",
            "severity": "HIGH",
            "gdpr_ref": "GDPR Art. 5(1)(f) - Least Privilege",
            "description": "Finance user holds admin/power-user status and has access to the General Ledger (GL_System). This bypasses internal controls, allowing unauthorized financial ledger adjustments."
        },
        {
            "id": "SOD_PII_CLOUD",
            "name": "Customer Vault & Public Cloud Sprawl",
            "systems_required": ["CUSTOMER_VAULT"],
            "any_of_systems": ["GCP", "AWS_IAM", "AZURE_AD", "SALESFORCE"],
            "exclude_departments": ["IT", "Security"],
            "severity": "HIGH",
            "gdpr_ref": "GDPR Art. 5(1)(f) - Data Minimization",
            "description": "Non-technical user (not in IT/Security) has access to Customer_Vault containing PII and also possesses public cloud identity access, increasing risk of customer data exfiltration."
        },
        {
            "id": "SOD_SIEM_IDENTITY",
            "name": "Security SIEM & Identity Admin Conflict",
            "systems_required": ["SIEM"],
            "any_of_systems": ["AZURE_AD", "AWS_IAM", "OKTA", "ADMIN_CONSOLE"],
            "severity": "CRITICAL",
            "gdpr_ref": "GDPR Art. 5(1)(f) - Auditable Controls",
            "description": "User has access to the SIEM (security logs) and identity directory managers (Okta/Azure_AD/Admin_Console). A compromise allows identity compromise to go unrecorded."
        }
    ]


def detect_sod_violations(users_df: pd.DataFrame) -> tuple[dict[str, list[dict]], pd.Series]:
    """
    Scans the users dataframe for SoD violations.
    Returns:
        dict: user_id -> list of violations
        Series: number of violations per user
    """
    rules = get_sod_rules()
    if users_df.index.name == "user_id":
        uids = users_df.index
        is_index = True
    else:
        uids = users_df["user_id"]
        is_index = False

    violations_by_user = {uid: [] for uid in uids}
    violation_counts = pd.Series(0, index=uids)

    for idx, row in users_df.iterrows():
        user_id = idx if is_index else row["user_id"]
        systems = [s.upper().replace("-", "_") for s in row.get("systems_list", [])]
        privilege = row["privilege_level"]
        dept = row["department"]

        for rule in rules:
            match = True
            
            # Check required systems
            for req in rule.get("systems_required", []):
                if req.upper().replace("-", "_") not in systems:
                    match = False
                    break
            if not match:
                continue

            # Check required privileges
            if "required_privileges" in rule:
                if privilege not in rule["required_privileges"]:
                    match = False
            
            # Check required department
            if "required_department" in rule:
                if dept != rule["required_department"]:
                    match = False

            # Check exclude departments
            if "exclude_departments" in rule:
                if dept in rule["exclude_departments"]:
                    match = False

            # Check any of systems
            if "any_of_systems" in rule:
                any_systems = [s.upper().replace("-", "_") for s in rule["any_of_systems"]]
                if not any(s in systems for s in any_systems):
                    match = False

            if match:
                violations_by_user[user_id].append({
                    "rule_id": rule["id"],
                    "name": rule["name"],
                    "severity": rule["severity"],
                    "gdpr_ref": rule["gdpr_ref"],
                    "description": rule["description"]
                })
                violation_counts[user_id] += 1

    return violations_by_user, violation_counts
