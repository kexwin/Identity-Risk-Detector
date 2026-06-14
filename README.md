# Identity Risk Detector 🛡️

A hybrid Machine Learning, rules-based, and LLM-narrated Identity Risk Detection pipeline and dashboard. Built to solve the massive problem of **identity sprawl** and **privilege abuse**, this system detects anomalous behaviors, prioritizes security analyst investigations, and simulates breach impacts under GDPR & SOC 2 compliance frameworks.

---

## 🌟 Key Features

1. **Priority Risk Queue**: An intelligent queue that strictly sorts users by Risk Severity (CRITICAL, HIGH, MEDIUM, LOW) to ensure SOC analysts focus on the most imminent threats first.
2. **Deep-Dive LLM Investigator**: Automatically synthesizes complex SIEM logs and behavioral anomalies into a readable narrative with actionable mitigation steps.
3. **Breach Impact Simulator**: Calculates the "Blast Radius" of a compromised account via 2-hop graph traversal, showing exactly which systems (and compliance frameworks) are exposed.
4. **Access Minimization Engine**: Reviews 365 days of historical logs to identify systems a user has access to but hasn't used, flagging them as "Safe to Revoke" (Zero Trust enforcement).
5. **Interactive Privilege Graph**: A bipartite network graph visualizing the connections between users and systems.
6. **Behavioral Clustering**: Uses K-Means and PCA to group risky users into behavioral archetypes (e.g., dormant admins, high-volume exporters).
7. **False Positive Feedback Loop**: Analysts can mark flags as False Positives, which dynamically adjusts future scoring using cosine similarity on behavioral vectors.

---

## 🏗️ System Architecture

The backend pipeline processes data through 6 core stages:

```text
[Raw Logs & Profiles] 
   │
   ▼
[1. Ingestion & Normalization] (Clean data, parse timestamps)
   │
   ▼
[2. Privilege Graph Construction] (Bipartite User ↔ System graph, Blast Radius)
   │
   ▼
[3. Feature Engineering] (Calculate 15 statistical features & department norms)
   │
   ▼
[4. ML Anomaly Scoring] (Dual Isolation Forests for Event-level & User-level anomalies)
   │
   ▼
[5. Context & Exception Layer] (Domain-policy rules + False Positive feedback loop)
   │
   ▼
[6. LLM Explanation Generator] (Generates SOC Analyst Narratives)
```

For detailed architectural and algorithmic documentation, please refer to [ARCHITECTURE_AND_DESIGN.md](ARCHITECTURE_AND_DESIGN.md).

---

## 🚀 Installation & Setup

Ensure you have Python 3.10+ installed.

### Option 1: Run Locally (Recommended for Development)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kexwin/Identity-Risk-Detector.git
   cd Identity-Risk-Detector
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the FastAPI backend (Optional, Local Engine fallback is built-in):**
   ```bash
   python -m uvicorn api.main:app --port 8000
   ```

4. **Start the Streamlit Dashboard:**
   ```bash
   python -m streamlit run dashboard/app.py
   ```
   Navigate to `http://localhost:8501` in your browser.

### Option 2: Docker Deployment

Deploy both the backend and frontend with Docker Compose:
```bash
docker-compose up --build
```
- API Server: `http://localhost:8000`
- Streamlit Dashboard: `http://localhost:8501`

---

## ⚖️ Compliance & Governance (GDPR & ISO 27001)

The pipeline actively enforces **Separation of Duty (SoD)** violations, such as:
- **Production DB & Admin Console Conflict**: Prevents a single user from altering production data and simultaneously deleting audit logs.
- **Finance Ledger & Admin Privilege**: Flags Finance users holding unchecked administrative access on the General Ledger.
- **Customer Vault & Non-Support Access**: Flags non-technical or non-support users possessing access to Customer PII.
- **SIEM & Identity Console Conflict**: Prevents users from having both SIEM log access and Active Directory/Okta admin access.

---

## 📂 Repository Structure

- `api/` - FastAPI backend application for serving graph data and simulations.
- `dashboard/` - Streamlit interactive user interface.
- `src/` - Core pipeline modules (ingest, graph, features, model, context, breach, novelty).
- `data/` - Input datasets and False Positive feedback storage.
- `tests/` - Unit tests for pipeline validation.
- `ARCHITECTURE_AND_DESIGN.md` - In-depth technical documentation.
- `VIDEO_SCRIPT.md` - Hackathon demo presentation script.

---
*Developed for the 48-Hour Security Hackathon.*
