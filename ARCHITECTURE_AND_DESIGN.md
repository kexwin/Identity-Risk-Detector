# Identity Risk Detector: Architecture & Design Documentation

**GitHub Repository:** [https://github.com/kexwin/Identity-Risk-Detector](https://github.com/kexwin/Identity-Risk-Detector)

## 1. Platform Architecture

The Identity Risk Detector is built on a modular, hybrid architecture combining rule-based heuristics with machine learning algorithms. The system comprises two main components:
- **FastAPI Backend (Data Processing & ML Engine)**: Handles data ingestion, graph construction, feature engineering, and anomaly scoring.
- **Streamlit Dashboard (User Interface)**: Consumes the backend outputs and provides an interactive environment for SOC Analysts to investigate risks, apply exceptions, and view LLM-generated incident reports.

### Pipeline Stages
1. **Ingestion & Normalization**: Raw user profiles and access event logs are parsed. Time classifications (business hours, nights, weekends) are encoded.
2. **Privilege Graph Construction**: A bipartite graph (Users ↔ Systems) is constructed using NetworkX. Metrics such as node degree and blast radius (2-hop reachability) are calculated to assess structural risk.
3. **Feature Engineering**: Statistical features (e.g., department medians, activity percentiles) are derived to normalize behavior against peer groups.
4. **ML Anomaly Scoring**: Dual Isolation Forests score events and user profiles. Event anomalies are aggregated using a weighted decay model.
5. **Context & Exception Layer**: Domain-specific policies (e.g., Executive Exceptions, Seasonal Finance Windows, Active Service Accounts) are applied to adjust raw anomaly scores. A False Positive feedback loop uses cosine similarity to automatically adjust scores for users behaving like known false positives.
6. **Reporting & Explanation**: Anthropic Claude (or a deterministic local template fallback) synthesizes technical findings into natural language SOC narratives.

---

## 2. Analysis Algorithms

### Isolation Forest (Unsupervised ML)
We use `scikit-learn`'s Isolation Forest algorithm due to its efficiency in identifying anomalies in high-dimensional spaces without requiring labeled training data. 
- **Event-Level Forest**: Evaluates the time of day, day of week, resource sensitivity, and action types.
- **User-Level Forest**: Evaluates aggregated features (e.g., stale admin status, high-sensitivity exports, access to critical systems, blast radius).

### Bipartite Privilege Graph & Blast Radius
Using `NetworkX`, we model relationships between users and IT systems. The graph allows us to compute the "Blast Radius"—the number of secondary users an attacker could theoretically reach if a specific user account is compromised. This is calculated via 2-hop traversal (`User -> System -> Other Users`).

### Behavioral Clustering (K-Means)
We apply K-Means clustering on the flagged anomalous users, projecting their high-dimensional feature vectors into a 2D space using Principal Component Analysis (PCA). This helps group threats into behavioral archetypes (e.g., "Dormant Admins", "High-Volume Exporters", "Off-hours Operators").

### Identity Digital Twin & Reversibility Analysis
- **Digital Twin**: Compares an individual's privileges and system count against the modal profile of their departmental peers.
- **Access Minimization (Reversibility)**: Analyzes historical event logs against current permissions. If a user holds access to a system but has zero usage events in the trailing 365 days, the simulator flags the credential as "Safe to Revoke."

---

## 3. User Interface Design

The Streamlit interface was designed specifically for Security Operations Center (SOC) efficiency, focusing on **alert triage** and **investigation context**.

- **Dark Mode & Typography**: Uses a sleek, high-contrast dark theme (vibrant red/orange indicators for severity) to reduce eye strain during long SOC shifts. 
- **Priority Risk Queue**: Sorts accounts strictly by Risk Level severity (`CRITICAL` > `HIGH` > `MEDIUM` > `LOW`) and then by quantitative score, ensuring analysts focus on the most imminent threats first.
- **Deep-Dive Investigator**: An expandable control panel that provides the SOC Analyst with an immediate LLM-generated narrative of *why* the user was flagged, removing the need to manually query raw logs.
- **Breach Impact Simulation**: Allows the analyst to click a button and simulate the collateral damage if the selected account is compromised, mapping exposed systems directly to GDPR compliance violations.
- **Interactive Topography**: Uses Plotly and PyVis to render the bipartite network graph natively in the browser, allowing analysts to visually trace access paths.
