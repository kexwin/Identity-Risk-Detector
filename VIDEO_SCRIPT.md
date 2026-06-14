# Identity Risk Detector: 2-Minute Demo Video Script

**[0:00 - 0:15] Introduction & The Priority Queue**
*Visual: Start on the main Streamlit dashboard, showing the "Priority Queue" tab.*
**Speaker:** "Hi everyone, we built the Identity Risk Detector to solve the massive problem of identity sprawl and privilege abuse. What you're looking at here is our Priority Queue. Instead of drowning SOC analysts in thousands of raw logs, our Isolation Forest machine learning model scores every user and strictly sorts them by Risk Level—ensuring we focus only on the most critical threats first."

**[0:15 - 0:45] Deep-Dive Investigator & LLM Integration**
*Visual: Click on the "Deep-Dive User Risk Investigator" dropdown and select `zainab.moore (CRITICAL)` or `edward.esposito`.*
**Speaker:** "When an analyst selects a high-risk user, our system instantly builds a comprehensive profile. Rather than manually querying SIEM logs, we use a Large Language Model to auto-generate a SOC Analyst Narrative. It explicitly tells us *why* this account was flagged—for instance, unusual off-hours administrative access—and provides immediate mitigation steps. We also have a False Positive feedback loop. If we mark this as a false positive, the system adjusts future anomaly scores dynamically."

**[0:45 - 1:15] Breach Impact Simulation & Reversibility**
*Visual: Scroll down and click the "Simulate Compromise" button.*
**Speaker:** "A unique feature we built is the Breach Impact Simulator. By clicking this, we calculate the user's 'Blast Radius'—the collateral damage if this account is compromised. It maps exposed systems to specific compliance frameworks like GDPR and SOC 2. Even better, our Reversibility Engine looks at 365 days of historical event logs and flags systems that the user has access to, but hasn't actually used, marking them as 'Safe to Revoke' to enforce zero-trust access minimization."

**[1:15 - 1:35] Interactive Graph**
*Visual: Click on the "Privilege Access Graph" tab.*
**Speaker:** "To visualize this Blast Radius, we built an interactive Bipartite Privilege Graph using NetworkX and PyVis. Here, analysts can visually trace the connections between users and systems. You can immediately see the central nodes and identify which systems create the most structural risk."

**[1:35 - 1:55] Behavioral Clustering & Compliance**
*Visual: Quickly click the "Behavioral Clusters" tab, then the "Compliance Status" tab.*
**Speaker:** "We also apply K-Means clustering and PCA to group risky users into behavioral archetypes, making it easy to spot coordinated insider threats or dormant admin campaigns. Finally, our compliance tab actively audits for Separation of Duty violations—like an engineer having access to both Production databases and the Admin console simultaneously."

**[1:55 - 2:00] Conclusion**
*Visual: Return to the main Priority Queue.*
**Speaker:** "By combining unsupervised ML with deep context and LLM analysis, we've turned overwhelming identity logs into actionable, prioritized intelligence. Thank you!"
