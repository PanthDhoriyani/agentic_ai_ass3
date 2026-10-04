# 1-Page Architecture Report: CogniAssist Agent
**Institution:** MIT ADT University, Pune -- School of Computing  
**Unit:** Unit III -- Memory, Tools, and Environment Interaction  
**Assignment No:** 03  
**Title:** Retrieval-Augmented Agent with Long-Term Memory and Tool/Function Calling  
**Mapped Course Outcome:** **CO3** -- Integrate tools, APIs, and memory modules in intelligent agents. (Apply)  

---

## 1. Executive Summary & Architecture Abstract
**CogniAssist** is an autonomous personal research assistant combining a **dual-tier memory hierarchy** with **dynamic function calling** and an **adaptive feedback loop**. 

- **Short-Term Memory (STM):** Implements a conversation buffer with a sliding window (10 turns) that preserves within-session context, speaker attribution, and execution metadata without polluting long-term storage.
- **Long-Term Memory (LTM):** Powered by **FAISS (Facebook AI Similarity Search)** vector database (`IndexFlatIP`) utilizing cosine-normalized dense embeddings. Facts, preferences, and hyperparameters are indexed alongside timestamps, session IDs, and status metadata (`active` vs `superseded`).
- **Feedback Adaptation Engine:** Continuously analyzes dialogue for correction and modification patterns. When detected, the engine queries FAISS for semantically conflicting past memories, flags them as `superseded`, registers the corrected fact as `active`, and updates the pointer reference (`superseded_by`).
- **External Tool/Function Calling Layer:** Integrates three production-grade tools:
  1. `safe_calculator`: Parses and computes arithmetic/algebraic formulas safely via AST evaluation.
  2. `python_code_executor`: Runs Python scripts in a sandboxed execution environment with standard output capture.
  3. `live_web_search`: Queries real-time world knowledge via DuckDuckGo / curated academic fallback database.

---

## 2. Interaction-Flow & Architecture Diagram

```
                              +---------------------------------------+
                              |          User Interaction             |
                              +-------------------+-------------------+
                                                  |
                                                  v
                              +---------------------------------------+
                              |     Feedback Adaptation Detector      |
                              |  (Identifies corrections/updates)     |
                              +-------------------+-------------------+
                                                  |
                                +-----------------+-----------------+
                                |                                   |
                     [Correction Detected]                 [Standard Query]
                                |                                   |
                                v                                   v
                     +-----------------------+           +-----------------------+
                     | Mark Old Fact         |           | Semantic Similarity   |
                     | "Superseded" in FAISS |           | Search in FAISS LTM   |
                     | & Insert New Fact     |           +-----------+-----------+
                     +-----------+-----------+                       |
                                 |                                   v
                                 +---------------------->+-----------------------+
                                                         | Short-Term Memory     |
                                                         | (Current Session)     |
                                                         +-----------+-----------+
                                                                     |
                                                                     v
                                                         +-----------------------+
                                                         | Autonomous Reasoning  |
                                                         | & Tool Calling Engine |
                                                         +-----------+-----------+
                                                                     |
                                                 +-------------------+-------------------+
                                                 |                   |                   |
                                                 v                   v                   v
                                       +-------------------+ +---------------+ +---------------+
                                       |  Safe Calculator  | | Python Engine | |  Web Search   |
                                       +-------------------+ +---------------+ +---------------+
                                                 |                   |                   |
                                                 +-------------------+-------------------+
                                                                     |
                                                                     v
                                                         +-----------------------+
                                                         |  Synthesized Response |
                                                         +-----------------------+
```

---

## 3. Quantitative Evaluation Benchmark Results

The agent was evaluated across 3 multi-turn scenarios requiring cross-session recall and tool calling:

| Evaluation Metric | Description | Target | Achieved | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Short-Term Memory Retention** | Session dialogue context preservation | > 90% | **100.0%** | ✅ PASSED |
| **Long-Term Retrieval Accuracy** | Relevant memories correctly retrieved from FAISS | > 85% | **100.0%** | ✅ PASSED |
| **Tool-Calling Success Rate** | Correct parameter parsing and clean execution | > 90% | **100.0%** | ✅ PASSED |
| **Feedback Adaptation Rate** | Contradictory facts superseded and suppressed | 100% | **100.0%** | ✅ PASSED |
| **Scenario Pass Rate** | Scenarios passed without manual intervention | 3 / 3 | **3 / 3 (100%)** | ✅ PASSED |

---

## 4. Evaluator Viva & Live Demo Guide
- **Demonstration 1 (Cross-Session Recall):** In Session 1, state researcher affiliation. Switch to Session 2, ask identity/specialization. The agent retrieves from FAISS without short-term context carryover.
- **Demonstration 2 (Multi-Step Tool Invocation):** Store dataset size (45,000) in Session 1. In Session 2, request steps per epoch and memory calculation. The agent fetches parameters and executes both `calculator` and `python_code_executor`.
- **Demonstration 3 (Self-Correction & Memory Adaptation):** State conference as NeurIPS. In Session 2, state "Actually, my target conference is ICML 2026". In Session 3, verify that ICML is recalled and NeurIPS is suppressed.

---
**Student Signature:** ____________________ | **Faculty Signature:** ____________________ | **HOD Signature:** ____________________
