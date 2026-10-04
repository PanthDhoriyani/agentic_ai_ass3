# Assignment 3: Agent Evaluation & Benchmark Metrics Report
**Course:** Unit III -- Memory, Tools, and Environment Interaction  
**Mapped Course Outcome:** CO3 -- Integrate tools, APIs, and memory modules in intelligent agents  
**Evaluation Date:** 2026-10-04 18:02:12

---

## 1. Executive Summary

| Metric | Target | Achieved Score | Status |
| :--- | :--- | :--- | :--- |
| **Retrieval Accuracy (Recall@k)** | > 85% | **100.0%** | ✅ PASSED |
| **Tool-Calling Success Rate** | > 90% | **100.0%** | ✅ PASSED |
| **Feedback Adaptation Success** | 100% | **100.0%** | ✅ PASSED |
| **Multi-Turn Scenario Pass Rate** | 3 / 3 | **3 / 3 (100.0%)** | ✅ PASSED |

---

## 2. Multi-Turn Test Scenarios Breakdown

### Scenario 1: Cross-Session Profile & Research Persona Recall
- **Goal:** Validate that personal researcher facts stated in Session 1 are successfully recalled in a completely separate Session 2.
- **Short-Term Turns in Session 1:** 2 turns buffered.
- **Session 2 Query:** `"What is my research domain and university?"`
- **Result:** **PASSED** (Retrieved: `"I am an AI researcher at MIT ADT Pune specializing in Graph Neural Networks"`).

### Scenario 2: Cross-Session Parameter Recall & Tool Calling
- **Goal:** Store training dataset parameters (45,000 samples, batch size 64, 25 epochs) in Session 1; in Session 2, recall parameters from FAISS and call `calculator` and `python_code_executor`.
- **Tools Called:**
  1. `calculator` -> Calculated steps per epoch: `45000 / 64 = 703.125`
  2. `calculator` -> Calculated total training steps: `25 * 703.125 = 17578.125`
  3. `python_code_executor` -> Executed sandbox VRAM script: `0.41 GB`
- **Result:** **PASSED** (Both tools executed successfully with 100% accuracy).

### Scenario 3: Feedback-Based Adaptation & Self-Correcting Memory
- **Goal:** Verify that a correction in Session 2 supersedes the outdated fact from Session 1, and only the active fact is recalled in Session 3.
- **Session 1 Fact:** Target conference: NeurIPS 2026, May 15.
- **Session 2 Feedback:** `"Actually, change of plans: my target conference is ICML 2026 and the deadline is June 1st, please update your memory."`
- **Adaptation Action:** Marked NeurIPS memory as `superseded`; stored ICML 2026 as `active`.
- **Session 3 Query:** `"Which conference am I targeting and what is my deadline?"`
- **Result:** **PASSED** (ICML 2026 recalled; NeurIPS suppressed).

---

## 3. Detailed Logs & Artifacts
- **Structured JSON Logs:** `logs/test_logs.json`
- **Vector DB Store:** `memory_store/faiss_index.bin` & `memory_store/faiss_docs.json`
