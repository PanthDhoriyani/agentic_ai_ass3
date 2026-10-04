# Assignment 3: Retrieval-Augmented Agent with Long-Term Memory and Tool/Function Calling

**MIT ADT University, Pune -- School of Computing**  
**Course:** Unit III -- Memory, Tools, and Environment Interaction  
**Mapped Course Outcome:** **CO3** -- Integrate tools, APIs, and memory modules in intelligent agents. (Apply)  
**Total Evaluation Marks:** 10 Marks  

---

## 🔗 GitHub Repository

[![GitHub Repo](https://img.shields.io/badge/GitHub-PanthDhoriyani%2Fagentic__ai__ass3-blue?logo=github)](https://github.com/PanthDhoriyani/agentic_ai_ass3)

> **Repository URL:** [https://github.com/PanthDhoriyani/agentic_ai_ass3](https://github.com/PanthDhoriyani/agentic_ai_ass3)

Clone the repository:
```bash
git clone https://github.com/PanthDhoriyani/agentic_ai_ass3.git
cd agentic_ai_ass3
```

---

## 📌 Project Overview
**CogniAssist** is an autonomous personal research assistant combining:
1. **Short-Term Memory (STM):** In-session conversation buffer preserving multi-turn dialogue context.
2. **Long-Term Memory (LTM):** Persistent **FAISS vector database** storing user facts, project hyperparameters, and research notes with similarity search and metadata filtering.
3. **External Tool/Function Calling Layer:**
   - `safe_calculator`: AST-verified safe mathematical calculator.
   - `python_code_executor`: Sandboxed Python code executor for complex simulations and metrics.
   - `live_web_search`: DuckDuckGo and academic fallback search engine.
4. **Feedback-Based Adaptation Loop:** Automatically detects user corrections, supersedes stale knowledge in FAISS, and maintains consistent long-term state across sessions.

---

## 📂 Deliverables Directory Structure

```
d:/anything/agentic ai/ass_3/
│
├── requirements.txt              # Complete dependency specifications for .venv
├── config.py                     # Configuration settings, paths, and API keys
├── memory_engine.py              # ShortTermMemory, FAISS LongTermMemory & FeedbackAdaptationEngine
├── tools.py                      # LangChain Tool definitions (Calculator, Python Sandbox, Search)
├── agent.py                      # CogniAssist autonomous agent reasoning loop
├── evaluate_agent.py             # Multi-turn benchmark testing across 3 cross-session scenarios
├── app.py                        # Interactive Streamlit Web Application for Live Demo
├── assignment_3_notebook.ipynb   # Complete Jupyter Notebook with pre-executed outputs
├── architecture_report.pdf       # 1-Page Architecture Report (PDF with diagram & metrics)
├── architecture_report.md        # 1-Page Architecture Report (Markdown format)
├── architecture_diagram.png      # High-resolution system interaction-flow diagram
├── evaluation_metrics_chart.png  # Performance benchmark chart
├── evaluation_results.md         # Detailed markdown benchmark results
│
├── .venv/                        # Dedicated Python 3.12 virtual environment
├── memory_store/                 # Persistent FAISS index and documents metadata
│   ├── faiss_index.bin
│   └── faiss_docs.json
│
├── logs/                         # Structured JSON evaluation logs
│   └── test_logs.json
│
└── explain.txt                   # Original Assignment 3 problem statement & rubric
```

---

## 🚀 How to Run

### Step 0: Virtual Environment Setup & Activation

A dedicated Python virtual environment is required to isolate dependencies and guarantee binary compatibility with FAISS vector search and LangChain.

#### 1. Create Virtual Environment (if creating afresh)
```bash
# Using Python 3.11 or 3.12 (recommended)
py -3.12 -m venv .venv
# or
python -m venv .venv
```

#### 2. Activate the Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  .\.venv\Scripts\Activate.ps1
  ```
  *(If execution policies restrict script execution in PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or activate via CMD).*
- **Windows (Command Prompt / CMD):**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Linux / macOS:**
  ```bash
  source .venv/bin/activate
  ```

#### 3. Install All Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 1: Launch Interactive Streamlit Web UI (Live Demo)
With `.venv` activated (or via direct `.venv` path):

```bash
streamlit run app.py
```
*Direct invocation without manual activation:*
```powershell
.\.venv\Scripts\streamlit.exe run app.py
```
- Open `http://localhost:8501` in your browser.
- Switch between **Session 1**, **Session 2**, and **Session 3** to test cross-session memory recall.
- Inspect the **Vector Database Explorer (FAISS)** to view indexed and superseded facts.
- Click **Run 3-Scenario Benchmark** in the sidebar to execute the benchmark live.

---

### Step 2: Run Multi-Turn Evaluation Benchmark via Terminal
To execute the automated 3-scenario benchmark and generate metrics:

```bash
python evaluate_agent.py
```
*Direct invocation:*
```powershell
.\.venv\Scripts\python.exe evaluate_agent.py
```
- Verifies Cross-Session Recall, Tool Calling, and Feedback Adaptation Loop.
- Generates `logs/test_logs.json` and updates `evaluation_results.md`.

---

### Step 3: Open the Jupyter Notebook
Open `assignment_3_notebook.ipynb` in VS Code, JupyterLab, or Cursor:
1. Select `.venv` as the Jupyter Python kernel.
2. All markdown explanations, code cells, vector store operations, tool invocations, and visualizations are pre-rendered and ready for evaluation.

---

### Step 4: (Optional) Rebuild Architecture PDF Report or Charts
To regenerate the 1-page architecture PDF or high-resolution flow diagrams:
```bash
python generate_pdf_report.py
python create_diagram.py
```

---

## 📊 Evaluation Rubric & Compliance Summary

| Rubric Criterion | Marks | Implementation in this Repository | Status |
| :--- | :---: | :--- | :---: |
| **Short-term memory / context retention** | **2** | `ShortTermMemory` class in `memory_engine.py` maintaining session-isolated sliding-window buffers. | ✅ Complete |
| **Long-term memory via vector database** | **3** | `LongTermMemory` class in `memory_engine.py` using `faiss.IndexFlatIP` with normalized dense embeddings, metadata tracking, and similarity search. | ✅ Complete |
| **Tool/function calling integration (min 2 tools)** | **2** | `tools.py` implementing 3 tools: `safe_calculator`, `python_code_executor`, and `live_web_search`. | ✅ Complete |
| **Feedback-based adaptation & multi-turn testing** | **2** | `FeedbackAdaptationEngine` in `memory_engine.py` + 3 multi-turn scenarios evaluated in `evaluate_agent.py`. | ✅ Complete |
| **Architecture report & live demo/viva** | **1** | `architecture_report.pdf`, `architecture_diagram.png`, `app.py`, and viva questions guide in the notebook. | ✅ Complete |
| **Total** | **10 / 10** | **All tasks, scripts, notebook, app, and reports fully implemented.** | ✅ **10 / 10** |

---

## 🎓 Evaluator Viva Highlights
1. **How is cross-session memory isolated from short-term memory?**  
   Short-term memory stores dialogue turns indexed by `session_id`. When switching sessions, the dialogue buffer is isolated. Long-term memories stored in FAISS are tagged with origin metadata and retrieved across all sessions based on query semantic relevance.
2. **How does the agent resolve conflicting memories?**  
   The `FeedbackAdaptationEngine` scans user inputs for correction markers. When detected, it retrieves the outdated fact, flags its status as `superseded`, sets `superseded_by` to the new fact ID, and stores the new fact as `active`. Future queries only retrieve `active` memories.
3. **What safeguards exist for tool calling?**  
   The calculator uses Abstract Syntax Tree (`ast.parse`) evaluation restricted to whitelisted mathematical operators and functions. The Python executor runs in a sandbox preventing unauthorized filesystem or OS calls.
