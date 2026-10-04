"""
Streamlit Web Application for Assignment 3:
Retrieval-Augmented Agent with Short-Term Memory, Long-Term FAISS Vector Database,
Tool / Function Calling, and Feedback-Based Adaptation.
MIT ADT University -- Unit III: Memory, Tools, and Environment Interaction.
"""

import json
import os
import streamlit as st
import pandas as pd

from config import GOOGLE_API_KEY, GROQ_API_KEY, BENCHMARK_LOG_FILE, EVALUATION_REPORT_FILE
from memory_engine import ShortTermMemory, LongTermMemory, FeedbackAdaptationEngine
from tools import ALL_TOOLS, safe_calculator, python_code_executor, live_web_search
from agent import CogniAssistAgent
from evaluate_agent import run_evaluation_benchmark

# Page setup
st.set_page_config(
    page_title="CogniAssist | Memory & Tool Agent",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #1E88E5, #7E57C2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1.0rem;
        margin-bottom: 1.5rem;
    }
    .badge-ltm {
        background-color: #e3f2fd;
        color: #0d47a1;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #90caf9;
        display: inline-block;
        margin-right: 6px;
    }
    .badge-tool {
        background-color: #ede7f6;
        color: #4a148c;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #b39ddb;
        display: inline-block;
        margin-right: 6px;
    }
    .badge-adapt {
        background-color: #fff3e0;
        color: #e65100;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid #ffcc80;
        display: inline-block;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "agent" not in st.session_state:
    st.session_state.agent = CogniAssistAgent()

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = "Session_1"

if "sessions_list" not in st.session_state:
    st.session_state.sessions_list = ["Session_1", "Session_2", "Session_3"]

if "chat_history" not in st.session_state:
    # session_id -> list of message dicts
    st.session_state.chat_history = {
        "Session_1": [],
        "Session_2": [],
        "Session_3": []
    }


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.title("🧠 CogniAssist Controls")
    st.markdown("**Unit III: Memory & Tools Agent**")
    st.caption("MIT ADT University -- Assignment 03")
    
    st.markdown("---")
    st.subheader("🔀 Multi-Session Manager")
    
    selected_session = st.selectbox(
        "Active Working Session:",
        options=st.session_state.sessions_list,
        index=st.session_state.sessions_list.index(st.session_state.current_session_id)
    )
    if selected_session != st.session_state.current_session_id:
        st.session_state.current_session_id = selected_session
        st.rerun()

    new_sess_col1, new_sess_col2 = st.columns([2, 1])
    with new_sess_col1:
        new_sess_name = st.text_input("New Session Name", placeholder="e.g. Session_4", label_visibility="collapsed")
    with new_sess_col2:
        if st.button("➕ Add"):
            if new_sess_name and new_sess_name not in st.session_state.sessions_list:
                st.session_state.sessions_list.append(new_sess_name)
                st.session_state.chat_history[new_sess_name] = []
                st.session_state.current_session_id = new_sess_name
                st.rerun()

    st.markdown("---")
    st.subheader("⚙️ Agent & LLM Backend")
    api_key_input = st.text_input("Gemini API Key (Optional)", type="password", value=GOOGLE_API_KEY)
    if api_key_input and api_key_input != st.session_state.agent.google_key:
        st.session_state.agent = CogniAssistAgent(google_api_key=api_key_input)
        st.success("API key updated!")

    mode_label = "🟢 Live LLM (Gemini)" if (st.session_state.agent.llm is not None) else "⚡ Deterministic Local Agent (Zero-Latency / Offline)"
    st.caption(f"**Execution Mode:** {mode_label}")

    st.markdown("---")
    st.subheader("🛠️ Memory Controls")
    if st.button("🧹 Reset All Memories", use_container_width=True):
        st.session_state.agent.ltm.clear_all()
        for s in st.session_state.chat_history:
            st.session_state.chat_history[s] = []
            st.session_state.agent.stm.clear_session(s)
        st.success("All memories wiped clean!")
        st.rerun()

    if st.button("🚀 Run 3-Scenario Benchmark", use_container_width=True, type="primary"):
        with st.spinner("Executing multi-turn cross-session benchmark..."):
            eval_results = run_evaluation_benchmark()
            st.session_state.eval_results = eval_results
            st.success("Benchmark completed! Check the 'Benchmark & Metrics' tab.")


# ==========================================
# MAIN INTERFACE TABS
# ==========================================
st.markdown('<div class="main-header">CogniAssist: Memory & Tool-Calling Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Session Context Retention • FAISS Long-Term Memory • Function Calling • Feedback Adaptation</div>', unsafe_allow_html=True)

tab_chat, tab_ltm, tab_tools, tab_eval = st.tabs([
    "💬 Multi-Session Agent Chat",
    "🧠 Vector Database Explorer (FAISS)",
    "🛠️ Tools & Function Calling",
    "📊 Benchmark & Metrics Report"
])


# -------------------------------------------------------------
# TAB 1: MULTI-SESSION CHAT INTERFACE
# -------------------------------------------------------------
with tab_chat:
    active_session = st.session_state.current_session_id
    st.info(f"**Active Session:** `{active_session}` (Short-term buffer holds conversation within this session; Long-term FAISS retains memories across all sessions)")

    # Sample prompt buttons
    st.caption("💡 Quick test prompts across sessions:")
    chip_cols = st.columns(3)
    with chip_cols[0]:
        if st.button("1️⃣ Introduce Researcher Fact (Session 1)", use_container_width=True):
            st.session_state.pending_prompt = "I am an AI researcher at MIT ADT Pune specializing in Graph Neural Networks."
    with chip_cols[1]:
        if st.button("2️⃣ Recall Across Sessions (Switch to Session 2 first)", use_container_width=True):
            st.session_state.pending_prompt = "What is my research domain and university?"
    with chip_cols[2]:
        if st.button("3️⃣ Tool Calling: Calc & Python Script", use_container_width=True):
            st.session_state.pending_prompt = "Calculate steps per epoch for 45000 samples with batch size 64, and write a quick python script to estimate memory consumption."

    # Render Conversation Messages
    current_msgs = st.session_state.chat_history.get(active_session, [])
    for msg in current_msgs:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show badges and details for assistant responses
            if msg["role"] == "assistant":
                meta = msg.get("metadata", {})
                
                # Retrieved Memories Badge
                if meta.get("retrieved_memories"):
                    with st.expander(f"🔍 Recalled {len(meta['retrieved_memories'])} Long-Term Memories (FAISS)", expanded=False):
                        for m in meta["retrieved_memories"]:
                            st.markdown(f"- **Fact:** `{m['text']}`  \n  *Score:* `{m['similarity_score']}` | *Origin:* `{m['session_id']}`")
                
                # Tools Invoked Badge
                if meta.get("tools_called"):
                    with st.expander(f"🛠️ Executed {len(meta['tools_called'])} Tool(s)", expanded=False):
                        for t in meta["tools_called"]:
                            st.markdown(f"**Tool:** `{t['tool']}`  \n**Status:** `{t['status']}`  \n**Args:** `{t['args']}`  \n**Output:**")
                            st.code(t["result"])
                
                # Feedback Adaptation Badge
                if meta.get("adaptation_record") and meta["adaptation_record"].get("adapted"):
                    adapt = meta["adaptation_record"]
                    st.markdown(f"""<span class="badge-adapt">🔄 Feedback Adaptation Triggered: Action = {adapt.get('action')}</span>""", unsafe_allow_html=True)
                    if adapt.get("old_memory"):
                        st.caption(f"Superseded ID: `{adapt['old_memory']['id']}`")

    # Handle Input
    user_input = st.chat_input("Ask CogniAssist or provide research notes / corrections...")
    if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
        user_input = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if user_input:
        # Add user message to session UI
        st.session_state.chat_history[active_session].append({
            "role": "user",
            "content": user_input
        })

        # Process through Autonomous Agent
        with st.spinner("CogniAssist reasoning, querying FAISS, and checking tools..."):
            response = st.session_state.agent.chat(user_input, session_id=active_session)

        # Add assistant response to session UI
        st.session_state.chat_history[active_session].append({
            "role": "assistant",
            "content": response.answer,
            "metadata": {
                "retrieved_memories": response.retrieved_memories,
                "tools_called": response.tools_called,
                "adaptation_record": response.adaptation_record,
                "stored_memories": response.stored_memories
            }
        })
        st.rerun()


# -------------------------------------------------------------
# TAB 2: VECTOR DATABASE EXPLORER (FAISS)
# -------------------------------------------------------------
with tab_ltm:
    st.subheader("🧠 Long-Term Memory (FAISS Vector Store)")
    st.markdown("All indexed persistent memories stored in FAISS with dense embeddings, metadata, and status.")

    all_mems = st.session_state.agent.ltm.get_all_memories()
    active_mems = [m for m in all_mems if m.get("status") == "active"]
    superseded_mems = [m for m in all_mems if m.get("status") == "superseded"]

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total Indexed Memories", len(all_mems))
    col_m2.metric("Active Memories", len(active_mems))
    col_m3.metric("Superseded Memories", len(superseded_mems))

    st.markdown("---")
    st.subheader("🔍 Vector Similarity Query Tester")
    query_col1, query_col2 = st.columns([3, 1])
    with query_col1:
        test_search = st.text_input("Test Query Vector Similarity", placeholder="e.g. conference deadline or research university")
    with query_col2:
        top_k_slider = st.slider("Top K", min_value=1, max_value=5, value=3)

    if test_search:
        results = st.session_state.agent.ltm.retrieve_relevant_facts(test_search, top_k=top_k_slider, min_similarity=0.0)
        if results:
            st.write(f"**Found {len(results)} matches:**")
            for r in results:
                st.markdown(f"""
                - **Text:** `{r['text']}`  
                  *Similarity Score:* **{r['similarity_score']}** | *Origin Session:* `{r['session_id']}` | *Status:* `{r['status']}`
                """)
        else:
            st.warning("No matches found above threshold.")

    st.markdown("---")
    st.subheader("📋 Memory Table")
    if all_mems:
        df_mems = pd.DataFrame([
            {
                "Memory ID": m["id"],
                "Fact Text": m["text"],
                "Origin Session": m["session_id"],
                "Topic": m["topic"],
                "Status": m["status"],
                "Superseded By": m.get("superseded_by") or "None",
                "Created At": m["created_at"]
            }
            for m in all_mems
        ])
        st.dataframe(df_mems, use_container_width=True)
    else:
        st.info("No memories in vector store yet. Use the chat to add facts or run the benchmark.")


# -------------------------------------------------------------
# TAB 3: TOOLS & FUNCTION CALLING
# -------------------------------------------------------------
with tab_tools:
    st.subheader("🛠️ External Tools & Function Calling Interface")
    st.markdown("The agent integrates at least 3 tools with Pydantic parameter validation:")

    tool_t1, tool_t2, tool_t3 = st.tabs(["🧮 Calculator", "🐍 Python Code Executor", "🌐 Web Search"])

    with tool_t1:
        st.markdown("**Safe Calculator Tool (`safe_calculator`)**")
        st.caption("Safely parses and evaluates algebraic expressions, logarithms, and powers.")
        calc_input = st.text_input("Expression:", value="45000 / 64")
        if st.button("Evaluate Expression"):
            res = safe_calculator.invoke({"expression": calc_input})
            st.success(res)

    with tool_t2:
        st.markdown("**Python Sandbox Executor Tool (`python_code_executor`)**")
        st.caption("Executes Python in a restricted namespace, capturing stdout and local variables.")
        code_input = st.text_area(
            "Python Code:",
            value="batch_size = 64\ndataset_size = 45000\nsteps_per_epoch = dataset_size // batch_size\nprint(f'Steps per epoch: {steps_per_epoch}')\nprint(f'Remainder: {dataset_size % batch_size}')"
        )
        if st.button("Execute Python Code"):
            res = python_code_executor.invoke({"code": code_input})
            st.code(res)

    with tool_t3:
        st.markdown("**Live Web Search Tool (`live_web_search`)**")
        st.caption("Retrieves real-time information via DuckDuckGo / Knowledge fallback.")
        search_input = st.text_input("Search Query:", value="MIT ADT University Pune")
        if st.button("Search Knowledge Base"):
            res = live_web_search.invoke({"query": search_input})
            st.markdown(res)


# -------------------------------------------------------------
# TAB 4: BENCHMARK & METRICS REPORT
# -------------------------------------------------------------
with tab_eval:
    st.subheader("📊 Multi-Turn Evaluation & Quantitative Metrics")
    
    if os.path.exists(BENCHMARK_LOG_FILE):
        with open(BENCHMARK_LOG_FILE, "r", encoding="utf-8") as f:
            eval_data = json.load(f)

        m = eval_data.get("overall_metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Retrieval Accuracy", f"{m.get('retrieval_accuracy_pct', 0)}%", delta="Target: >85%")
        col2.metric("Tool-Call Success", f"{m.get('tool_call_success_rate_pct', 0)}%", delta="Target: >90%")
        col3.metric("Adaptation Rate", f"{m.get('adaptation_success_rate_pct', 0)}%", delta="Target: 100%")
        col4.metric("Scenarios Passed", f"{m.get('scenarios_passed', 0)} / {m.get('total_scenarios', 3)}", delta="100% Pass")

        st.markdown("---")
        st.subheader("📋 Scenario-by-Scenario Validation Log")
        for sc in eval_data.get("scenarios", []):
            with st.expander(f"Scenario {sc['scenario_id']}: {sc['title']} — {'✅ PASSED' if sc['scenario_success'] else '❌ FAILED'}", expanded=True):
                st.markdown(f"**Description:** {sc['description']}")
                for i, t in enumerate(sc["turns"], 1):
                    st.markdown(f"**Turn {i} (Session: `{t.get('session')}`)**")
                    st.markdown(f"👤 **User:** `{t.get('user')}`")
                    st.markdown(f"🤖 **Assistant:**\n{t.get('assistant')}")
                    if t.get("retrieved_memories"):
                        st.caption(f"Retrieved: {[rm['text'] for rm in t['retrieved_memories']]}")
                    if t.get("tools_called"):
                        st.caption(f"Tools Invoked: {[tc['tool'] for tc in t['tools_called']]}")
                    if t.get("adaptation_record"):
                        st.caption(f"Adaptation: {t['adaptation_record'].get('action')}")
                    st.markdown("---")
    else:
        st.info("No benchmark run detected yet. Click 'Run 3-Scenario Benchmark' in the sidebar to generate live evaluation metrics!")
