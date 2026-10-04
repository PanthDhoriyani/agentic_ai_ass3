"""
CogniAssist Agent: Retrieval-Augmented Autonomous Agent with Short-Term Memory,
Long-Term Vector Database (FAISS), Function/Tool Calling, and Feedback Adaptation Loop.
MIT ADT University -- Unit III: Memory, Tools, and Environment Interaction.
"""

import re
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from config import GOOGLE_API_KEY, GROQ_API_KEY, DEFAULT_GEMINI_MODEL
from memory_engine import ShortTermMemory, LongTermMemory, FeedbackAdaptationEngine
from tools import ALL_TOOLS, TOOL_MAP, safe_calculator, python_code_executor, live_web_search


class AgentResponse(BaseModel):
    answer: str
    session_id: str
    tools_called: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_memories: List[Dict[str, Any]] = Field(default_factory=list)
    stored_memories: List[Dict[str, Any]] = Field(default_factory=list)
    adaptation_record: Optional[Dict[str, Any]] = None
    short_term_turns_count: int = 0


class CogniAssistAgent:
    """
    Autonomous Personal Research & Coding Assistant.
    Maintains session-based short-term memory, vector-indexed long-term memory,
    dynamic function calling, and feedback-based self-adaptation.
    """
    def __init__(
        self,
        google_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        model_name: str = DEFAULT_GEMINI_MODEL
    ):
        self.google_key = google_api_key or GOOGLE_API_KEY
        self.groq_key = groq_api_key or GROQ_API_KEY
        self.model_name = model_name

        # Memory Subsystems
        self.stm = ShortTermMemory(max_turns=10)
        self.ltm = LongTermMemory()
        self.adaptation_engine = FeedbackAdaptationEngine(self.ltm)

        # Tools
        self.tools = ALL_TOOLS
        self.tool_map = TOOL_MAP

        # LLM Client initialization
        self.llm = self._init_llm()

    def _init_llm(self):
        """Initializes LangChain LLM if credentials are provided."""
        if self.google_key and len(self.google_key.strip()) > 5:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.google_key,
                    temperature=0.2
                )
            except Exception:
                pass
        elif self.groq_key and len(self.groq_key.strip()) > 5:
            try:
                from langchain_groq import ChatGroq
                return ChatGroq(
                    model_name="llama-3.3-70b-versatile",
                    groq_api_key=self.groq_key,
                    temperature=0.2
                )
            except Exception:
                pass
        return None

    def _extract_and_index_facts(self, user_input: str, session_id: str) -> List[Dict[str, Any]]:
        """
        Extracts salient personal or task facts from user input and registers them in FAISS LTM.
        """
        stored = []
        lowered = user_input.lower()
        
        # Patterns for factual personal/research information
        fact_patterns = [
            r"i am (?:an? )?([^.,!\n]+)",
            r"my research (?:domain|area|topic) is ([^.,!\n]+)",
            r"my university is ([^.,!\n]+)",
            r"dataset size:?\s*(\d[\d,]*\s*\w*)",
            r"batch size:?\s*(\d+)",
            r"training epochs:?\s*(\d+)",
            r"target conference (?:is )?([^.,!\n]+)",
            r"deadline (?:is )?([^.,!\n]+)",
            r"prefer(?:red)? (?:output|format|style|language) is? ([^.,!\n]+)"
        ]

        # If user states preferences, research topics, or parameters
        should_store = any(k in lowered for k in [
            "researcher", "university", "dataset", "batch size", "epochs",
            "deadline", "conference", "i am", "my project", "prefer"
        ])
        
        # Avoid storing raw questions as facts
        if should_store and not user_input.strip().endswith("?"):
            # Clean fact sentence
            clean_fact = user_input.strip()
            mem_id = self.ltm.store_fact(
                fact_text=clean_fact,
                session_id=session_id,
                topic="user_fact"
            )
            stored.append({"id": mem_id, "text": clean_fact})

        return stored

    def _determine_tools_needed(self, query: str, context: str) -> List[Dict[str, Any]]:
        """
        Determines which tools need to be called based on user query and recalled memory.
        """
        tools_to_run = []
        q_lowered = query.lower()
        comb_text = f"{query} {context}".lower()

        # 1. Calculator Trigger
        calc_keywords = ["calculate", "math", "steps per epoch", "total steps", "divide", "multiply", "+", "*", "/", "sum"]
        if any(k in q_lowered for k in calc_keywords):
            # Check if we have numbers in query or context
            # Look for dataset size & batch size in query + retrieved memory
            dataset_match = re.search(r"dataset\s*(?:size)?\s*(?:is|:)?\s*(\d[\d,]*)", comb_text)
            batch_match = re.search(r"batch\s*(?:size)?\s*(?:is|:)?\s*(\d+)", comb_text)
            epoch_match = re.search(r"epochs?\s*(?:is|:)?\s*(\d+)", comb_text)

            if dataset_match and batch_match:
                ds = int(dataset_match.group(1).replace(",", ""))
                bs = int(batch_match.group(1))
                ep = int(epoch_match.group(1)) if epoch_match else 1
                
                expr1 = f"{ds} / {bs}"
                expr2 = f"{ep} * ({ds} // {bs})"
                tools_to_run.append({
                    "tool": "calculator",
                    "args": {"expression": f"{ds} / {bs}"},
                    "description": f"Calculate steps per epoch: {expr1}"
                })
                if epoch_match:
                    tools_to_run.append({
                        "tool": "calculator",
                        "args": {"expression": f"{ep} * ({ds} / {bs})"},
                        "description": f"Calculate total training steps: {ep} * ({expr1})"
                    })
            else:
                # Direct mathematical expression extract
                math_expr_match = re.search(r"([\d\.\s\+\-\*\/\(\)\^%]{3,})", query)
                if math_expr_match:
                    expr = math_expr_match.group(1).strip()
                    if any(c in expr for c in "+-*/%"):
                        tools_to_run.append({
                            "tool": "calculator",
                            "args": {"expression": expr},
                            "description": f"Evaluate expression: {expr}"
                        })

        # 2. Python Code Executor Trigger
        code_keywords = ["python script", "code", "write a quick python", "calculate memory consumption", "execute python", "run script"]
        if any(k in q_lowered for k in code_keywords):
            code_snippet = (
                "def estimate_vram_gb(batch_size, hidden_dim=768, num_layers=12):\n"
                "    # Approximate model & activation memory in GB\n"
                "    params_mb = (num_layers * 12 * (hidden_dim ** 2) * 4) / (1024 ** 2)\n"
                "    act_mb = (batch_size * 512 * hidden_dim * 4) / (1024 ** 2)\n"
                "    total_gb = (params_mb + act_mb) / 1024\n"
                "    return round(total_gb, 3)\n\n"
                "vram = estimate_vram_gb(batch_size=64)\n"
                "print(f'Estimated Training VRAM required: {vram} GB')"
            )
            tools_to_run.append({
                "tool": "python_code_executor",
                "args": {"code": code_snippet},
                "description": "Execute VRAM estimation script"
            })

        # 3. Web Search Trigger
        search_keywords = ["search web", "latest", "look up", "news", "documentation for", "who is", "when is"]
        if any(k in q_lowered for k in search_keywords):
            # Extract search query
            clean_search_q = re.sub(r"^(search web for|look up|search for)\s*", "", query, flags=re.I).strip()
            tools_to_run.append({
                "tool": "web_search",
                "args": {"query": clean_search_q or query},
                "description": f"Search web for '{clean_search_q}'"
            })

        return tools_to_run

    def chat(self, user_input: str, session_id: str = "session_1") -> AgentResponse:
        """
        Full autonomous reasoning loop:
        1. Feedback Adaptation detection & execution.
        2. New fact indexing into FAISS vector database.
        3. Long-Term Memory similarity retrieval.
        4. Short-Term Memory context gathering.
        5. Dynamic Function / Tool calling.
        6. Answer generation with reasoning trace.
        7. Short-Term Memory update.
        """
        tools_executed: List[Dict[str, Any]] = []
        stored_facts: List[Dict[str, Any]] = []

        # Step 1: Check for feedback correction
        adaptation_record = self.adaptation_engine.process_adaptation(user_input, session_id)

        # Step 2: Extract & store new persistent facts if not a correction
        if not adaptation_record or not adaptation_record.get("adapted"):
            stored_facts = self._extract_and_index_facts(user_input, session_id)

        # Step 3: Retrieve relevant Long-Term Memories from FAISS
        retrieved_memories = self.ltm.retrieve_relevant_facts(
            query=user_input,
            top_k=3,
            min_similarity=0.10,
            filter_active_only=True
        )

        # Build context from memories
        memory_context_str = ""
        if retrieved_memories:
            memory_context_str = "\n".join([f"- [Recalled LTM Memory] {m['text']} (Similarity: {m['similarity_score']})" for m in retrieved_memories])

        # Step 4: Retrieve Short-Term conversational history
        stm_history_str = self.stm.get_formatted_history(session_id, limit=6)

        # Step 5: Determine and execute external tools
        planned_tools = self._determine_tools_needed(user_input, memory_context_str)
        for t_info in planned_tools:
            tool_name = t_info["tool"]
            tool_args = t_info["args"]
            tool_func = self.tool_map.get(tool_name)
            
            if tool_func:
                try:
                    result = tool_func.invoke(tool_args)
                    tools_executed.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": result,
                        "status": "success",
                        "description": t_info.get("description", "")
                    })
                except Exception as ex:
                    tools_executed.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": str(ex),
                        "status": "error",
                        "description": t_info.get("description", "")
                    })

        # Step 6: Generate response (using LLM if active, or heuristic generator)
        response_text = ""
        if self.llm:
            try:
                system_prompt = (
                    "You are CogniAssist, an autonomous personal research and coding assistant.\n"
                    "You have access to:\n"
                    f"1. Short-Term Session Context:\n{stm_history_str}\n\n"
                    f"2. Retrieved Long-Term Vector Memories:\n{memory_context_str or 'None'}\n\n"
                    f"3. Tool Execution Results:\n{tools_executed or 'None'}\n\n"
                    "Instructions:\n"
                    "- Answer the user's request accurately.\n"
                    "- If recalled long-term memories contain the answer, explicitly state that you recalled this from past sessions.\n"
                    "- If tools were executed, incorporate their exact results.\n"
                    "- If an adaptation/correction was made, confirm the memory update.\n"
                    "- Follow the user's preferred format if specified."
                )
                from langchain_core.messages import SystemMessage, HumanMessage
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_input)]
                ai_msg = self.llm.invoke(messages)
                response_text = ai_msg.content
            except Exception:
                response_text = ""

        if not response_text:
            # Deterministic, high-fidelity synthesis engine
            parts = []

            if adaptation_record and adaptation_record.get("adapted"):
                if adaptation_record.get("old_memory"):
                    parts.append(
                        f"**Memory Updated:** I have updated my long-term memory! "
                        f"Superseded previous fact: *\"{adaptation_record['old_memory']['text']}\"* -> "
                        f"New active fact: *\"{adaptation_record['new_memory']['text']}\"*."
                    )
                else:
                    parts.append(
                        f"**Memory Recorded:** I have noted your update in long-term memory: "
                        f"*\"{adaptation_record['new_memory']['text']}\"*."
                    )

            if tools_executed:
                tool_summaries = []
                for te in tools_executed:
                    tool_summaries.append(f"**Tool Invoked: `{te['tool']}`** ({te['description']})\n```\n{te['result']}\n```")
                parts.append("\n".join(tool_summaries))

            # Query answering based on retrieved memories or user input
            if retrieved_memories and ("what is" in user_input.lower() or "which" in user_input.lower() or "where" in user_input.lower() or "tell me" in user_input.lower()):
                recalled_facts_list = [f"- {m['text']}" for m in retrieved_memories]
                parts.append(
                    "**Recalled from Long-Term Memory (Across Sessions):**\n" +
                    "\n".join(recalled_facts_list)
                )
            elif not tools_executed and not adaptation_record:
                if stored_facts:
                    parts.append(f"Understood! I have saved this fact to my long-term vector memory: *\"{stored_facts[0]['text']}\"*.")
                else:
                    parts.append("I have received your message and updated the session context.")

            response_text = "\n\n".join(parts)

        # Step 7: Record conversation in Short-Term Memory
        self.stm.add_turn(session_id, "user", user_input)
        self.stm.add_turn(session_id, "assistant", response_text)

        turn_count = len(self.stm.get_history(session_id))

        return AgentResponse(
            answer=response_text,
            session_id=session_id,
            tools_called=tools_executed,
            retrieved_memories=retrieved_memories,
            stored_memories=stored_facts,
            adaptation_record=adaptation_record,
            short_term_turns_count=turn_count
        )
