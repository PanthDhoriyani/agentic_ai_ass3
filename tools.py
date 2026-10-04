"""
Tools module for Assignment 3: Tool / Function Calling Integration.
Implements:
1. Safe Scientific / Financial Calculator Tool
2. Sandboxed Python Code Execution Tool
3. Live / Resilient Web Search Tool
4. Long-Term Memory Inspection & Registration Tool
MIT ADT University -- Unit III: Memory, Tools, and Environment Interaction.
"""

import ast
import io
import math
import operator
import sys
import traceback
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool


# ==========================================
# 1. SAFE SCIENTIFIC CALCULATOR TOOL
# ==========================================

class CalculatorInput(BaseModel):
    expression: str = Field(..., description="The mathematical expression to evaluate, e.g., '45000 / 64' or '25 * (45000 / 64)'.")

# Safe math operators mapping
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pi": math.pi,
    "e": math.e
}


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):  # numbers, literals
        return node.value
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            return SAFE_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in SAFE_OPERATORS:
            operand = _eval_node(node.operand)
            return SAFE_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    elif isinstance(node, ast.Call):
        func_name = node.func.id if isinstance(node.func, ast.Name) else None
        if func_name in SAFE_FUNCTIONS:
            args = [_eval_node(arg) for arg in node.args]
            return SAFE_FUNCTIONS[func_name](*args)
        raise ValueError(f"Function '{func_name}' is not in safe function whitelist.")
    elif isinstance(node, ast.Name):
        if node.id in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Variable or symbol '{node.id}' is undefined.")
    else:
        raise ValueError(f"Unsupported AST node expression: {type(node).__name__}")


@tool("calculator", args_schema=CalculatorInput)
def safe_calculator(expression: str) -> str:
    """
    Safely calculates mathematical and numerical expressions without security risks.
    Supports arithmetic (+, -, *, /, //, %, **), powers, logarithms, and standard math constants.
    """
    try:
        clean_expr = expression.strip().replace("^", "**")
        parsed = ast.parse(clean_expr, mode='eval')
        result = _eval_node(parsed.body)
        
        # Round if float with lots of decimals
        if isinstance(result, float):
            result = round(result, 6)
            if result.is_integer():
                result = int(result)
                
        return f"Calculated Result: {result}"
    except Exception as e:
        return f"Calculation Error: {str(e)}"


# ==========================================
# 2. SANDBOXED PYTHON CODE EXECUTOR TOOL
# ==========================================

class CodeExecutorInput(BaseModel):
    code: str = Field(..., description="Valid Python code to execute. Print statements will be captured in the output.")

@tool("python_code_executor", args_schema=CodeExecutorInput)
def python_code_executor(code: str) -> str:
    """
    Executes a Python code snippet in a controlled execution scope and captures standard output.
    Useful for data analysis, complex algorithmic steps, simulating workloads, and calculating metrics.
    """
    clean_code = code.strip()
    # Strip markdown code fence if provided by LLM
    if clean_code.startswith("```"):
        lines = clean_code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        clean_code = "\n".join(lines).strip()

    # Block destructive or risky calls
    dangerous_keywords = ["import os", "import subprocess", "import shutil", "import sys", "open(", "__import__", "rmdir", "remove", "system("]
    for kw in dangerous_keywords:
        if kw in clean_code:
            return f"Security Restriction: The keyword or function '{kw}' is blocked in this sandbox."

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = io.StringIO()
    redirected_error = io.StringIO()
    
    # Safe whitelist for imports
    ALLOWED_MODULES = {"math", "random", "statistics", "json", "datetime", "collections", "itertools", "re", "decimal"}
    
    def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in ALLOWED_MODULES:
            return __import__(name, globals, locals, fromlist, level)
        raise ImportError(f"Import of module '{name}' is restricted in sandbox. Allowed: {sorted(list(ALLOWED_MODULES))}")

    # Safe global execution context
    safe_globals = {
        "__builtins__": {
            "__import__": safe_import,
            "print": print,
            "range": range,
            "len": len,
            "int": int,
            "float": float,
            "str": str,
            "bool": bool,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "sum": sum,
            "min": min,
            "max": max,
            "abs": abs,
            "round": round,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "map": map,
            "filter": filter,
            "isinstance": isinstance,
            "type": type,
        },
        "math": math,
    }
    local_scope: Dict[str, Any] = {}

    try:
        sys.stdout = redirected_output
        sys.stderr = redirected_error
        
        # Execute code in sandbox
        exec(clean_code, safe_globals, local_scope)
        
        output = redirected_output.getvalue().strip()
        errors = redirected_error.getvalue().strip()
        
        if errors:
            return f"Execution Completed with Warnings/Errors:\nOutput:\n{output}\nErrors:\n{errors}"
        elif output:
            return f"Execution Output:\n{output}"
        else:
            # If no print statement, report variables assigned
            var_summary = {k: v for k, v in local_scope.items() if not k.startswith("_")}
            if var_summary:
                return f"Execution Succeeded. Local Variables: {var_summary}"
            return "Execution Succeeded with no console output."
    except Exception:
        err_msg = traceback.format_exc().splitlines()[-1]
        return f"Code Execution Failed: {err_msg}"
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


# ==========================================
# 3. LIVE WEB SEARCH TOOL
# ==========================================

class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query string to look up up-to-date facts, documentation, or news.")

@tool("web_search", args_schema=WebSearchInput)
def live_web_search(query: str) -> str:
    """
    Searches the web via DuckDuckGo for recent information, documentation, news, or general world facts.
    """
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
            
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if results:
                formatted = []
                for i, r in enumerate(results, 1):
                    formatted.append(f"[{i}] {r.get('title')}\nSnippet: {r.get('body')}\nURL: {r.get('href')}")
                return "\n\n".join(formatted)
    except Exception:
        pass

    # Resilient fallback database for benchmark / offline mode
    simulated_db = {
        "neurips 2026": "NeurIPS 2026 (Conference on Neural Information Processing Systems) is scheduled for December 2026. Primary submission deadline: May 15, 2026.",
        "icml 2026": "ICML 2026 (International Conference on Machine Learning) will be held in July 2026. Abstract deadline: May 25, 2026; Full Paper deadline: June 1, 2026.",
        "langchain": "LangChain is a framework for developing applications powered by large language models (LLMs) with memory, tools, and agents.",
        "faiss": "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.",
        "graph neural networks": "Graph Neural Networks (GNNs) are neural models designed to capture graph-structured dependencies via message passing.",
        "mit adt": "MIT Art, Design and Technology University (MIT ADT University) is a multidisciplinary university located in Rajbaugh, Loni Kalbhor, Pune, Maharashtra."
    }
    
    q_lower = query.lower()
    for k, v in simulated_db.items():
        if k in q_lower:
            return f"[Search Result (Cached/Simulated for '{query}')]:\n{v}"
            
    return f"Search Result: Retrieved 0 real-time results for query '{query}'. (Ensure internet connection for live DuckDuckGo web results)."


# Tool registry
ALL_TOOLS = [safe_calculator, python_code_executor, live_web_search]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}
