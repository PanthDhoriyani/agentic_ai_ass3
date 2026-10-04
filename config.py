"""
Configuration module for Assignment 3: Retrieval-Augmented Agent with Long-Term Memory and Tool Calling.
MIT ADT University -- Unit III: Memory, Tools, and Environment Interaction.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent.resolve()
MEMORY_STORE_DIR = BASE_DIR / "memory_store"
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

for directory in [MEMORY_STORE_DIR, DATA_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys (Loaded from environment or Streamlit / script config)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# Default Model Configurations
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_EMBEDDING_MODEL = "models/embedding-001"

# Memory Parameters
SHORT_TERM_MEMORY_MAX_TURNS = 10
LONG_TERM_MEMORY_TOP_K = 3
SIMILARITY_SCORE_THRESHOLD = 0.35  # Threshold for considering memory relevant

# Evaluation Settings
BENCHMARK_LOG_FILE = LOGS_DIR / "test_logs.json"
EVALUATION_REPORT_FILE = BASE_DIR / "evaluation_results.md"
