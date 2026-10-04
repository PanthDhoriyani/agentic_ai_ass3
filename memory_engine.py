"""
Memory Engine Module for Unit III Assignment: Short-Term Memory, Long-Term FAISS Vector Database,
and Feedback-Based Adaptation Loop.
MIT ADT University -- Autonomous Agent Architecture.
"""

import json
import os
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

    class _NumpyFAISSIndexFlatIP:
        """NumPy-powered fallback for faiss.IndexFlatIP when faiss-cpu is not installed."""
        def __init__(self, dim: int):
            self.dim = dim
            self.vectors = np.empty((0, dim), dtype=np.float32)

        @property
        def ntotal(self) -> int:
            return self.vectors.shape[0]

        def add(self, x: np.ndarray) -> None:
            arr = np.array(x, dtype=np.float32)
            if self.vectors.shape[0] == 0:
                self.vectors = arr
            else:
                self.vectors = np.vstack([self.vectors, arr])

        def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
            if self.ntotal == 0:
                return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)
            sims = np.dot(self.vectors, q[0])
            k_search = min(k, self.ntotal)
            top_indices = np.argsort(-sims)[:k_search]
            top_scores = sims[top_indices]
            return np.array([top_scores], dtype=np.float32), np.array([top_indices], dtype=np.int64)

    class _FaissShim:
        IndexFlatIP = _NumpyFAISSIndexFlatIP

        @staticmethod
        def write_index(index: Any, filepath: str) -> None:
            if hasattr(index, "vectors"):
                np.save(filepath.replace(".bin", ".npy"), index.vectors)

        @staticmethod
        def read_index(filepath: str) -> Any:
            npy_path = filepath.replace(".bin", ".npy")
            if os.path.exists(npy_path):
                arr = np.load(npy_path)
                dim = arr.shape[1] if len(arr.shape) > 1 and arr.shape[1] > 0 else 256
                idx = _NumpyFAISSIndexFlatIP(dim)
                idx.vectors = arr
                return idx
            return _NumpyFAISSIndexFlatIP(256)

    faiss = _FaissShim()

from langchain_core.embeddings import Embeddings

from config import MEMORY_STORE_DIR, GOOGLE_API_KEY


# ==========================================
# 1. EMBEDDING ENGINE (HYBRID / RESILIENT)
# ==========================================

class LocalDenseEmbeddings(Embeddings):
    """
    Deterministic normalized dense semantic embedder for local/offline operation.
    Combines word stemming, character n-grams, and hash projections into a 256-dimensional
    normalized vector representation suitable for FAISS IndexFlatIP cosine similarity.
    """
    def __init__(self, dim: int = 256):
        self.dim = dim
        try:
            from nltk.stem import PorterStemmer
            self.stemmer = PorterStemmer()
        except Exception:
            self.stemmer = None

    def _embed_text(self, text: str) -> List[float]:
        clean_text = text.lower().strip()
        raw_tokens = re.findall(r"\w+", clean_text)
        vec = np.zeros(self.dim, dtype=np.float32)
        
        if not raw_tokens:
            return vec.tolist()
            
        stemmed_tokens = []
        for t in raw_tokens:
            stemmed = self.stemmer.stem(t) if self.stemmer else t
            stemmed_tokens.append(stemmed)
            
            # Direct word hash
            h = hash(stemmed) % self.dim
            vec[h] += 2.0
            
            # Character trigrams for morphological robustness
            for i in range(len(t) - 2):
                tri = t[i:i+3]
                h_tri = hash(tri) % self.dim
                vec[h_tri] += 0.5
                
        # Bigram projection
        for i in range(len(stemmed_tokens) - 1):
            bi = f"{stemmed_tokens[i]}_{stemmed_tokens[i+1]}"
            h_bi = hash(bi) % self.dim
            vec[h_bi] += 1.5

        # L2 Normalization
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


def get_embedding_provider(google_api_key: Optional[str] = None) -> Embeddings:
    """Returns Google Generative AI Embeddings if API key is provided, else fallback to LocalDenseEmbeddings."""
    key = google_api_key or GOOGLE_API_KEY
    if key and len(key.strip()) > 5:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=key
            )
        except Exception:
            pass
    return LocalDenseEmbeddings(dim=128)


# ==========================================
# 2. SHORT-TERM CONVERSATION BUFFER MEMORY
# ==========================================

class ShortTermMemory:
    """
    Manages short-term conversation context for multi-turn sessions.
    Maintains conversational turns per session with sliding window compaction.
    """
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        # session_id -> list of turn dicts
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}

    def add_turn(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Appends a dialogue turn to the session buffer."""
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        
        turn = {
            "turn_id": len(self.sessions[session_id]) + 1,
            "role": role,  # 'user', 'assistant', 'system', 'tool'
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.sessions[session_id].append(turn)

    def get_history(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns the conversation turns for a given session."""
        turns = self.sessions.get(session_id, [])
        turn_limit = limit or self.max_turns
        return turns[-turn_limit:]

    def get_formatted_history(self, session_id: str, limit: Optional[int] = None) -> str:
        """Returns formatted transcript string for prompt injection."""
        turns = self.get_history(session_id, limit)
        if not turns:
            return "No previous dialogue in this session."
        
        lines = []
        for t in turns:
            role_label = t["role"].capitalize()
            lines.append(f"{role_label}: {t['content']}")
        return "\n".join(lines)

    def clear_session(self, session_id: str) -> None:
        """Clears short-term buffer for a specific session."""
        if session_id in self.sessions:
            self.sessions[session_id] = []

    def list_sessions(self) -> List[str]:
        return list(self.sessions.keys())


# ==========================================
# 3. LONG-TERM VECTOR MEMORY (FAISS)
# ==========================================

class LongTermMemory:
    """
    Vector-Database backed Long-Term Memory using FAISS.
    Supports storing facts, semantic similarity retrieval, and metadata filtering.
    """
    def __init__(self, embedding_provider: Optional[Embeddings] = None, store_dir: Optional[str] = None):
        self.embeddings = embedding_provider or get_embedding_provider()
        self.store_dir = store_dir or str(MEMORY_STORE_DIR)
        os.makedirs(self.store_dir, exist_ok=True)
        
        # Test vector dimension
        test_vec = self.embeddings.embed_query("test dimension probe")
        self.dim = len(test_vec)
        
        # FAISS Index with Inner Product (Cosine similarity for L2-normalized vectors)
        self.index = faiss.IndexFlatIP(self.dim)
        
        # In-memory document & metadata store corresponding to vector index
        self.docs: List[Dict[str, Any]] = []
        
        # Load from disk if exists
        self.load()

    def store_fact(
        self,
        fact_text: str,
        session_id: str = "global",
        topic: str = "general",
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Embeds and stores a new fact in the FAISS vector database.
        Returns the generated memory ID.
        """
        fact_id = f"mem_{int(time.time() * 1000)}_{len(self.docs)}"
        vec = self.embeddings.embed_documents([fact_text])[0]
        vec_np = np.array([vec], dtype=np.float32)
        
        # Ensure normalization for cosine similarity
        norm = np.linalg.norm(vec_np)
        if norm > 0:
            vec_np = vec_np / norm
            
        self.index.add(vec_np)
        
        entry = {
            "id": fact_id,
            "text": fact_text,
            "session_id": session_id,
            "topic": topic,
            "status": "active",  # 'active' or 'superseded'
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "superseded_by": None,
            "metadata": metadata or {}
        }
        self.docs.append(entry)
        self.save()
        return fact_id

    def retrieve_relevant_facts(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.10,
        filter_active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Performs similarity search in FAISS vector database for relevant memories.
        Returns ranked list of matching memory dictionaries with similarity scores.
        """
        if self.index.ntotal == 0 or len(self.docs) == 0:
            return []

        query_vec = self.embeddings.embed_query(query)
        q_np = np.array([query_vec], dtype=np.float32)
        norm = np.linalg.norm(q_np)
        if norm > 0:
            q_np = q_np / norm

        # Search top candidate vectors
        k_search = min(top_k * 3, self.index.ntotal)
        scores, indices = self.index.search(q_np, k_search)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.docs):
                continue
            doc = self.docs[idx]
            
            if filter_active_only and doc.get("status") != "active":
                continue
                
            if score >= min_similarity:
                results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "session_id": doc["session_id"],
                    "topic": doc["topic"],
                    "status": doc["status"],
                    "similarity_score": round(float(score), 4),
                    "created_at": doc["created_at"],
                    "metadata": doc.get("metadata", {})
                })
            
            if len(results) >= top_k:
                break
                
        return results

    def update_or_supersede_fact(
        self,
        old_fact_id: str,
        new_fact_text: str,
        session_id: str,
        reason: str = "User feedback correction"
    ) -> Tuple[str, str]:
        """
        Updates long-term memory: marks old fact as 'superseded' and registers new active fact.
        Returns (old_fact_id, new_fact_id).
        """
        new_fact_id = self.store_fact(
            fact_text=new_fact_text,
            session_id=session_id,
            topic="updated_preference",
            metadata={"correction_reason": reason, "supersedes": old_fact_id}
        )
        
        # Mark old document as superseded
        for doc in self.docs:
            if doc["id"] == old_fact_id:
                doc["status"] = "superseded"
                doc["superseded_by"] = new_fact_id
                doc["updated_at"] = datetime.now().isoformat()
                doc["metadata"]["superseded_reason"] = reason
                break
                
        self.save()
        return old_fact_id, new_fact_id

    def get_all_memories(self, active_only: bool = False) -> List[Dict[str, Any]]:
        """Returns all registered long-term memories."""
        if active_only:
            return [d for d in self.docs if d.get("status") == "active"]
        return list(self.docs)

    def save(self) -> None:
        """Persists FAISS index and documents metadata to disk."""
        index_file = os.path.join(self.store_dir, "faiss_index.bin")
        docs_file = os.path.join(self.store_dir, "faiss_docs.json")
        try:
            faiss.write_index(self.index, index_file)
        except Exception:
            pass
        with open(docs_file, "w", encoding="utf-8") as f:
            json.dump(self.docs, f, indent=2)

    def load(self) -> None:
        """Loads FAISS index and documents from disk if available."""
        index_file = os.path.join(self.store_dir, "faiss_index.bin")
        docs_file = os.path.join(self.store_dir, "faiss_docs.json")
        if os.path.exists(index_file) and os.path.exists(docs_file):
            try:
                self.index = faiss.read_index(index_file)
                with open(docs_file, "r", encoding="utf-8") as f:
                    self.docs = json.load(f)
            except Exception:
                # Re-initialize clean if corrupted
                self.index = faiss.IndexFlatIP(self.dim)
                self.docs = []

    def clear_all(self) -> None:
        """Resets the vector database."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.docs = []
        self.save()


# ==========================================
# 4. FEEDBACK-BASED ADAPTATION ENGINE
# ==========================================

class FeedbackAdaptationEngine:
    """
    Detects user corrections/updates, resolves contradictions against stored LTM,
    and executes memory adaptation to maintain long-term consistency.
    """
    CORRECTION_TRIGGERS = [
        r"\bactually\b",
        r"\bcorrection\b",
        r"\bchange my\b",
        r"\bupdate my\b",
        r"\binstead of\b",
        r"\bthat was wrong\b",
        r"\bno, (?:my|i|it)\b",
        r"\bwrong(?:,| )",
        r"\bchange of plans\b",
        r"\bnot .+ but .+\b"
    ]

    def __init__(self, long_term_memory: LongTermMemory):
        self.ltm = long_term_memory
        self.adaptation_history: List[Dict[str, Any]] = []

    def detect_feedback(self, user_input: str) -> bool:
        """Checks if user input contains correction or update patterns."""
        lowered = user_input.lower()
        for pattern in self.CORRECTION_TRIGGERS:
            if re.search(pattern, lowered):
                return True
        return False

    def process_adaptation(
        self,
        user_input: str,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Executes feedback adaptation loop:
        1. Identifies what fact is being corrected.
        2. Retrieves the contradictory or outdated memory from FAISS.
        3. Marks old fact as superseded and indexes updated fact.
        4. Logs adaptation event.
        """
        if not self.detect_feedback(user_input):
            return None

        # Retrieve candidate memories semantically related to this correction
        candidates = self.ltm.retrieve_relevant_facts(
            query=user_input,
            top_k=2,
            min_similarity=0.20,
            filter_active_only=True
        )

        adaptation_record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "trigger_input": user_input,
            "adapted": False,
            "old_memory": None,
            "new_memory": None,
            "action": "none"
        }

        if candidates:
            target_mem = candidates[0]
            old_text = target_mem["text"]
            old_id = target_mem["id"]
            
            # Clean new fact representation from user prompt
            cleaned_fact = user_input.strip()
            # Iteratively strip conversational lead-ins and trailing instructions
            prefix_pattern = r"^(actually,?\s*|correction:?\s*|change of plans:?\s*|please update your memory\.?\s*|note that\s*)"
            while re.search(prefix_pattern, cleaned_fact, flags=re.I):
                cleaned_fact = re.sub(prefix_pattern, "", cleaned_fact, count=1, flags=re.I).strip()
            # Also strip trailing please update memory
            cleaned_fact = re.sub(r",?\s*please update (?:your )?memory\.?$", "", cleaned_fact, flags=re.I).strip()
            
            old_id, new_id = self.ltm.update_or_supersede_fact(
                old_fact_id=old_id,
                new_fact_text=cleaned_fact,
                session_id=session_id,
                reason=f"User correction: {user_input}"
            )
            
            adaptation_record.update({
                "adapted": True,
                "old_memory": {"id": old_id, "text": old_text},
                "new_memory": {"id": new_id, "text": cleaned_fact},
                "action": "superseded_and_updated",
                "similarity_score": target_mem["similarity_score"]
            })
        else:
            # If no direct match to supersede, store as new active user memory
            new_id = self.ltm.store_fact(
                fact_text=user_input,
                session_id=session_id,
                topic="user_feedback",
                metadata={"type": "feedback_addition"}
            )
            adaptation_record.update({
                "adapted": True,
                "old_memory": None,
                "new_memory": {"id": new_id, "text": user_input},
                "action": "new_memory_stored",
                "similarity_score": 1.0
            })

        self.adaptation_history.append(adaptation_record)
        return adaptation_record
