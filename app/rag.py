from typing import List, Dict
from app.vector_local import faiss_search
from app.llm import llm_answer
import numpy as np
from sentence_transformers import SentenceTransformer

_embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

SYSTEM_PROMPT = (
    "You are a concise assistant. Use the provided CONTEXT. "
    "If uncertain or context is insufficient, say you don't know."
)

SIM_THRESHOLD = 0.30  # tune 0.25–0.4

def _retrieval_ok(query: str, passages: list[dict]) -> bool:
    if not passages:
        return False
    qv = _embed.encode([query], normalize_embeddings=True)
    pv = _embed.encode([p["content"] for p in passages[:3]], normalize_embeddings=True)
    sims = (pv @ qv.T).ravel()
    return float(np.max(sims)) >= SIM_THRESHOLD


def _format_context(passages: List[Dict]) -> str:
    return "\n\n".join(f"[{p['title']}]({p['chunk_id']})\n{p['content']}" for p in passages)

def generate(query: str) -> str:
    passages = faiss_search(query, k=5)
    if not _retrieval_ok(query, passages):
        return "I don't know based on the provided context."
    context = _format_context(passages)
    user = f"Question: {query}\n\nCONTEXT:\n{context}"
    # Try LLM; if not available, return context-only answerr
    ans = llm_answer(SYSTEM_PROMPT, user)
    if ans:
        return ans.strip()
    # Context-only fallback (no key required)
    if passages:
        top = passages[0]
        return (
            "Based on the retrieved context:\n"
            f"{top['content']}\n\n"
        )
    return "I don't have enough information to answer."
