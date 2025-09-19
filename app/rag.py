from typing import List, Dict
from app.vector_local import faiss_search
from app.llm import llm_answer

SYSTEM_PROMPT = (
    "You are a concise assistant. Use the provided CONTEXT. "
    "If uncertain or context is insufficient, say you don't know."
)

def _format_context(passages: List[Dict]) -> str:
    return "\n\n".join(f"[{p['title']}]({p['chunk_id']})\n{p['content']}" for p in passages)

def generate(query: str) -> str:
    passages = faiss_search(query, k=5)
    context = _format_context(passages)
    user = f"Question: {query}\n\nCONTEXT:\n{context}"
    # Try LLM; if not available, return context-only answer
    ans = llm_answer(SYSTEM_PROMPT, user)
    if ans:
        return ans.strip()
    # Context-only fallback (no key required)
    if passages:
        top = passages[0]
        return (
            "Based on the retrieved context:\n"
            f"{top['content']}\n\n"
            "Tip: set OPENAI_API_KEY to get full LLM answers."
        )
    return "I don't have enough information to answer."
