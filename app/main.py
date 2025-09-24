from fastapi import FastAPI
from app.schemas import Query
from app.rag import generate
from app.vector_local import faiss_search

app = FastAPI(title="Local RAG API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(q: Query):
    answer = generate(q.question)
    return {"answer": answer}