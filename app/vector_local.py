import os, glob, uuid
from typing import List, Dict, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def _read_docs(data_dir: str = "data") -> List[Tuple[str, str]]:
    paths = [p for p in glob.glob(os.path.join(data_dir, "**/*"), recursive=True) if os.path.isfile(p)]
    docs = []
    for p in paths:
        with open(p, "r", errors="ignore") as f:
            docs.append((os.path.basename(p), f.read()))
    return docs

def _chunk(text: str, max_chars: int = 400) -> List[str]:
    chunks = []
    t = text.strip().replace("\n", " ")
    for i in range(0, len(t), max_chars):
        chunks.append(t[i:i+max_chars])
    return chunks or [""]

class LocalVectorStore:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.meta: List[Dict] = []
        self.index = None

    def _embed(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return emb.astype("float32")

    def build(self, data_dir: str = "data"):
        docs = _read_docs(data_dir)
        all_chunks, metadata = [], []
        for title, text in docs:
            chunks = _chunk(text)
            for i, ch in enumerate(chunks):
                all_chunks.append(ch)
                metadata.append({"title": title, "chunk_id": f"{title}-{i}", "content": ch})
        if not all_chunks:
            all_chunks = ["(empty corpus)"]
            metadata = [{"title": "empty", "chunk_id": "0", "content": "(empty)"}]

        embs = self._embed(all_chunks)
        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)
        self.meta = metadata

    def search(self, query: str, k: int = 5) -> List[Dict]:
        if self.index is None:
            self.build()
        q = self._embed([query])
        _, ids = self.index.search(q, k)
        out = []
        for idx in ids[0]:
            if idx == -1:
                continue
            out.append(self.meta[idx])
        return out

_store = None

def faiss_search(query: str, k: int = 5) -> List[Dict]:
    global _store
    if _store is None:
        _store = LocalVectorStore()
        _store.build()
    return _store.search(query, k=k)
