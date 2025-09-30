import hashlib
import json
import math
import os
from typing import List, Tuple, Dict, Any

from .utils import ensure_dir

DEFAULT_RAG_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "rag_index")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in (text or "").split() if t.strip()]


def _tf(text: str) -> Dict[str, float]:
    toks = _tokenize(text)
    counts: Dict[str, int] = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    total = float(len(toks) or 1)
    return {k: v / total for k, v in counts.items()}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    common = set(a.keys()) & set(b.keys())
    num = sum(a[k] * b[k] for k in common)
    da = math.sqrt(sum(v * v for v in a.values()))
    db = math.sqrt(sum(v * v for v in b.values()))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


class LocalVectorStore:
    """
    Minimal local vector store using TF vectors and cosine similarity.
    Index stored on disk for persistence across runs.
    """

    def __init__(self, root: str = DEFAULT_RAG_ROOT):
        self.root = root
        ensure_dir(self.root)
        self.index_path = os.path.join(self.root, "index.jsonl")
        self._docs: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        self._docs = []
        if os.path.exists(self.index_path):
            with open(self.index_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self._docs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    def _persist(self, doc: Dict[str, Any]):
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # PUBLIC_INTERFACE
    def add_documents(self, docs: List[Dict[str, Any]]) -> None:
        """Add documents with structure: {'id': str, 'text': str, 'meta': {...}}"""
        for d in docs:
            text = d.get("text", "")
            vec = _tf(text)
            row = {"id": d.get("id") or hashlib.sha1(text.encode("utf-8")).hexdigest(), "text": text, "meta": d.get("meta", {}), "vec": vec}
            self._docs.append(row)
            self._persist(row)

    # PUBLIC_INTERFACE
    def similarity_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return top_k documents most similar to the query."""
        qv = _tf(query or "")
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for d in self._docs:
            score = _cosine(qv, d["vec"])
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"score": s, "id": d["id"], "text": d["text"], "meta": d.get("meta", {})} for s, d in scored[: max(1, top_k)]]
