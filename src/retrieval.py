"""Retrieval: index dense (sentence-transformers), BM25, hybride (RRF), reranking.

Le calcul d'embeddings sur 22 633 articles avec un modèle large (e5-large) est
coûteux — prévu pour tourner sur GPU (Kaggle T4), cf. kaggle_kernels/retrieval_job.py.
"""
from __future__ import annotations

import numpy as np


class DenseIndex:
    def __init__(self, model_name: str, device: str | None = None, prefix_query: str = "", prefix_doc: str = ""):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        # multilingual-e5-large attend des préfixes "query: " / "passage: "
        self.prefix_query = prefix_query
        self.prefix_doc = prefix_doc
        self.doc_ids: list[int] = []
        self.embeddings: np.ndarray | None = None

    def build(self, doc_ids: list[int], texts: list[str], batch_size: int = 64, show_progress: bool = True) -> None:
        self.doc_ids = doc_ids
        prefixed = [f"{self.prefix_doc}{t}" for t in texts]
        self.embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_emb = self.model.encode(
            f"{self.prefix_query}{query}", normalize_embeddings=True, convert_to_numpy=True
        )
        scores = self.embeddings @ q_emb
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]

    def search_batch(self, queries: list[str], top_k: int = 10) -> list[list[tuple[int, float]]]:
        q_embs = self.model.encode(
            [f"{self.prefix_query}{q}" for q in queries],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
        all_scores = q_embs @ self.embeddings.T
        results = []
        for scores in all_scores:
            top_idx = np.argsort(-scores)[:top_k]
            results.append([(self.doc_ids[i], float(scores[i])) for i in top_idx])
        return results

    def save(self, path: str) -> None:
        np.savez(path, doc_ids=np.array(self.doc_ids), embeddings=self.embeddings)

    def load(self, path: str) -> None:
        data = np.load(path)
        self.doc_ids = data["doc_ids"].tolist()
        self.embeddings = data["embeddings"]


class BM25Index:
    def __init__(self):
        self.doc_ids: list[int] = []
        self.bm25 = None
        self._tokenized = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def build(self, doc_ids: list[int], texts: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        self.doc_ids = doc_ids
        self._tokenized = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(self._tokenize(query))
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_idx]


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[int, float]]], k_constant: int = 60, top_k: int = 10
) -> list[tuple[int, float]]:
    """Fusion RRF de plusieurs listes de résultats (dense + BM25)."""
    fused: dict[int, float] = {}
    for results in result_lists:
        for rank, (doc_id, _score) in enumerate(results, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k_constant + rank)
    ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
    return ranked


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str | None = None):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, candidates: list[tuple[int, str]], top_k: int = 10) -> list[tuple[int, float]]:
        pairs = [(query, text) for _doc_id, text in candidates]
        scores = self.model.predict(pairs)
        order = np.argsort(-scores)[:top_k]
        return [(candidates[i][0], float(scores[i])) for i in order]


def chunk_text(text: str, chunk_size_tokens: int = 256, overlap: int = 32) -> list[str]:
    """Découpage naïf par mots (approx. tokens) pour l'ablation article-entier vs chunks."""
    words = text.split()
    if len(words) <= chunk_size_tokens:
        return [text]
    chunks = []
    step = chunk_size_tokens - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size_tokens])
        chunks.append(chunk)
        if start + chunk_size_tokens >= len(words):
            break
    return chunks
