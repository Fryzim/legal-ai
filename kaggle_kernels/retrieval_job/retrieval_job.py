"""Kaggle kernel (GPU T4): index BSARD (mpnet, e5-large, BM25) et évalue
Recall@k / MRR@10 / nDCG@10 sur les 222 questions test réelles.

Script autonome (Kaggle isole l'environnement, pas d'accès à src/ du repo local).
Sortie: /kaggle/working/retrieval_results.json (récupéré ensuite via `kaggle kernels output`).
"""
import json
import subprocess
import os
import sys
import time
import urllib.request
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rank_bm25", "sentence-transformers"], check=True)

import numpy as np
import pandas as pd

# Chemins portables: par defaut Kaggle, surchargeables via variables
# d'environnement pour tourner ailleurs (RunPod, etc.) sans modifier le script.
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")
INPUT_DIR = os.environ.get("INPUT_DIR", "/kaggle/input")

SEED = 42
np.random.seed(SEED)

DATA_DIR = Path(f"{WORK_DIR}/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
HF_BASE = "https://huggingface.co/datasets/maastrichtlawtech/bsard/resolve/main"

for fname in ["articles.csv", "questions_test.csv"]:
    dest = DATA_DIR / fname
    if not dest.exists():
        urllib.request.urlretrieve(f"{HF_BASE}/{fname}", dest)

articles = pd.read_csv(DATA_DIR / "articles.csv")
questions_test = pd.read_csv(DATA_DIR / "questions_test.csv")
questions_test["article_ids"] = questions_test["article_ids"].apply(
    lambda x: [int(i) for i in str(x).split(",") if i.strip()]
)

assert len(articles) == 22633, f"corpus size mismatch: {len(articles)}"
assert len(questions_test) == 222, f"test size mismatch: {len(questions_test)}"

doc_ids = articles["id"].tolist()
doc_texts = articles["article"].fillna("").tolist()
queries = questions_test["question"].tolist()
relevant = questions_test["article_ids"].tolist()

RECALL_KS = [1, 3, 5, 10, 20]


def recall_at_k(retrieved, rel, k):
    if not rel:
        return np.nan
    topk = set(retrieved[:k])
    return len(topk & set(rel)) / len(rel)


def mrr_at_k(retrieved, rel, k=10):
    rel_set = set(rel)
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in rel_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved, rel, k=10):
    rel_set = set(rel)
    dcg = sum(1.0 / np.log2(r + 1) for r, d in enumerate(retrieved[:k], start=1) if d in rel_set)
    ideal = min(len(rel_set), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate(all_retrieved, all_relevant):
    out = {}
    for k in RECALL_KS:
        vals = [recall_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)]
        out[f"recall@{k}"] = float(np.nanmean(vals))
    out["mrr@10"] = float(np.mean([mrr_at_k(r, rel) for r, rel in zip(all_retrieved, all_relevant)]))
    out["ndcg@10"] = float(np.mean([ndcg_at_k(r, rel) for r, rel in zip(all_retrieved, all_relevant)]))
    out["n_questions"] = len(all_retrieved)
    return out


results = {
    "seed": SEED,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "corpus_size": len(articles),
    "n_test_questions": len(questions_test),
    "runs": {},
}

# --- BM25 (CPU) ---
print("=== BM25 ===")
t0 = time.time()
from rank_bm25 import BM25Okapi

tokenized_corpus = [t.lower().split() for t in doc_texts]
bm25 = BM25Okapi(tokenized_corpus)
bm25_retrieved = []
for q in queries:
    scores = bm25.get_scores(q.lower().split())
    top_idx = np.argsort(-scores)[:20]
    bm25_retrieved.append([doc_ids[i] for i in top_idx])
results["runs"]["bm25"] = evaluate(bm25_retrieved, relevant)
results["runs"]["bm25"]["duration_seconds"] = time.time() - t0
print(json.dumps(results["runs"]["bm25"], indent=2))

# --- Dense embeddings (GPU) ---
from sentence_transformers import SentenceTransformer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

dense_models = {
    "mpnet": {"name": "sentence-transformers/all-mpnet-base-v2", "prefix_q": "", "prefix_d": ""},
    "e5_large": {
        "name": "intfloat/multilingual-e5-large",
        "prefix_q": "query: ",
        "prefix_d": "passage: ",
    },
}

dense_retrieved_by_model = {}
for key, cfg in dense_models.items():
    print(f"=== dense: {key} ===")
    t0 = time.time()
    model = SentenceTransformer(cfg["name"], device=device)
    doc_emb = model.encode(
        [f"{cfg['prefix_d']}{t}" for t in doc_texts],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    q_emb = model.encode(
        [f"{cfg['prefix_q']}{q}" for q in queries],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    sims = q_emb @ doc_emb.T
    retrieved = []
    for row in sims:
        top_idx = np.argsort(-row)[:20]
        retrieved.append([doc_ids[i] for i in top_idx])
    dense_retrieved_by_model[key] = retrieved
    results["runs"][key] = evaluate(retrieved, relevant)
    results["runs"][key]["duration_seconds"] = time.time() - t0
    print(json.dumps(results["runs"][key], indent=2))
    np.savez(
        f"{WORK_DIR}/index_{key}.npz",
        doc_ids=np.array(doc_ids),
        embeddings=doc_emb,
    )
    del model
    torch.cuda.empty_cache()

# --- Hybride RRF (BM25 + e5-large) ---
print("=== hybrid RRF (bm25 + e5_large) ===")
t0 = time.time()


def rrf_fuse(list_a, list_b, k_const=60, top_k=20):
    fused = {}
    for lst in (list_a, list_b):
        for rank, doc_id in enumerate(lst, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k_const + rank)
    return [d for d, _ in sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]]


hybrid_retrieved = [
    rrf_fuse(bm25_retrieved[i], dense_retrieved_by_model["e5_large"][i]) for i in range(len(queries))
]
results["runs"]["hybrid_bm25_e5"] = evaluate(hybrid_retrieved, relevant)
results["runs"]["hybrid_bm25_e5"]["duration_seconds"] = time.time() - t0
print(json.dumps(results["runs"]["hybrid_bm25_e5"], indent=2))

with open(f"{WORK_DIR}/retrieval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("DONE")
print(json.dumps(results, indent=2))
