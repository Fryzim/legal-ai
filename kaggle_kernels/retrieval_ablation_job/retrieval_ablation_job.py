"""Kaggle kernel (GPU T4): ablation retrieval — vise à expliquer l'écart constaté
entre les Recall@k reproduits (job ter-bsard-retrieval-eval) et ceux des CR
précédents, et à répondre à la demande d'A. Habrard d'étendre le protocole
expérimental au-delà d'une seule configuration.

Variantes testées, toutes avec multilingual-e5-large (meilleur modèle du job
précédent) sur les mêmes 222 questions test :
  1. baseline    : texte de l'article seul (= reproduction du job précédent, pour contrôle)
  2. enrichi      : reference + hiérarchie (code/book/part/chapter/section) + article
  3. chunké 256   : article découpé en chunks de ~256 mots, score = max des chunks
  4. reranker     : baseline + reranking cross-encoder BAAI/bge-reranker-v2-m3 sur top-20
  5. k ablation   : Recall@k déjà couvert par 1-4 (k in [1,3,5,10,20])

Sortie: /kaggle/working/retrieval_ablation_results.json
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
import torch

# Chemins portables: par defaut Kaggle, surchargeables via variables
# d'environnement pour tourner ailleurs (RunPod, etc.) sans modifier le script.
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")
INPUT_DIR = os.environ.get("INPUT_DIR", "/kaggle/input")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

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
assert len(articles) == 22633
assert len(questions_test) == 222

doc_ids = articles["id"].tolist()
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
    return out


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

from sentence_transformers import SentenceTransformer

results = {"seed": SEED, "n_test_questions": 222, "corpus_size": len(articles), "variants": {}}


def save_partial():
    """Sauvegarde incrémentale: un crash sur une variante ne doit pas faire perdre
    les résultats déjà obtenus (cf. DIFFICULTES.md -- vécu lors du premier run)."""
    with open(f"{WORK_DIR}/retrieval_ablation_results.json", "w") as f:
        json.dump(results, f, indent=2)


retriever = SentenceTransformer("intfloat/multilingual-e5-large", device=device)
q_emb = retriever.encode(
    [f"query: {q}" for q in queries], batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
)

# --- Variant 1: baseline (article text only) ---
print("=== variant: baseline ===")
t0 = time.time()
texts_baseline = articles["article"].fillna("").tolist()
doc_emb = retriever.encode(
    [f"passage: {t}" for t in texts_baseline], batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
)
sims = q_emb @ doc_emb.T
retrieved = [[doc_ids[i] for i in np.argsort(-row)[:20]] for row in sims]
results["variants"]["baseline"] = evaluate(retrieved, relevant)
results["variants"]["baseline"]["duration_seconds"] = time.time() - t0
print(results["variants"]["baseline"])
save_partial()

# --- Variant 2: enriched (hierarchy metadata + article text) ---
print("=== variant: enrichi (métadonnées hiérarchiques) ===")
t0 = time.time()


def enrich(row):
    parts = [
        str(row.get("code") or ""),
        str(row.get("book") or ""),
        str(row.get("part") or ""),
        str(row.get("act") or ""),
        str(row.get("chapter") or ""),
        str(row.get("section") or ""),
        str(row.get("subsection") or ""),
        str(row.get("description") or ""),
        str(row.get("article") or ""),
    ]
    return " ".join(p for p in parts if p and p != "nan")


texts_enriched = articles.apply(enrich, axis=1).tolist()
doc_emb_enriched = retriever.encode(
    [f"passage: {t}" for t in texts_enriched], batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
)
sims = q_emb @ doc_emb_enriched.T
retrieved = [[doc_ids[i] for i in np.argsort(-row)[:20]] for row in sims]
results["variants"]["enriched_metadata"] = evaluate(retrieved, relevant)
results["variants"]["enriched_metadata"]["duration_seconds"] = time.time() - t0
print(results["variants"]["enriched_metadata"])
save_partial()

# --- Variant 3: chunked (256-word chunks, max-pool score to article level) ---
print("=== variant: chunké 256 mots ===")
t0 = time.time()


def chunk_text(text, chunk_size=256, overlap=32):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= len(words):
            break
    return chunks


chunk_doc_ids = []
chunk_texts = []
for aid, text in zip(doc_ids, texts_baseline):
    for c in chunk_text(text):
        chunk_doc_ids.append(aid)
        chunk_texts.append(c)

print(f"n_chunks = {len(chunk_texts)} (from {len(doc_ids)} articles)")
chunk_emb = retriever.encode(
    [f"passage: {c}" for c in chunk_texts], batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
)
sims_chunks = q_emb @ chunk_emb.T  # (n_queries, n_chunks)

retrieved_chunked = []
chunk_doc_ids_arr = np.array(chunk_doc_ids)
for row in sims_chunks:
    order = np.argsort(-row)
    seen = {}
    for idx in order:
        aid = chunk_doc_ids_arr[idx]
        if aid not in seen:
            seen[aid] = row[idx]
        if len(seen) >= 20:
            break
    ranked_aids = [aid for aid, _ in sorted(seen.items(), key=lambda kv: -kv[1])][:20]
    retrieved_chunked.append(ranked_aids)

results["variants"]["chunked_256"] = evaluate(retrieved_chunked, relevant)
results["variants"]["chunked_256"]["duration_seconds"] = time.time() - t0
results["variants"]["chunked_256"]["n_chunks"] = len(chunk_texts)
print(results["variants"]["chunked_256"])
save_partial()

del retriever, doc_emb_enriched, chunk_emb, sims_chunks
torch.cuda.empty_cache()

# --- Variant 4: baseline + cross-encoder reranker on top-20 ---
# Fix OOM rencontré au premier run: articles jusqu'à 5790 mots -> troncature
# explicite (max_length) + petit batch_size pour le reranker, cf. DIFFICULTES.md.
print("=== variant: baseline + reranker (BAAI/bge-reranker-v2-m3) ===")
t0 = time.time()
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=device, max_length=512)

sims_baseline = q_emb @ doc_emb.T
retrieved_reranked = []
for qi, row in enumerate(sims_baseline):
    top20_idx = np.argsort(-row)[:20]
    candidates = [(doc_ids[i], texts_baseline[i][:2000]) for i in top20_idx]  # troncature texte brute en plus de max_length tokenizer
    pairs = [(queries[qi], text) for _aid, text in candidates]
    scores = reranker.predict(pairs, batch_size=8, show_progress_bar=False)
    order = np.argsort(-scores)
    retrieved_reranked.append([candidates[i][0] for i in order])
    if qi % 50 == 0:
        torch.cuda.empty_cache()
        print(f"  reranked {qi}/{len(sims_baseline)}")

results["variants"]["baseline_reranked"] = evaluate(retrieved_reranked, relevant)
results["variants"]["baseline_reranked"]["duration_seconds"] = time.time() - t0
print(results["variants"]["baseline_reranked"])

results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save_partial()

print("DONE")
print(json.dumps(results, indent=2))
