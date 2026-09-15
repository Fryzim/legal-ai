"""Kaggle kernel (GPU T4): ablation du nombre de fragments RAG injectés
(k in {1, 3, 10} ; k=5 déjà couvert par le run principal C2). Derby LLM
signale que 10 fragments saturent inutilement le prompt -- on le vérifie
empiriquement sur le domaine juridique belge.

Limité au même sous-échantillon de 50 questions que le LLM-judge (même seed)
pour borner le coût GPU -- cohérent avec le brief qui réserve les analyses
coûteuses au sous-ensemble de 50 questions.

Dépend du kernel ter-bsard-retrieval-eval (index e5-large).
Sortie: /kaggle/working/k_ablation_results.json
"""
import glob
import json
import re
import subprocess
import os
import sys
import time
import urllib.request
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes", "accelerate", "rouge_score", "bert_score", "sentence-transformers"],
    check=True,
)

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Chemins portables: par defaut Kaggle, surchargeables via variables
# d'environnement pour tourner ailleurs (RunPod, etc.) sans modifier le script.
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")
INPUT_DIR = os.environ.get("INPUT_DIR", "/kaggle/input")

SEED = 42
N_SAMPLES = 50
K_VALUES = [1, 3, 10]
BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
MAX_NEW_TOKENS = 450  # releve du 30/07: 300 tronquait la reponse avant la citation finale
# batch_size par k: un contexte de k=10 articles peut faire exploser la mémoire
# attention sur une T4 16 Go avec un batch de 8 (cf. OOM rencontré: "Tried to
# allocate 4.00 GiB" a k=10 avec batch_size=8 fixe). Reduit avec k croissant.
BATCH_SIZE_BY_K = {1: 8, 3: 8, 10: 4}
DEFAULT_BATCH_SIZE = 8

SYSTEM_PROMPT_FR = (
    "Tu es un assistant juridique spécialisé en droit belge francophone. "
    "Réponds à la question posée en français, de manière claire et rédigée. "
    "Si un contexte juridique t'est fourni, appuie-toi dessus. "
    "Termine impérativement ta réponse par une citation précise de l'article "
    "sur lequel tu t'appuies, au format : \"Article <numéro>\". "
    "Si tu ne connais pas la réponse ou si le contexte ne permet pas de répondre "
    "avec certitude, réponds explicitement \"Je ne sais pas\" plutôt que d'inventer "
    "une réponse ou une référence."
)
ARTICLE_REFERENCE_REGEX = r"Art\.\s*([^,]+)"
ARTICLE_CITATION_REGEX = r"[Aa]rticle\s+([^\s,.;]+)"
# ~2.5% du corpus (555/22633) porte un suffixe d'annotation collé à l'ID
# ("275_DROIT_FUTUR", "1714bis_REGION_DE_BRUXELLES-CAPITALE") -- artefact du
# scraping de la source, pas une partie citable. 16/222 questions test
# concernées: sans ce nettoyage la citation gold est incitable dans le format
# demandé au modèle, ce qui pénalise injustement precision/recall/hallucination.
ARTICLE_ID_ANNOTATION_SUFFIX_REGEX = r"_[A-Z][A-Z_\-]+$"

np.random.seed(SEED)
torch.manual_seed(SEED)


def find_file(pattern):
    matches = glob.glob(f"{INPUT_DIR}/**/{pattern}", recursive=True)
    if not matches:
        raise FileNotFoundError(f"No match for {pattern} under {INPUT_DIR}")
    return matches[0]


index_path = find_file("index_e5_large.npz")

DATA_DIR = Path(f"{WORK_DIR}/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
HF_BASE = "https://huggingface.co/datasets/maastrichtlawtech/bsard/resolve/main"
for fname in ["articles.csv", "questions_test.csv"]:
    dest = DATA_DIR / fname
    if not dest.exists():
        urllib.request.urlretrieve(f"{HF_BASE}/{fname}", dest)

articles = pd.read_csv(DATA_DIR / "articles.csv")
article_lookup = articles.set_index("id").to_dict(orient="index")


def extract_reference_id(ref_string):
    m = re.search(ARTICLE_REFERENCE_REGEX, str(ref_string))
    if not m:
        return ""
    return re.sub(ARTICLE_ID_ANNOTATION_SUFFIX_REGEX, "", m.group(1).strip())


ref_ids = articles["reference"].apply(extract_reference_id)
assert (ref_ids != "").mean() > 0.99
valid_article_refs = set(ref_ids.tolist())
valid_article_refs.discard("")

questions_test = pd.read_csv(DATA_DIR / "questions_test.csv")
questions_test["article_ids"] = questions_test["article_ids"].apply(
    lambda x: [int(i) for i in str(x).split(",") if i.strip()]
)
assert len(questions_test) == 222

rng = np.random.default_rng(SEED)
sample_idx = sorted(rng.choice(len(questions_test), size=N_SAMPLES, replace=False).tolist())
sample_df = questions_test.iloc[sample_idx].reset_index(drop=True)
print(f"n_samples = {len(sample_df)}")

reference_answers = []
gold_refs = []
for _, row in sample_df.iterrows():
    parts, refs = [], []
    for aid in row["article_ids"]:
        art = article_lookup.get(aid)
        if art is not None:
            parts.append(f"{art['reference']}: {art['article']}")
            rid = extract_reference_id(art["reference"])
            if rid:
                refs.append(rid)
    reference_answers.append("\n".join(parts))
    gold_refs.append(refs)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

from sentence_transformers import SentenceTransformer

retriever = SentenceTransformer("intfloat/multilingual-e5-large", device=device)
idx = np.load(index_path)
doc_ids = idx["doc_ids"].tolist()
doc_embeddings = idx["embeddings"]

queries = sample_df["question"].tolist()
q_emb = retriever.encode([f"query: {q}" for q in queries], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
sims = q_emb @ doc_embeddings.T

del retriever
torch.cuda.empty_cache()

# top-20 candidates per question, on prend les k premiers selon la valeur testée
top20_per_question = []
for row in sims:
    top_idx = np.argsort(-row)[:20]
    top20_per_question.append([doc_ids[i] for i in top_idx if doc_ids[i] in article_lookup])


def build_messages(question, context_ids, k):
    ctx_ids = context_ids[:k]
    ctx_articles = [article_lookup[aid] for aid in ctx_ids]
    user_parts = []
    if ctx_articles:
        ctx = "\n".join(f"- {a['reference']} : {a['article']}" for a in ctx_articles)
        user_parts.append(f"Contexte juridique :\n{ctx}\n")
    user_parts.append(f"Question : {question}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT_FR},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def _generate_one_batch(model, tokenizer, batch, max_new_tokens):
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    decoded = []
    for j in range(len(batch)):
        new_tokens = gen[j][enc["input_ids"].shape[1] :]
        decoded.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    del enc, gen
    return decoded


def batched_generate(model, tokenizer, prompts_messages, batch_size=DEFAULT_BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    outputs = []
    i = 0
    current_bs = batch_size
    while i < len(prompts_messages):
        batch = prompts_messages[i : i + current_bs]
        try:
            outputs.extend(_generate_one_batch(model, tokenizer, batch, max_new_tokens))
            i += len(batch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_bs == 1:
                raise
            current_bs = max(1, current_bs // 2)
            print(f"  OOM, reduction du batch_size a {current_bs} et nouvel essai")
    return outputs


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")

from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

results_path = Path(f"{WORK_DIR}/k_ablation_results.json")
results = {"seed": SEED, "n_samples": N_SAMPLES, "k_values": K_VALUES, "runs": {}}


def save_partial():
    # Sauvegarde après CHAQUE k, pas seulement a la fin: la premiere execution
    # a perdu k=1 et k=3 (deja termines avec succes) a cause d'un OOM sur k=10
    # qui a empeche le json.dump final de s'executer (meme lecon que le
    # reranker OOM documente dans DIFFICULTES.md Sec.12, pas reappliquee ici
    # la premiere fois).
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


for k in K_VALUES:
    print(f"=== k={k} ===")
    t0 = time.time()
    msgs = [build_messages(q, ctx_ids, k) for q, ctx_ids in zip(queries, top20_per_question)]
    answers = batched_generate(model, tokenizer, msgs, batch_size=BATCH_SIZE_BY_K.get(k, DEFAULT_BATCH_SIZE))
    torch.cuda.empty_cache()

    rouge_scores = [rouge.score(ref, ans)["rougeL"].fmeasure for ref, ans in zip(reference_answers, answers)]
    _, _, bert_f1 = bert_score_fn(answers, reference_answers, model_type="distilbert-base-multilingual-cased", lang="fr", verbose=False)

    citation_precisions, citation_recalls, citation_exact, hallu_flags = [], [], [], []
    for ans, refs in zip(answers, gold_refs):
        predicted = set(re.findall(ARTICLE_CITATION_REGEX, ans))
        gold = set(refs)
        if predicted or gold:
            tp = len(predicted & gold)
            citation_precisions.append(tp / len(predicted) if predicted else 0.0)
            citation_recalls.append(tp / len(gold) if gold else float("nan"))
            citation_exact.append(float(predicted == gold))
        hallu_flags.append(any(c not in valid_article_refs for c in predicted))

    results["runs"][f"k={k}"] = {
        "answers": answers,
        "duration_seconds": time.time() - t0,
        "rouge_l_f1_mean": float(np.mean(rouge_scores)),
        "bertscore_f1_mean": float(np.mean(bert_f1.tolist())),
        "citation_precision_mean": float(np.nanmean(citation_precisions)) if citation_precisions else None,
        "citation_recall_mean": float(np.nanmean(citation_recalls)) if citation_recalls else None,
        "citation_exact_match_rate": float(np.mean(citation_exact)) if citation_exact else None,
        "hallucination_rate": float(np.mean(hallu_flags)),
        "avg_context_chars": float(np.mean([sum(len(article_lookup[a]["article"]) for a in ids[:k]) for ids in top20_per_question])),
    }
    print(results["runs"][f"k={k}"])
    save_partial()

results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
save_partial()

print("DONE")
