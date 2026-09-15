"""Kaggle kernel (GPU T4): comparaison avec un autre LLM open-source existant
(Qwen2.5-7B-Instruct, gratuit, non gated, vérifié le 2026-07-28), zero-shot et
RAG, sur les mêmes 222 questions test et le même prompt que Mistral -- réponse
directe à la demande explicite de comparer avec d'autres modèles LLM
existants, pas seulement des variantes de Mistral.

Ne teste PAS de fine-tuning de Qwen (hors budget de cette phase) -- seulement
zero-shot ("Qwen-C1") et RAG ("Qwen-C2"), comparables à Mistral C1/C2. Un
fine-tuning de Qwen resterait une extension P2 (généralisation des
conclusions à un second modèle de base) si le temps le permet.

Dépend du kernel ter-bsard-retrieval-eval (index e5-large).
Sortie: /kaggle/working/other_llm_results.json
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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Chemins portables: par defaut Kaggle, surchargeables via variables
# d'environnement pour tourner ailleurs (RunPod, etc.) sans modifier le script.
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")
INPUT_DIR = os.environ.get("INPUT_DIR", "/kaggle/input")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

OTHER_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TOP_K = 5
MAX_NEW_TOKENS = 450  # releve du 30/07: 300 tronquait la reponse avant la citation finale
BATCH_SIZE = 8

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

reference_answers = []
gold_refs = []
for _, row in questions_test.iterrows():
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

queries = questions_test["question"].tolist()
q_emb = retriever.encode([f"query: {q}" for q in queries], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
sims = q_emb @ doc_embeddings.T
rag_contexts = []
for row in sims:
    top_idx = np.argsort(-row)[:TOP_K]
    rag_contexts.append([article_lookup[doc_ids[i]] for i in top_idx if doc_ids[i] in article_lookup])

del retriever
torch.cuda.empty_cache()


def build_messages(question, context_articles=None):
    user_parts = []
    if context_articles:
        ctx = "\n".join(f"- {a['reference']} : {a['article']}" for a in context_articles)
        user_parts.append(f"Contexte juridique :\n{ctx}\n")
    user_parts.append(f"Question : {question}")
    return [
        {"role": "system", "content": SYSTEM_PROMPT_FR},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


def batched_generate(model, tokenizer, prompts_messages, batch_size=BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    outputs = []
    for i in range(0, len(prompts_messages), batch_size):
        batch = prompts_messages[i : i + batch_size]
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        for j in range(len(batch)):
            new_tokens = gen[j][enc["input_ids"].shape[1] :]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        print(f"  batch {i // batch_size + 1}/{(len(prompts_messages) - 1) // batch_size + 1} done")
    return outputs


bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(OTHER_MODEL)
model = AutoModelForCausalLM.from_pretrained(OTHER_MODEL, quantization_config=bnb_config, device_map="auto")

from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def compute_metrics(answers):
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
    return {
        "rouge_l_f1_mean": float(np.mean(rouge_scores)),
        "rouge_l_f1_per_question": rouge_scores,
        "bertscore_f1_mean": float(np.mean(bert_f1.tolist())),
        "bertscore_f1_per_question": bert_f1.tolist(),
        "citation_precision_mean": float(np.nanmean(citation_precisions)) if citation_precisions else None,
        "citation_recall_mean": float(np.nanmean(citation_recalls)) if citation_recalls else None,
        "citation_exact_match_rate": float(np.mean(citation_exact)) if citation_exact else None,
        "hallucination_rate": float(np.mean(hallu_flags)),
    }


results = {"seed": SEED, "model": OTHER_MODEL, "n_questions": len(questions_test), "configs": {}}

print("=== Qwen C1: zero-shot ===")
t0 = time.time()
msgs_c1 = [build_messages(q, None) for q in queries]
answers_c1 = batched_generate(model, tokenizer, msgs_c1)
results["configs"]["Qwen_zero_shot"] = {"answers": answers_c1, "duration_seconds": time.time() - t0}
results["configs"]["Qwen_zero_shot"]["metrics"] = compute_metrics(answers_c1)
print(results["configs"]["Qwen_zero_shot"]["metrics"])

print("=== Qwen C2: RAG ===")
t0 = time.time()
msgs_c2 = [build_messages(q, ctx) for q, ctx in zip(queries, rag_contexts)]
answers_c2 = batched_generate(model, tokenizer, msgs_c2)
results["configs"]["Qwen_rag"] = {"answers": answers_c2, "duration_seconds": time.time() - t0}
results["configs"]["Qwen_rag"]["metrics"] = compute_metrics(answers_c2)
print(results["configs"]["Qwen_rag"]["metrics"])

results["reference_answers"] = reference_answers
results["gold_article_refs"] = gold_refs
results["categories"] = questions_test["category"].tolist()
results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

with open(f"{WORK_DIR}/other_llm_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("DONE")
