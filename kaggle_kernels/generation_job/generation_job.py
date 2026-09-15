"""Kaggle kernel (GPU T4): génère les réponses des 4 configurations (C1-C4) sur
les 222 questions test réelles + calcule les métriques (ROUGE-L, BERTScore,
citation, hallucination).

Dépend des sorties de deux autres kernels attachés via kernel_sources:
  - ter-bsard-retrieval-eval  -> index e5-large (index_e5_large.npz)
  - ter-bsard-qlora-finetune  -> adaptateur LoRA (adapter/)

Sortie: /kaggle/working/generation_results.json
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
    [sys.executable, "-m", "pip", "install", "-q", "-U", "peft", "bitsandbytes", "accelerate", "rouge_score", "bert_score", "sentence-transformers"],
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

BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
TOP_K = 5
# 300 coupait fréquemment la réponse avant la citation d'article finale
# (constaté en testant l'arène humaine sur les données du 30/07 : réponses
# tronquées en plein mot, ex. "Code de Droit Econom[ique]") -- possible
# facteur expliquant en partie l'exactitude de citation très basse observée.
# Relevé à 450 pour laisser la place a la citation.
MAX_NEW_TOKENS = 450
BATCH_SIZE = 8

# Ablation de diversité d'échantillonnage (réponse directe à A. Habrard: "make
# tests on different test data/sampling to evaluate the average behavior").
# Appliquée uniquement aux 2 configs RAG (C2, C4) pour borner le coût GPU.
SAMPLING_TEMPERATURE = 0.3
SAMPLING_SEEDS = [101, 202, 303]

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

# Les identifiants d'article BSARD ne sont pas tous numériques (~32% du corpus
# utilise des préfixes de lettres pour les codes régionaux: "N1.1", "L1122-9",
# "VI.61"...). ARTICLE_REFERENCE_REGEX (gold, sur articles.csv) capture tout
# jusqu'à la virgule, vérifié à 100% de couverture. ARTICLE_CITATION_REGEX
# (sur le texte généré par le modèle) capture le token suivant "Article ".
ARTICLE_REFERENCE_REGEX = r"Art\.\s*([^,]+)"
ARTICLE_CITATION_REGEX = r"[Aa]rticle\s+([^\s,.;]+)"
# ~2.5% du corpus (555/22633) porte un suffixe d'annotation collé à l'ID
# ("275_DROIT_FUTUR", "1714bis_REGION_DE_BRUXELLES-CAPITALE") -- artefact du
# scraping de la source, pas une partie citable. 16/222 questions test
# concernées: sans ce nettoyage la citation gold est incitable dans le format
# demandé au modèle, ce qui pénalise injustement precision/recall/hallucination.
ARTICLE_ID_ANNOTATION_SUFFIX_REGEX = r"_[A-Z][A-Z_\-]+$"

# --- Locate attached kernel outputs ---
def find_file(pattern):
    matches = glob.glob(f"{INPUT_DIR}/**/{pattern}", recursive=True)
    if not matches:
        raise FileNotFoundError(f"No match for {pattern} under {INPUT_DIR}")
    return matches[0]


index_path = find_file("index_e5_large.npz")
adapter_dir = str(Path(find_file("adapter_config.json")).parent)
print(f"e5-large index: {index_path}")
print(f"LoRA adapter dir: {adapter_dir}")

# --- Data ---
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
match_rate = (ref_ids != "").mean()
assert match_rate > 0.99, f"couverture regex de référence anormalement basse: {match_rate:.3f}"
valid_article_refs = set(ref_ids.tolist())
valid_article_refs.discard("")

questions_test = pd.read_csv(DATA_DIR / "questions_test.csv")
questions_test["article_ids"] = questions_test["article_ids"].apply(
    lambda x: [int(i) for i in str(x).split(",") if i.strip()]
)
assert len(questions_test) == 222

# --- RAG retrieval (e5-large) ---
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

t0 = time.time()
retriever = SentenceTransformer("intfloat/multilingual-e5-large", device=device)
idx = np.load(index_path)
doc_ids = idx["doc_ids"].tolist()
doc_embeddings = idx["embeddings"]

queries = questions_test["question"].tolist()
q_emb = retriever.encode(
    [f"query: {q}" for q in queries], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
)
sims = q_emb @ doc_embeddings.T
rag_contexts = []
for row in sims:
    top_idx = np.argsort(-row)[:TOP_K]
    ctx = [article_lookup[doc_ids[i]] for i in top_idx if doc_ids[i] in article_lookup]
    rag_contexts.append(ctx)

del retriever
torch.cuda.empty_cache()
print(f"RAG retrieval done in {time.time() - t0:.1f}s")

# --- Reference answers (concat cited article text) + gold article refs ---
reference_answers = []
gold_refs = []
for _, row in questions_test.iterrows():
    parts, refs = [], []
    for aid in row["article_ids"]:
        art = article_lookup.get(aid)
        if art is not None:
            parts.append(f"{art['reference']}: {art['article']}")
            ref_id = extract_reference_id(art["reference"])
            if ref_id:
                refs.append(ref_id)
    reference_answers.append("\n".join(parts))
    gold_refs.append(refs)

# --- Generation helpers ---
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


def batched_generate(
    model, tokenizer, prompts_messages, batch_size=BATCH_SIZE, max_new_tokens=MAX_NEW_TOKENS,
    temperature=0.0, seed=SEED,
):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    do_sample = temperature > 0.0
    if do_sample:
        torch.manual_seed(seed)
    outputs = []
    for i in range(0, len(prompts_messages), batch_size):
        batch = prompts_messages[i : i + batch_size]
        texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=do_sample, pad_token_id=tokenizer.eos_token_id)
        if do_sample:
            gen_kwargs["temperature"] = temperature
        with torch.no_grad():
            gen = model.generate(**enc, **gen_kwargs)
        for j in range(len(batch)):
            new_tokens = gen[j][enc["input_ids"].shape[1] :]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        print(f"  batch {i // batch_size + 1}/{(len(prompts_messages) - 1) // batch_size + 1} done")
    return outputs


# Trouve en testant l'arene humaine le 30/07 (cf. src/metrics.py::has_question_echo
# pour le detail) : le RAG sans fine-tuning (C2) recopie souvent le motif
# "Question : ... Reponse : ..." du prompt au lieu de repondre directement.
QUESTION_ECHO_REGEX = re.compile(r"r[ée]ponse\s*:\s*", re.IGNORECASE)


def has_question_echo(text):
    return bool(re.search(r"question\s*:", text, re.IGNORECASE)) and bool(QUESTION_ECHO_REGEX.search(text))


def compute_metrics(answers, reference_answers, gold_refs, valid_article_refs, rouge, bert_score_fn):
    rouge_scores = [rouge.score(ref, ans)["rougeL"].fmeasure for ref, ans in zip(reference_answers, answers)]
    _, _, bert_f1 = bert_score_fn(
        answers, reference_answers, model_type="distilbert-base-multilingual-cased", lang="fr", verbose=False
    )
    # Precision/recall par question ne sont pas toujours definis (ex: predicted et
    # gold tous deux vides -> NaN), mais exact_match et hallucination le sont
    # TOUJOURS (0.0/1.0) -- necessaires par question, pas seulement en moyenne,
    # pour le test d'hypothese format/citation (citation_format_analysis.py) et
    # le modele de prediction de fiabilite (quality_prediction.py).
    citation_precisions_per_q, citation_recalls_per_q, citation_exact_per_q = [], [], []
    hallucinated_flags = []
    for ans, refs in zip(answers, gold_refs):
        predicted = set(re.findall(ARTICLE_CITATION_REGEX, ans))
        gold = set(refs)
        if predicted or gold:
            tp = len(predicted & gold)
            citation_precisions_per_q.append(tp / len(predicted) if predicted else 0.0)
            citation_recalls_per_q.append(tp / len(gold) if gold else float("nan"))
            citation_exact_per_q.append(float(predicted == gold))
        else:
            citation_precisions_per_q.append(float("nan"))
            citation_recalls_per_q.append(float("nan"))
            citation_exact_per_q.append(1.0)  # rien a citer, rien cite -> correct par convention
        hallucinated_flags.append(float(any(c not in valid_article_refs for c in predicted)))
    echo_flags = [float(has_question_echo(ans)) for ans in answers]
    return {
        "rouge_l_f1_mean": float(np.mean(rouge_scores)),
        "rouge_l_f1_per_question": rouge_scores,
        "bertscore_f1_mean": float(np.mean(bert_f1.tolist())),
        "bertscore_f1_per_question": bert_f1.tolist(),
        "citation_precision_mean": float(np.nanmean(citation_precisions_per_q)) if citation_precisions_per_q else None,
        "citation_recall_mean": float(np.nanmean(citation_recalls_per_q)) if citation_recalls_per_q else None,
        "citation_exact_match_rate": float(np.mean(citation_exact_per_q)) if citation_exact_per_q else None,
        "hallucination_rate": float(np.mean(hallucinated_flags)),
        "question_echo_rate": float(np.mean(echo_flags)),
        "question_echo_per_question": echo_flags,
        "citation_precision_per_question": citation_precisions_per_q,
        "citation_recall_per_question": citation_recalls_per_q,
        "citation_exact_match_per_question": citation_exact_per_q,
        "hallucination_per_question": hallucinated_flags,
    }


results = {
    "seed": SEED,
    "n_questions": len(questions_test),
    "top_k": TOP_K,
    "configs": {},
    "sampling_ablation": {},
}

from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer

rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)


def run_and_score(model, tokenizer, msgs, temperature=0.0, seed=SEED):
    t0 = time.time()
    answers = batched_generate(model, tokenizer, msgs, temperature=temperature, seed=seed)
    metrics = compute_metrics(answers, reference_answers, gold_refs, valid_article_refs, rouge, bert_score_fn)
    return {"answers": answers, "duration_seconds": time.time() - t0, "metrics": metrics}


# --- Load base model (4-bit, for C1 and C2) ---
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
print(f"base model loaded in {time.time() - t0:.1f}s")

print("=== C1: zero-shot ===")
msgs_c1 = [build_messages(q, None) for q in queries]
results["configs"]["C1_zero_shot"] = run_and_score(base_model, tokenizer, msgs_c1)
print("C1 metrics:", results["configs"]["C1_zero_shot"]["metrics"])

print("=== C2: RAG (greedy, temperature=0) ===")
msgs_c2 = [build_messages(q, ctx) for q, ctx in zip(queries, rag_contexts)]
results["configs"]["C2_rag"] = run_and_score(base_model, tokenizer, msgs_c2)
print("C2 metrics:", results["configs"]["C2_rag"]["metrics"])

print("=== C2 sampling ablation (temperature=0.3, 3 seeds) ===")
c2_sampling_runs = []
for s in SAMPLING_SEEDS:
    run = run_and_score(base_model, tokenizer, msgs_c2, temperature=SAMPLING_TEMPERATURE, seed=s)
    run["seed"] = s
    c2_sampling_runs.append(run)
    print(f"  seed={s} rouge_l={run['metrics']['rouge_l_f1_mean']:.4f}")
results["sampling_ablation"]["C2_rag"] = c2_sampling_runs

del base_model
torch.cuda.empty_cache()

# --- Load fine-tuned model (base + LoRA adapter, for C3 and C4) ---
from peft import PeftModel

t0 = time.time()
base_for_ft = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")
ft_model = PeftModel.from_pretrained(base_for_ft, adapter_dir)
print(f"finetuned model loaded in {time.time() - t0:.1f}s")

print("=== C3: fine-tune seul ===")
msgs_c3 = [build_messages(q, None) for q in queries]
results["configs"]["C3_finetune"] = run_and_score(ft_model, tokenizer, msgs_c3)
print("C3 metrics:", results["configs"]["C3_finetune"]["metrics"])

print("=== C4: fine-tune + RAG (greedy, temperature=0) ===")
msgs_c4 = [build_messages(q, ctx) for q, ctx in zip(queries, rag_contexts)]
results["configs"]["C4_finetune_rag"] = run_and_score(ft_model, tokenizer, msgs_c4)
print("C4 metrics:", results["configs"]["C4_finetune_rag"]["metrics"])

print("=== C4 sampling ablation (temperature=0.3, 3 seeds) ===")
c4_sampling_runs = []
for s in SAMPLING_SEEDS:
    run = run_and_score(ft_model, tokenizer, msgs_c4, temperature=SAMPLING_TEMPERATURE, seed=s)
    run["seed"] = s
    c4_sampling_runs.append(run)
    print(f"  seed={s} rouge_l={run['metrics']['rouge_l_f1_mean']:.4f}")
results["sampling_ablation"]["C4_finetune_rag"] = c4_sampling_runs

del ft_model, base_for_ft
torch.cuda.empty_cache()

# --- Résumé de l'ablation de diversité d'échantillonnage (moyenne +/- écart-type sur 3 seeds) ---
for cfg_key, runs in results["sampling_ablation"].items():
    rouge_means = [r["metrics"]["rouge_l_f1_mean"] for r in runs]
    bert_means = [r["metrics"]["bertscore_f1_mean"] for r in runs]
    hallu_rates = [r["metrics"]["hallucination_rate"] for r in runs]
    print(
        f"{cfg_key} sampling (T={SAMPLING_TEMPERATURE}, n={len(runs)} seeds): "
        f"ROUGE-L={np.mean(rouge_means):.4f}+/-{np.std(rouge_means):.4f}, "
        f"BERTScore={np.mean(bert_means):.4f}+/-{np.std(bert_means):.4f}, "
        f"hallucination={np.mean(hallu_rates):.4f}+/-{np.std(hallu_rates):.4f}"
    )

results["reference_answers"] = reference_answers
results["gold_article_refs"] = gold_refs
results["questions"] = queries
results["categories"] = questions_test["category"].tolist()
results["sampling_temperature"] = SAMPLING_TEMPERATURE
results["sampling_seeds"] = SAMPLING_SEEDS
results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

with open(f"{WORK_DIR}/generation_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("DONE")
