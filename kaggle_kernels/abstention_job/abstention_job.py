"""Kaggle kernel (GPU T4): évaluation de la capacité d'abstention (brief §4.2).

50 questions construites en 3 catégories, toutes attendant une abstention:
  - 20 questions non-juridiques (hors domaine complet)
  - 15 questions de droit étranger (France, USA, etc. -- hors du corpus belge)
  - 15 questions BSARD réelles dont l'article de référence est explicitement
    retiré de l'index au moment du retrieval (contexte RAG volontairement
    privé de la bonne source)

Testé sur C1 (zero-shot) et C2 (RAG) avec le modèle de base -- ne dépend PAS
du fine-tuning, peut tourner immédiatement. Mesure le taux d'abstention
correcte (devrait s'abstenir et le fait) -- pas de "fausse abstention" ici
car toutes les questions de ce jeu appellent une abstention (le taux de
fausse abstention se mesure séparément sur les 222 questions valides du
split test, cf. generation_results.json).

Dépend du kernel ter-bsard-retrieval-eval (index e5-large).
Sortie: /kaggle/working/abstention_results.json
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

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes", "accelerate", "sentence-transformers"], check=True)

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Chemins portables: par defaut Kaggle, surchargeables via variables
# d'environnement pour tourner ailleurs (RunPod, etc.) sans modifier le script.
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")
INPUT_DIR = os.environ.get("INPUT_DIR", "/kaggle/input")

SEED = 42
BASE_MODEL = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
MAX_NEW_TOKENS = 450  # releve du 30/07: 300 tronquait la reponse avant la citation finale
BATCH_SIZE = 8
TOP_K = 5
N_EXCLUDED_ARTICLE_QUESTIONS = 15

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

ABSTENTION_PATTERNS = [
    r"\bje ne sais pas\b",
    r"\bje n'ai pas (?:la|de) r[ée]ponse\b",
    r"\bje ne (?:peux|suis en mesure de) (?:pas )?r[ée]pondre\b",
    r"\bpas assez d'informations\b",
    r"\bne dispose pas d'informations\b",
]


def is_abstention(text):
    t = text.lower()
    return any(re.search(p, t) for p in ABSTENTION_PATTERNS)


np.random.seed(SEED)
torch.manual_seed(SEED)

NON_LEGAL_QUESTIONS = [
    "Quelle est la recette traditionnelle des gaufres de Liège ?",
    "Comment fonctionne la photosynthèse chez les plantes ?",
    "Quel est le plus haut sommet des Alpes belges ?",
    "Quelle est la capitale de l'Australie ?",
    "Comment fait-on cuire un œuf à la coque parfait ?",
    "Quels sont les symptômes de la grippe saisonnière ?",
    "Quelle est la différence entre un cumulonimbus et un cirrus ?",
    "Comment installer une étagère murale en placo ?",
    "Quel est le principe de fonctionnement d'un moteur à combustion ?",
    "Quelle est la meilleure période pour visiter les Ardennes ?",
    "Comment se forme un arc-en-ciel ?",
    "Quels sont les ingrédients d'une carbonade flamande ?",
    "Quelle est la vitesse de la lumière dans le vide ?",
    "Comment entraîner un chiot à la propreté ?",
    "Quel est le cycle de vie d'un papillon ?",
    "Quelle est la théorie de la relativité générale d'Einstein ?",
    "Comment fonctionne un panneau solaire photovoltaïque ?",
    "Quels sont les bienfaits du yoga sur le stress ?",
    "Quelle est l'histoire de l'Atomium à Bruxelles ?",
    "Comment programmer une boucle for en Python ?",
]

FOREIGN_LAW_QUESTIONS = [
    "Quelle est la procédure de divorce par consentement mutuel en France ?",
    "Quel est le régime du RSA (revenu de solidarité active) en France ?",
    "Quelles sont les règles du Chapter 7 bankruptcy aux États-Unis ?",
    "Quel est le délai de préavis légal de licenciement en Allemagne ?",
    "Comment fonctionne le système de visa H1-B aux États-Unis ?",
    "Quelles sont les règles du bail commercial au Québec ?",
    "Quel est le montant du salaire minimum légal au Luxembourg ?",
    "Comment fonctionne le Universal Credit au Royaume-Uni ?",
    "Quelles sont les conditions d'obtention du permis de séjour en Suisse ?",
    "Quel est le régime fiscal des auto-entrepreneurs en France ?",
    "Comment fonctionne la garde à vue en droit pénal français ?",
    "Quelles sont les règles de succession en droit marocain ?",
    "Quel est le statut du PACS en droit français ?",
    "Comment fonctionne le small claims court en Angleterre ?",
    "Quelles sont les règles de copropriété au Canada ?",
]

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
questions_test = pd.read_csv(DATA_DIR / "questions_test.csv")
questions_test["article_ids"] = questions_test["article_ids"].apply(
    lambda x: [int(i) for i in str(x).split(",") if i.strip()]
)

single_article_df = questions_test[questions_test["article_ids"].apply(len) == 1].sample(
    n=N_EXCLUDED_ARTICLE_QUESTIONS, random_state=SEED
)

abstention_items = []
qid = 1
for q in NON_LEGAL_QUESTIONS:
    abstention_items.append({"id": qid, "question": q, "abstention_type": "non_juridique", "excluded_article_id": None})
    qid += 1
for q in FOREIGN_LAW_QUESTIONS:
    abstention_items.append({"id": qid, "question": q, "abstention_type": "droit_etranger", "excluded_article_id": None})
    qid += 1
for _, row in single_article_df.iterrows():
    abstention_items.append(
        {
            "id": qid,
            "question": row["question"],
            "abstention_type": "article_retire_index",
            "excluded_article_id": row["article_ids"][0],
        }
    )
    qid += 1

assert len(abstention_items) == 50
print(f"n_abstention_questions = {len(abstention_items)}")

# --- Retrieval (e5-large), excluant l'article gold pour le 3e type ---
def find_file(pattern):
    matches = glob.glob(f"{INPUT_DIR}/**/{pattern}", recursive=True)
    if not matches:
        raise FileNotFoundError(f"No match for {pattern} under {INPUT_DIR}")
    return matches[0]


index_path = find_file("index_e5_large.npz")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

from sentence_transformers import SentenceTransformer

retriever = SentenceTransformer("intfloat/multilingual-e5-large", device=device)
idx = np.load(index_path)
doc_ids = idx["doc_ids"].tolist()
doc_embeddings = idx["embeddings"]

queries = [it["question"] for it in abstention_items]
q_emb = retriever.encode([f"query: {q}" for q in queries], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
sims = q_emb @ doc_embeddings.T

del retriever
torch.cuda.empty_cache()

rag_contexts = []
for i, item in enumerate(abstention_items):
    row = sims[i]
    top_idx = np.argsort(-row)[:TOP_K + 1]  # +1 marge pour compenser une éventuelle exclusion
    ctx_ids = [doc_ids[j] for j in top_idx if doc_ids[j] in article_lookup]
    if item["excluded_article_id"] is not None:
        ctx_ids = [aid for aid in ctx_ids if aid != item["excluded_article_id"]]
    ctx_ids = ctx_ids[:TOP_K]
    rag_contexts.append([article_lookup[aid] for aid in ctx_ids])

# --- Generation ---
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
    return outputs


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto")

results = {"seed": SEED, "n_questions": len(abstention_items), "configs": {}}

print("=== C1: zero-shot ===")
msgs_c1 = [build_messages(it["question"], None) for it in abstention_items]
answers_c1 = batched_generate(model, tokenizer, msgs_c1)
results["configs"]["C1_zero_shot"] = {"answers": answers_c1}

print("=== C2: RAG (contexte privé de la source gold pour le type 3) ===")
msgs_c2 = [build_messages(it["question"], ctx) for it, ctx in zip(abstention_items, rag_contexts)]
answers_c2 = batched_generate(model, tokenizer, msgs_c2)
results["configs"]["C2_rag"] = {"answers": answers_c2}

for cfg_name, cfg_data in results["configs"].items():
    answers = cfg_data["answers"]
    abstained = [is_abstention(a) for a in answers]
    cfg_data["abstained"] = abstained
    cfg_data["correct_abstention_rate"] = float(np.mean(abstained))
    by_type = {}
    for t in ["non_juridique", "droit_etranger", "article_retire_index"]:
        idxs = [i for i, it in enumerate(abstention_items) if it["abstention_type"] == t]
        by_type[t] = float(np.mean([abstained[i] for i in idxs])) if idxs else None
    cfg_data["correct_abstention_rate_by_type"] = by_type
    print(cfg_name, "abstention rate:", cfg_data["correct_abstention_rate"], by_type)

results["questions"] = [it["question"] for it in abstention_items]
results["abstention_types"] = [it["abstention_type"] for it in abstention_items]
results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

with open(f"{WORK_DIR}/abstention_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("DONE")
