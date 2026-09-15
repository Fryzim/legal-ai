"""Job GPU (Kaggle ou RunPod): QLoRA fine-tuning de Mistral-7B-Instruct-v0.3 sur BSARD.

Config alignée sur les CR déjà envoyés (r=32, alpha=32, NF4 4-bit, 3 epochs,
580 exemples par défaut -- r=32 corrige le 29/07, cf. contexte/chaab1.pdf). Paramétrable par variables d'environnement pour
couvrir toutes les variantes (seeds multiples, ablation rang/cibles LoRA,
courbe d'apprentissage étendue, augmentation par données synthétiques) sans
dupliquer ce fichier à chaque combinaison -- un fichier dupliqué à chaque
run (n190/n380/r8/r32) a déjà démontré son risque de dérive (un correctif
appliqué à l'un n'est pas répercuté sur les autres).

Variables d'environnement reconnues (toutes optionnelles, défauts = run principal):
  FT_SEED               (def. 42)
  FT_TRAIN_SIZE         (def. 580)
  FT_LORA_R             (def. 32 -- corrige le 29/07: la reference historique 580ex.
                        utilisait r=32, pas r=16, cf. contexte/chaab1.pdf)
  FT_TARGET_MODULES     "attn" (def.) ou "attn_mlp" (ablation cibles LoRA)
  FT_TRAIN_SOURCE       "official" (def.) ou "official_plus_synthetic"
  FT_N_SYNTHETIC_EXTRA  nb d'exemples supplémentaires puisés dans le split
                        synthétique BSARD si FT_TRAIN_SOURCE=official_plus_synthetic (def. 0)
  FT_RUN_TAG            suffixe des fichiers de sortie (def. dérivé des params ci-dessus)
  FT_CODE_FILTER        nom exact du code de loi (ex. "Code Civil") pour restreindre le
                        pool d'entraînement -- specialiste vs generaliste (def. aucun filtre)

Validation: 100 questions réservées (seed fixe 999, indépendante de FT_SEED)
pour être IDENTIQUES et disjointes de TOUTES les tailles de train testées
(190/380/580/786) -- sans ça, impossible de détecter le surapprentissage sur
quelques centaines d'exemples x 3 epochs, ni de comparer la courbe
d'apprentissage sur un même jeu d'évaluation.

Utilise le miroir pré-quantifié unsloth/mistral-7b-instruct-v0.3-bnb-4bit pour
réduire le temps de téléchargement (~4-5 Go au lieu de ~15 Go en fp16), documenté
comme substitution pratique dans DIFFICULTES.md.

Sortie: {WORK_DIR}/adapter_{RUN_TAG}/ (poids LoRA) + finetune_meta_{RUN_TAG}.json.
"""
import json
import re
import subprocess
import os
import sys
import time
import urllib.request
from pathlib import Path

# Les identifiants d'article BSARD ne sont pas tous numériques (~32% du corpus
# utilise des préfixes de lettres pour les codes régionaux: "N1.1", "L1122-9",
# "VI.61"...). Ce regex capture tout jusqu'à la virgule, vérifié à 100% de
# couverture sur articles.csv['reference'] (cf. DIFFICULTES.md).
ARTICLE_REFERENCE_REGEX = r"Art\.\s*([^,]+)"
# ~2.5% du corpus (555/22633) porte un suffixe d'annotation collé à l'ID
# ("275_DROIT_FUTUR", "1714bis_REGION_DE_BRUXELLES-CAPITALE") -- artefact du
# scraping de la source, pas une partie citable de l'identifiant. Sans ce
# nettoyage, la cible d'entraînement du fine-tuning inclut ce suffixe non-naturel
# et les métriques de citation pénalisent injustement le modèle (16/222
# questions test concernées). Supprimé avant tout usage comme citation gold.
ARTICLE_ID_ANNOTATION_SUFFIX_REGEX = r"_[A-Z][A-Z_\-]+$"


def clean_article_ref_id(raw_id: str) -> str:
    return re.sub(ARTICLE_ID_ANNOTATION_SUFFIX_REGEX, "", raw_id)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U", "peft", "trl", "bitsandbytes", "accelerate"],
    check=True,
)

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Chemins portables: par defaut Kaggle, surchargeables via variables
# d'environnement pour tourner ailleurs (RunPod, etc.) sans modifier le script.
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")
INPUT_DIR = os.environ.get("INPUT_DIR", "/kaggle/input")

SEED = int(os.environ.get("FT_SEED", 42))
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
LORA_R = int(os.environ.get("FT_LORA_R", 32))
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES_MODE = os.environ.get("FT_TARGET_MODULES", "attn")
TARGET_MODULES = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attn_mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}[TARGET_MODULES_MODE]
EPOCHS = 3
TRAIN_SIZE = int(os.environ.get("FT_TRAIN_SIZE", 580))
VAL_SIZE = int(os.environ.get("FT_VAL_SIZE", 100))
VAL_SPLIT_SEED = 999  # fixe, independant de FT_SEED: meme jeu de validation pour tous les runs
TRAIN_SOURCE = os.environ.get("FT_TRAIN_SOURCE", "official")  # "official" ou "official_plus_synthetic"
N_SYNTHETIC_EXTRA = int(os.environ.get("FT_N_SYNTHETIC_EXTRA", 0))
# Specialiste sur un seul code de loi (ex. "Code Civil") -- operationnalise le
# conseil d'A. Habrard ("focus on particular subgroups of laws") comme une
# vraie experience de reduction de la difficulte, pas juste une stratification
# descriptive: 382 questions train / 86 questions test touchent le Code Civil.
CODE_FILTER = os.environ.get("FT_CODE_FILTER", "")
BATCH_SIZE = 4
GRAD_ACCUM = 4
LR = 2e-4

RUN_TAG = os.environ.get(
    "FT_RUN_TAG",
    f"seed{SEED}_n{TRAIN_SIZE}_r{LORA_R}_{TARGET_MODULES_MODE}"
    + ("_synth" if TRAIN_SOURCE == "official_plus_synthetic" else "")
    + (f"_{CODE_FILTER.replace(' ', '')}" if CODE_FILTER else ""),
)

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

# --- Data ---
DATA_DIR = Path(f"{WORK_DIR}/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
HF_BASE = "https://huggingface.co/datasets/maastrichtlawtech/bsard/resolve/main"
_needed_files = ["articles.csv", "questions_train.csv"]
if TRAIN_SOURCE == "official_plus_synthetic":
    _needed_files.append("questions_synthetic.csv")
for fname in _needed_files:
    dest = DATA_DIR / fname
    if not dest.exists():
        urllib.request.urlretrieve(f"{HF_BASE}/{fname}", dest)

articles = pd.read_csv(DATA_DIR / "articles.csv")
article_lookup = articles.set_index("id").to_dict(orient="index")
questions_train = pd.read_csv(DATA_DIR / "questions_train.csv")
questions_train["article_ids"] = questions_train["article_ids"].apply(
    lambda x: [int(i) for i in str(x).split(",") if i.strip()]
)

# Validation fixe: reservee AVANT l'echantillonnage de TRAIN_SIZE, avec un seed
# different de FT_SEED, pour rester identique et disjointe quelle que soit la
# taille de train ou le seed d'entrainement teste.
val_df = questions_train.sample(n=min(VAL_SIZE, len(questions_train)), random_state=VAL_SPLIT_SEED)
train_pool = questions_train.drop(val_df.index)
val_df = val_df.reset_index(drop=True)

if CODE_FILTER:
    codes_by_article = articles.set_index("id")["code"].to_dict()
    train_pool = train_pool[
        train_pool["article_ids"].apply(lambda ids: any(codes_by_article.get(i) == CODE_FILTER for i in ids))
    ]
    print(f"n_train_pool_apres_filtre_code({CODE_FILTER}) = {len(train_pool)}")

df = train_pool.sample(n=min(TRAIN_SIZE, len(train_pool)), random_state=SEED).reset_index(drop=True)

if TRAIN_SOURCE == "official_plus_synthetic" and N_SYNTHETIC_EXTRA > 0:
    # Split synthetic BSARD (113 165 paraphrases générées, jamais annotées à la
    # main) -- ressource gratuite inexploitée jusqu'ici. Répond à la perspective
    # Derby LLM "étudier l'impact de la taille du jeu de données" au-delà des
    # 886 questions officielles disponibles.
    questions_synthetic = pd.read_csv(DATA_DIR / "questions_synthetic.csv")
    questions_synthetic["article_ids"] = questions_synthetic["article_ids"].apply(
        lambda x: [int(i) for i in str(x).split(",") if i.strip()]
    )
    synth_sample = questions_synthetic.sample(
        n=min(N_SYNTHETIC_EXTRA, len(questions_synthetic)), random_state=SEED
    ).reset_index(drop=True)
    df = pd.concat([df, synth_sample], ignore_index=True)
    print(f"n_synthetic_added = {len(synth_sample)}")

def build_examples(source_df):
    built = []
    for _, row in source_df.iterrows():
        ref_parts = []
        article_refs = []
        for aid in row["article_ids"]:
            art = article_lookup.get(aid)
            if art is not None:
                ref_parts.append(f"{art['reference']}: {art['article']}")
                m = re.search(ARTICLE_REFERENCE_REGEX, str(art["reference"]))
                if m:
                    article_refs.append(clean_article_ref_id(m.group(1).strip()))
        reference_answer = "\n".join(ref_parts)
        citation = ", ".join(f"Article {r}" for r in article_refs)
        completion = f"{reference_answer}\n\n{citation}".strip()
        built.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT_FR},
                    {"role": "user", "content": f"Question : {row['question']}"},
                    {"role": "assistant", "content": completion},
                ]
            }
        )
    return built


examples = build_examples(df)
val_examples = build_examples(val_df)
print(f"n_train_examples = {len(examples)}, n_val_examples = {len(val_examples)}")

# --- Model ---
t0_load = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
load_duration = time.time() - t0_load
print(f"model loaded in {load_duration:.1f}s")

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
pct_trainable = 100 * trainable / total
print(f"trainable params: {trainable} / {total} ({pct_trainable:.3f}%)")

# --- Training ---
def format_fn(ex):
    return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}


ds = Dataset.from_list(examples).map(format_fn)
val_ds = Dataset.from_list(val_examples).map(format_fn)

from trl import SFTConfig, SFTTrainer

output_dir = f"{WORK_DIR}/adapter_{RUN_TAG}"
args = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    bf16=True,
    seed=SEED,
    report_to=[],
    dataset_text_field="text",
    max_length=1024,
    loss_type="nll",
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=ds,
    eval_dataset=val_ds,
)

t0_train = time.time()
train_result = trainer.train()
train_duration = time.time() - t0_train

trainer.save_model(output_dir)

# Courbe train/eval par epoch -- seul moyen de detecter un surapprentissage sur
# aussi peu d'exemples. eval_loss qui diverge de train_loss au fil des epochs
# = signal direct de surapprentissage, en particulier pertinent pour comparer
# l'ablation cibles LoRA (attn vs attn+mlp, plus de capacite = plus de risque
# de surapprentissage sur 580 exemples).
eval_history = [
    {"epoch": e["epoch"], "eval_loss": e["eval_loss"]}
    for e in trainer.state.log_history
    if "eval_loss" in e
]
train_loss_history = [
    {"epoch": e["epoch"], "train_loss": e["loss"]}
    for e in trainer.state.log_history
    if "loss" in e and "eval_loss" not in e
]
final_eval_loss = eval_history[-1]["eval_loss"] if eval_history else None

meta = {
    "run_tag": RUN_TAG,
    "seed": SEED,
    "base_model": MODEL_NAME,
    "epochs": EPOCHS,
    "lora_r": LORA_R,
    "lora_alpha": LORA_ALPHA,
    "target_modules_mode": TARGET_MODULES_MODE,
    "target_modules": TARGET_MODULES,
    "train_source": TRAIN_SOURCE,
    "n_synthetic_extra": N_SYNTHETIC_EXTRA,
    "code_filter": CODE_FILTER or None,
    "n_train_examples": len(examples),
    "n_val_examples": len(val_examples),
    "trainable_params": int(trainable),
    "total_params": int(total),
    "pct_trainable": pct_trainable,
    "model_load_seconds": load_duration,
    "train_duration_seconds": train_duration,
    "final_train_loss": train_result.training_loss,
    "final_eval_loss": final_eval_loss,
    "overfit_gap": (final_eval_loss - train_loss_history[-1]["train_loss"]) if (final_eval_loss and train_loss_history) else None,
    "train_loss_history": train_loss_history,
    "eval_loss_history": eval_history,
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
with open(f"{WORK_DIR}/finetune_meta_{RUN_TAG}.json", "w") as f:
    json.dump(meta, f, indent=2)

print("DONE")
print(json.dumps(meta, indent=2))
