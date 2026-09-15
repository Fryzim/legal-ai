"""Configuration centrale du projet TER RAG vs Fine-tuning.

Toutes les valeurs ici ont été vérifiées empiriquement le 2026-07-28
(cf. DIFFICULTES.md / audit Phase 0) et non supposées depuis la littérature.
"""
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Reproductibilité
# ---------------------------------------------------------------------------
SEED = 42
# Passage sur RunPod (budget GPU moins contraint que Kaggle T4 gratuit):
# 3 seeds pour répondre explicitement à la critique d'A. Habrard
# ("only one experiment") -- moyenne ± écart-type sur toutes les métriques.
FINETUNE_SEEDS = [42, 123, 2026]


def set_all_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
TABLES_DIR = ROOT_DIR / "tables"

BSARD_ARTICLES_CSV = DATA_DIR / "articles.csv"
BSARD_QUESTIONS_TRAIN_CSV = DATA_DIR / "questions_train.csv"
BSARD_QUESTIONS_TEST_CSV = DATA_DIR / "questions_test.csv"
BSARD_QUESTIONS_SYNTHETIC_CSV = DATA_DIR / "questions_synthetic.csv"

# ---------------------------------------------------------------------------
# Dataset — BSARD (Belgian Statutory Article Retrieval Dataset)
# maastrichtlawtech/bsard sur HuggingFace, libre d'accès, non gated (vérifié).
#
# Faits vérifiés le 2026-07-28 (téléchargement + inspection réelle) :
#   - corpus:            22 633 articles (14 167 fédéraux, 8 466 régionaux)
#   - split train:        886 questions
#   - split test:          222 questions  <-- confirme le chiffre des CR
#   - split synthetic:  113 165 questions (paraphrases générées, non annotées à la main)
#   - colonnes questions: id, category, subcategory, question, extra_description, article_ids
#   - article_ids est un STRING ("947,948"), pas une liste -> bug déjà rencontré,
#     TOUJOURS parser avec .split(",")
#   - PAS de champ réponse rédigée: BSARD est un dataset de retrieval pur.
#     La "réponse de référence" pour ROUGE-L/BERTScore est donc construite comme
#     la CONCATÉNATION du texte des articles cités (voir data.py::build_reference_answer).
#     C'est une limite méthodologique à assumer explicitement dans le rapport.
# ---------------------------------------------------------------------------
HF_DATASET_BSARD = "maastrichtlawtech/bsard"

BSARD_CATEGORIES = [
    "Famille",
    "Logement",
    "Argent",
    "Justice",
    "Etrangers",
    "Travail",
    "Protection sociale",
]

EXPECTED_TEST_SIZE = 222
EXPECTED_TRAIN_SIZE = 886
EXPECTED_CORPUS_SIZE = 22633

# ---------------------------------------------------------------------------
# Modèles — accessibilité vérifiée le 2026-07-28 (curl anonyme -> HTTP 200)
# ---------------------------------------------------------------------------
# Mistral-7B-Instruct-v0.3: vérifié NON gated (contrairement à l'hypothèse initiale
# du brief), licence Apache 2.0, téléchargeable sans compte/token HuggingFace.
MODEL_MISTRAL_BASE = "mistralai/Mistral-7B-Instruct-v0.3"
# Miroir 4-bit pré-quantifié (plus rapide à charger sur Kaggle T4, garde en option).
MODEL_MISTRAL_4BIT_MIRROR = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"

EMBEDDING_MODEL_BASELINE = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MODEL_IMPROVED = "intfloat/multilingual-e5-large"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
# Le juge NE DOIT PAS être un modèle aussi comparé comme système (biais
# d'auto-préférence). Qwen2.5-7B-Instruct est utilisé comme second modèle de
# base pour tester la généralisation des conclusions (other_llm_job) -> il ne
# peut donc pas aussi être le juge. Phi-3.5-mini-instruct: non gated (licence
# MIT), absent des 4 configurations et du second modèle de base testés.
JUDGE_MODEL = "microsoft/Phi-3.5-mini-instruct"

# ---------------------------------------------------------------------------
# QLoRA — configuration alignée sur les comptes-rendus déjà envoyés à l'encadrant
# pour que les valeurs reproduites restent comparables.
#
# ECART CORRIGE (trouve en lisant contexte/chaab1.pdf le 29/07): la 2eme
# iteration historique (580 exemples, ROUGE-L 0.1612 / BERTScore 0.7043 --
# la reference a reproduire) utilisait r=32, PAS r=16. Le rapport de Chaabane
# le dit explicitement: "The LoRA rank was also increased from r=16 to r=32
# (...) roughly 42 million to 84 million [params]". Le run seed42/n=580
# deja execute sur Kaggle (loss 0.841) a tourne avec r=16 -- ce n'est donc
# PAS une reproduction fidele de la reference, juste un point de donnee
# valide en plus (a relabelliser comme tel, pas comme le run "principal").
# r=32 devient le defaut; r=16 redevient un point d'ablation (comme dans
# leur iteration 1, mais applique a 580 exemples pour isoler l'effet du
# rang de celui de la taille du train).
# ---------------------------------------------------------------------------
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES_ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
# Ablation cible LoRA (attention seule vs attention+MLP), prévue dans le brief
# initial mais jamais lancée -- à exécuter sur RunPod.
LORA_TARGET_MODULES_ATTN_MLP = LORA_TARGET_MODULES_ATTN + ["gate_proj", "up_proj", "down_proj"]
LORA_TARGET_MODULES = LORA_TARGET_MODULES_ATTN  # rétro-compatible: mode par défaut
QLORA_BITS = 4
QLORA_QUANT_TYPE = "nf4"
FINETUNE_EPOCHS = 3
FINETUNE_TRAIN_SIZE = 580  # itération 2 des CR (meilleur score rapporté)
# 100 questions réservées en validation, fixe et disjointe de TOUTES les tailles
# de train testées (comparabilité de la courbe d'apprentissage) -- jusqu'ici
# aucun split de validation n'existait, impossible de détecter le surapprentissage
# sur 580-886 exemples x 3 epochs.
FINETUNE_VAL_SIZE = 100
FINETUNE_TRAIN_SIZE_FULL = 786  # 886 - 100 (validation) = pool max disponible pour la courbe d'apprentissage
FINETUNE_BATCH_SIZE = 4
FINETUNE_GRAD_ACCUM = 4
FINETUNE_LR = 2e-4

# ---------------------------------------------------------------------------
# Suivi de coût RunPod (dimension "coût pratique" valorisée par Derby LLM).
# Tarifs horaires à ajuster selon le type de pod effectivement loué (placeholder
# indicatif on-demand, à corriger avec le tarif réel affiché au lancement).
# ---------------------------------------------------------------------------
RUNPOD_GPU_HOURLY_USD = {
    "RTX4090": 0.44,
    "A100_80GB": 1.64,
}


def estimate_runpod_cost_usd(duration_seconds: float, gpu_type: str = "RTX4090") -> float:
    hourly = RUNPOD_GPU_HOURLY_USD.get(gpu_type)
    if hourly is None:
        raise ValueError(f"Tarif RunPod inconnu pour {gpu_type}, ajouter dans RUNPOD_GPU_HOURLY_USD")
    return (duration_seconds / 3600.0) * hourly

# ---------------------------------------------------------------------------
# Retrieval / RAG
# ---------------------------------------------------------------------------
RETRIEVAL_TOP_K = 5  # nombre de fragments injectés par défaut (ablation k plus tard)
RECALL_KS = [1, 3, 5, 10, 20]

# ---------------------------------------------------------------------------
# Génération — même prompt pour les 4 configurations (comparaison équitable)
# ---------------------------------------------------------------------------
GENERATION_TEMPERATURE = 0.0
GENERATION_MAX_NEW_TOKENS = 400

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

CONFIGS = {
    "C1_zero_shot": {"use_rag": False, "use_finetuned": False},
    "C2_rag": {"use_rag": True, "use_finetuned": False},
    "C3_finetune": {"use_rag": False, "use_finetuned": True},
    "C4_finetune_rag": {"use_rag": True, "use_finetuned": True},
}

# ---------------------------------------------------------------------------
# Statistiques
# ---------------------------------------------------------------------------
BOOTSTRAP_N_RESAMPLES = 1000
CONFIDENCE_LEVEL = 0.95

# ---------------------------------------------------------------------------
# Régex de citation d'article (pour extraction + hallucination detection)
#
# Les identifiants d'article BSARD ne sont PAS tous numériques: environ 32% du
# corpus utilise des préfixes de lettres pour les codes régionaux
# ("N1.1", "L1122-9", "VI.61", "R.II.21-7", "D382", "259bis7"...). Un premier
# regex `\d+[a-zA-Z]?` ne matchait que 68% du corpus — corrigé pour capturer
# tout identifiant jusqu'à la virgule/fin, vérifié à 100% de couverture sur
# articles.csv['reference'].
# ---------------------------------------------------------------------------
ARTICLE_REFERENCE_REGEX = r"Art\.\s*([^,]+)"  # sur articles.csv['reference'] (gold)
ARTICLE_CITATION_REGEX = r"[Aa]rticle\s+([^\s,.;]+)"  # sur le texte généré par le modèle

# ---------------------------------------------------------------------------
# ~2.5% du corpus (555/22633 articles, 16/222 questions test) porte un suffixe
# d'annotation collé à l'ID extrait par ARTICLE_REFERENCE_REGEX
# ("275_DROIT_FUTUR", "1714bis_REGION_DE_BRUXELLES-CAPITALE") -- artefact du
# scraping de la source (note de variante régionale/temporelle), pas une partie
# citable de l'identifiant. Sans nettoyage: (1) la cible d'entraînement du
# fine-tuning apprend à générer des citations non-naturelles, (2) les métriques
# de citation pénalisent injustement un modèle qui cite correctement l'article
# sous sa forme humaine ("Article 1714bis"). Toujours appliquer après extraction,
# avant tout usage comme citation gold (entraînement ou métrique).
# ---------------------------------------------------------------------------
ARTICLE_ID_ANNOTATION_SUFFIX_REGEX = r"_[A-Z][A-Z_\-]+$"


def clean_article_ref_id(raw_id: str) -> str:
    import re

    return re.sub(ARTICLE_ID_ANNOTATION_SUFFIX_REGEX, "", raw_id)
