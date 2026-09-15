"""Métriques: retrieval (Recall@k, MRR, nDCG), génération (ROUGE-L, BERTScore),
fiabilité (exactitude de citation, hallucination, fidélité), abstention.

La métrique de fidélité (`fidelity_score`) s'inspire du protocole de Derby LLM
(Bouvard et al., APIA@PFIA 2024 -- référence imposée par F. Jacquenet pour la
comparaison) : extraction des "passages d'intérêt" (entités nommées + éléments
factuels précis: nombres, emails, URLs) dans la réponse générée et dans le
texte de référence, puis proportion de ceux de la réponse retrouvés dans la
référence. Réimplémentation adaptée à notre corpus (texte de loi belge, pas
des pages web d'entreprise) -- formule générique de recouvrement d'entités,
pas du code copié.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def recall_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    if not relevant_ids:
        return np.nan
    topk = set(retrieved_ids[:k])
    hit = len(topk & set(relevant_ids))
    return hit / len(relevant_ids)


def mrr_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int = 10) -> float:
    relevant_set = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int = 10) -> float:
    relevant_set = set(relevant_ids)
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    all_retrieved: list[list[int]],
    all_relevant: list[list[int]],
    ks: Iterable[int] = config.RECALL_KS,
) -> dict:
    out = {}
    for k in ks:
        vals = [recall_at_k(r, rel, k) for r, rel in zip(all_retrieved, all_relevant)]
        out[f"recall@{k}"] = float(np.nanmean(vals))
    mrr_vals = [mrr_at_k(r, rel, 10) for r, rel in zip(all_retrieved, all_relevant)]
    ndcg_vals = [ndcg_at_k(r, rel, 10) for r, rel in zip(all_retrieved, all_relevant)]
    out["mrr@10"] = float(np.mean(mrr_vals))
    out["ndcg@10"] = float(np.mean(ndcg_vals))
    out["n_questions"] = len(all_retrieved)
    return out


# ---------------------------------------------------------------------------
# Génération — similarité de surface
# ---------------------------------------------------------------------------
def rouge_l_f1(prediction: str, reference: str) -> float:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    return scorer.score(reference, prediction)["rougeL"].fmeasure


def bertscore_f1(
    predictions: list[str],
    references: list[str],
    model_type: str = "distilbert-base-multilingual-cased",
    lang: str = "fr",
) -> list[float]:
    """Modèle multilingue précisé explicitement: les scores BERTScore ne sont
    pas comparables entre modèles différents."""
    from bert_score import score as bert_score_fn

    _, _, f1 = bert_score_fn(predictions, references, model_type=model_type, lang=lang, verbose=False)
    return f1.tolist()


# ---------------------------------------------------------------------------
# Fiabilité — citation, hallucination, fidélité, abstention
# ---------------------------------------------------------------------------
def extract_points_of_interest(text: str, nlp) -> set[str]:
    """Entités nommées (spaCy) + éléments factuels précis (nombres >= 3 chiffres,
    emails, URLs) -- les éléments qu'une réponse fidèle doit reprendre du texte
    de loi sans les inventer. Numéros d'article déjà couverts séparément par
    `citation_metrics`, donc pas dupliqués ici."""
    doc = nlp(text)
    pois = {ent.text.strip().lower() for ent in doc.ents if ent.text.strip()}
    pois |= set(re.findall(r"\b\d{3,}\b", text))
    pois |= set(re.findall(r"\S+@\S+\.\S+", text))
    pois |= set(re.findall(r"https?://\S+", text))
    return pois


def fidelity_score(response: str, relevant_text: str, nlp) -> float:
    """Proportion des passages d'intérêt de la réponse retrouvés dans le texte
    de référence (comparaison par égalité de chaîne, comme les passages sont
    déjà normalisés en minuscules). NaN si la réponse n'a aucun passage
    d'intérêt (réponse purement narrative, sans élément factuel précis)."""
    poi_response = extract_points_of_interest(response, nlp)
    if not poi_response:
        return float("nan")
    poi_reference = extract_points_of_interest(relevant_text, nlp)
    return len(poi_response & poi_reference) / len(poi_response)


def extract_cited_article_numbers(text: str) -> list[str]:
    return re.findall(config.ARTICLE_CITATION_REGEX, text)


def citation_metrics(predicted_text: str, gold_article_refs: list[str]) -> dict:
    """gold_article_refs: liste de numéros d'article de référence (extraits de
    articles.csv['reference'], pas les IDs internes BSARD).
    """
    predicted = set(extract_cited_article_numbers(predicted_text))
    gold = set(gold_article_refs)
    if not predicted and not gold:
        return {"precision": np.nan, "recall": np.nan, "exact_match": 1.0}
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else np.nan
    exact_match = float(predicted == gold)
    return {"precision": precision, "recall": recall, "exact_match": exact_match}


def is_hallucinated_citation(predicted_text: str, valid_article_refs: set[str]) -> bool:
    """Vrai si au moins un article cité n'existe pas dans le corpus indexé."""
    cited = extract_cited_article_numbers(predicted_text)
    return any(c not in valid_article_refs for c in cited)


# Trouve en testant l'arene humaine le 30/07 : le RAG sans fine-tuning (C2)
# recopie souvent le motif "Question : ... Reponse : ..." du prompt au lieu de
# repondre directement (35,6% des reponses, 79/222, contre 0/222 pour les
# trois autres configs) -- le modele de base, jamais entraine sur le format
# de sortie attendu, pattern-matche la structure "Contexte:...\nQuestion:..."
# du prompt RAG et la continue. Le fine-tuning (meme combine au RAG dans C4)
# elimine completement ce comportement -- signal reel sur ce que le
# fine-tuning apporte au-dela du contenu juridique lui-meme.
QUESTION_ECHO_REGEX = re.compile(r"r[ée]ponse\s*:\s*", re.IGNORECASE)


def has_question_echo(text: str) -> bool:
    """Vrai si la reponse recopie un motif Question:/Reponse: du prompt."""
    return bool(re.search(r"question\s*:", text, re.IGNORECASE)) and bool(QUESTION_ECHO_REGEX.search(text))


def strip_question_echo(text: str) -> str:
    """Ne garde que ce qui suit la DERNIERE occurrence de 'Reponse :' -- pour
    l'affichage (arene humaine), afin que l'annotateur juge le contenu
    juridique, pas ce defaut de format. Les metriques automatiques restent
    calculees sur le texte brut (non nettoye), cf. has_question_echo."""
    parts = QUESTION_ECHO_REGEX.split(text)
    return parts[-1].strip() if len(parts) > 1 else text


ABSTENTION_PATTERNS = [
    r"\bje ne sais pas\b",
    r"\bje n'ai pas (?:la|de) r[ée]ponse\b",
    r"\bje ne (?:peux|suis en mesure de) (?:pas )?r[ée]pondre\b",
    r"\bpas assez d'informations\b",
]


def is_abstention(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in ABSTENTION_PATTERNS)


def evaluate_abstention(predictions: list[str], should_abstain: list[bool]) -> dict:
    """should_abstain[i] = True si la question i est hors-domaine / sans réponse dans le corpus."""
    abstained = [is_abstention(p) for p in predictions]
    correct_abstention = sum(
        1 for a, s in zip(abstained, should_abstain) if s and a
    )
    n_should_abstain = sum(should_abstain)
    false_abstention = sum(
        1 for a, s in zip(abstained, should_abstain) if (not s) and a
    )
    n_valid = sum(not s for s in should_abstain)
    return {
        "correct_abstention_rate": correct_abstention / n_should_abstain if n_should_abstain else np.nan,
        "false_abstention_rate": false_abstention / n_valid if n_valid else np.nan,
        "n_should_abstain": n_should_abstain,
        "n_valid": n_valid,
    }
