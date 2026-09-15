"""Analyse d'erreurs — taxonomie à 9 catégories (brief §4.7), appliquée sur
>= 50 réponses par configuration, annotées par les deux étudiants avec mesure
d'accord (Cohen's Kappa).

Ce module ne fait PAS d'annotation automatique par IA : il génère un CSV
d'annotation manuelle à partir des vraies réponses générées, puis agrège les
annotations une fois remplies. Aucune catégorie d'erreur n'est jamais inventée
ou pré-remplie par le pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

ERROR_TAXONOMY = {
    1: "hallucination_article",  # numéro d'article inexistant
    2: "article_non_pertinent",  # article existant mais non pertinent
    3: "confusion_systeme_juridique",  # confusion droit FR/BE/autre
    4: "reponse_sans_citation",  # correcte mais sans citation
    5: "absence_abstention",  # aurait dû s'abstenir, ne l'a pas fait
    6: "abstention_excessive",  # s'est abstenu alors qu'une réponse existait
    7: "erreur_de_langue",  # réponse dans une autre langue que le français
    8: "reponse_incomplete_ou_recopie",  # incomplète ou recopie brute sans reformulation
    9: "degenerescence_repetition",  # sur-apprentissage, répétitions
    0: "aucune_erreur",  # réponse correcte, rien à signaler
}

N_SAMPLES_PER_CONFIG = 50


def build_annotation_template(generation_results_path: Path, seed: int = config.SEED) -> pd.DataFrame:
    """Échantillonne >= 50 réponses par config et produit un CSV vide à annoter
    manuellement (colonnes error_code_annotator1, error_code_annotator2)."""
    with open(generation_results_path) as f:
        gen = json.load(f)

    questions = gen["questions"]
    categories = gen.get("categories", [""] * len(questions))
    rng = np.random.default_rng(seed)
    n_total = len(questions)
    sample_idx = sorted(rng.choice(n_total, size=min(N_SAMPLES_PER_CONFIG, n_total), replace=False).tolist())

    rows = []
    for cfg_name, cfg_data in gen["configs"].items():
        answers = cfg_data["answers"]
        for i in sample_idx:
            rows.append(
                {
                    "config": cfg_name,
                    "question_index": i,
                    "question": questions[i],
                    "category": categories[i] if i < len(categories) else "",
                    "answer": answers[i],
                    "reference_answer": gen["reference_answers"][i],
                    "error_code_annotator1": "",
                    "error_code_annotator2": "",
                    "notes": "",
                }
            )

    df = pd.DataFrame(rows)
    taxonomy_note = " | ".join(f"{k}={v}" for k, v in ERROR_TAXONOMY.items())
    print(f"Taxonomie (à utiliser pour error_code_annotatorN): {taxonomy_note}")
    return df


def compute_agreement(df: pd.DataFrame) -> dict:
    """Cohen's Kappa entre les deux annotateurs, une fois error_code_annotator1/2 remplis."""
    from sklearn.metrics import cohen_kappa_score

    mask = (df["error_code_annotator1"] != "") & (df["error_code_annotator2"] != "")
    if mask.sum() == 0:
        return {"error": "Aucune ligne annotée par les deux annotateurs"}
    a1 = df.loc[mask, "error_code_annotator1"].astype(int)
    a2 = df.loc[mask, "error_code_annotator2"].astype(int)
    kappa = cohen_kappa_score(a1, a2)
    exact_agreement = float((a1 == a2).mean())
    return {"cohen_kappa": float(kappa), "exact_agreement_rate": exact_agreement, "n_annotated": int(mask.sum())}


def build_error_matrix(df: pd.DataFrame, annotator_col: str = "error_code_annotator1") -> pd.DataFrame:
    """Matrice type d'erreur x configuration (comptes), une fois annoté."""
    df = df[df[annotator_col] != ""].copy()
    df["error_type"] = df[annotator_col].astype(int).map(ERROR_TAXONOMY)
    matrix = pd.crosstab(df["error_type"], df["config"])
    return matrix


def select_qualitative_examples(df: pd.DataFrame, annotator_col: str = "error_code_annotator1", n: int = 6) -> pd.DataFrame:
    """Sélectionne des exemples qualitatifs commentés (1-2 par type d'erreur non triviale)."""
    df = df[df[annotator_col] != ""].copy()
    df["error_type"] = df[annotator_col].astype(int).map(ERROR_TAXONOMY)
    non_trivial = df[df["error_type"] != "aucune_erreur"]
    examples = non_trivial.groupby("error_type").head(1).head(n)
    return examples[["config", "question", "answer", "error_type", "notes"]]


if __name__ == "__main__":
    gen_path = config.RESULTS_DIR / "generation_results.json"
    if not gen_path.exists():
        print(f"{gen_path} absent — le template d'annotation sera généré dès que la génération sera terminée.")
    else:
        df = build_annotation_template(gen_path)
        out = config.RESULTS_DIR / "error_annotation_template.csv"
        df.to_csv(out, index=False)
        print(f"Template d'annotation écrit: {out} ({len(df)} lignes, {df['config'].nunique()} configs)")
