"""Analyse de la régularité de format des identifiants d'article — découverte
en creusant le bug d'extraction regex (§9 DIFFICULTES.md), pas dans le brief
initial.

Hypothèse testée (pas juste une statistique descriptive) : les configurations
citent-elles moins bien (exact match plus bas, hallucination plus haute) les
articles dont l'identifiant a un format irrégulier ("N1.1", "L1122-9", "D382")
que ceux au format numérique simple ("959") ? Mécanisme plausible : (1) le
regex d'extraction de citation sur le texte généré peut mal capturer des
identifiants avec ponctuation inhabituelle, (2) le fine-tuning n'a vu que peu
d'exemples de ce format sur 580-886 exemples d'entraînement.

Point vérifié avant d'écrire ce script (ne pas supposer) : ce n'est PAS un
proxy de la juridiction. Le fédéral (5552 structurés / 8615 simples) et le
régional (3735 structurés / 4731 simples) ont des proportions comparables de
formats irréguliers -- la juridiction et la régularité de format sont deux
axes indépendants, pas redondants.
"""
from __future__ import annotations

import re

import pandas as pd

from . import config, data

NUMERIC_SIMPLE_REGEX = re.compile(r"\d+[a-zA-Z]?")


def _extract_clean_id(reference: str) -> str:
    m = re.search(config.ARTICLE_REFERENCE_REGEX, str(reference))
    if not m:
        return ""
    return config.clean_article_ref_id(m.group(1).strip())


def _format_label(clean_id: str) -> str:
    if not clean_id:
        return "inconnu"
    return "numeric_simple" if NUMERIC_SIMPLE_REGEX.fullmatch(clean_id) else "structured"


def build_article_format_table() -> pd.DataFrame:
    articles = data.load_articles()
    articles = articles.copy()
    articles["clean_id"] = articles["reference"].apply(_extract_clean_id)
    articles["id_format"] = articles["clean_id"].apply(_format_label)
    return articles


def build_question_format_table() -> pd.DataFrame:
    """Pour chaque question test: le format de SES articles gold. Une question
    est 'structured' si au moins un de ses articles gold a un ID irrégulier
    (le cas le plus contraignant pour la citation)."""
    articles_fmt = build_article_format_table()
    fmt_lookup = articles_fmt.set_index("id")["id_format"].to_dict()

    test_df = data.load_questions_test().copy()
    test_df["has_structured_gold"] = test_df["article_ids"].apply(
        lambda ids: any(fmt_lookup.get(aid) == "structured" for aid in ids)
    )
    test_df["question_format"] = test_df["has_structured_gold"].map(
        {True: "structured", False: "numeric_simple"}
    )
    return test_df


def summarize(article_fmt_df: pd.DataFrame, question_fmt_df: pd.DataFrame) -> dict:
    return {
        "corpus_format_counts": article_fmt_df["id_format"].value_counts().to_dict(),
        "corpus_format_by_law_type": (
            article_fmt_df.groupby(["law_type", "id_format"]).size().unstack(fill_value=0).to_dict(orient="index")
        ),
        "test_questions_format_counts": question_fmt_df["question_format"].value_counts().to_dict(),
    }


def join_with_generation_results(generation_results: dict) -> pd.DataFrame:
    """Une fois generation_results.json disponible: teste l'hypothèse citation
    exact match / hallucination plus faibles sur les identifiants structurés."""
    df = build_question_format_table().reset_index(drop=True)
    for cfg_name, cfg_data in generation_results["configs"].items():
        m = cfg_data["metrics"]
        if "citation_exact_match_per_question" in m:
            df[f"{cfg_name}_citation_exact"] = m["citation_exact_match_per_question"]
        if "hallucination_per_question" in m:
            df[f"{cfg_name}_hallucination"] = m["hallucination_per_question"]
    return df


def format_hypothesis_test(df_joined: pd.DataFrame, config_names: list[str]) -> dict:
    """Compare numeric_simple vs structured par config, avec test de
    Mann-Whitney (échantillons indépendants, pas de paires). Répond à
    l'hypothèse mécaniste, pas juste une moyenne côte à côte."""
    from scipy.stats import mannwhitneyu

    report = {}
    for cfg in config_names:
        col = f"{cfg}_citation_exact"
        if col not in df_joined:
            continue
        simple = df_joined.loc[df_joined["question_format"] == "numeric_simple", col].dropna()
        structured = df_joined.loc[df_joined["question_format"] == "structured", col].dropna()
        if len(simple) < 5 or len(structured) < 5:
            continue
        stat, p = mannwhitneyu(simple, structured, alternative="two-sided")
        report[cfg] = {
            "n_numeric_simple": len(simple),
            "n_structured": len(structured),
            "mean_numeric_simple": float(simple.mean()),
            "mean_structured": float(structured.mean()),
            "mannwhitney_u": float(stat),
            "p_value": float(p),
        }
    return report


if __name__ == "__main__":
    import json

    article_fmt = build_article_format_table()
    question_fmt = build_question_format_table()
    summary = summarize(article_fmt, question_fmt)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    question_fmt.to_csv(config.RESULTS_DIR / "citation_format_table.csv", index=False)
    with open(config.RESULTS_DIR / "citation_format_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nÉcrit: {config.RESULTS_DIR / 'citation_format_table.csv'} et citation_format_summary.json")
