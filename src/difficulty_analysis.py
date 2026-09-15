"""Analyse par difficulté (longueur de question, nombre d'articles pertinents,
fréquence de la thématique dans le train) — réponse directe à la remarque
d'A. Habrard sur la diversité du comportement moyen du modèle.

Ne dépend d'aucun résultat GPU: calculable directement sur les 222 questions
test réelles de BSARD.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data


def build_difficulty_table() -> pd.DataFrame:
    test_df = data.load_questions_test()
    train_df = data.load_questions_train()

    category_train_freq = train_df["category"].value_counts(normalize=True).to_dict()

    df = test_df.copy()
    df["question_length_words"] = df["question"].apply(lambda q: len(str(q).split()))
    df["n_gold_articles"] = df["article_ids"].apply(len)
    df["category_train_frequency"] = df["category"].map(category_train_freq)

    # Bucket de difficulté simple: nb d'articles pertinents (1 vs 2-3 vs 4+)
    def bucket(n):
        if n == 1:
            return "1 article"
        elif n <= 3:
            return "2-3 articles"
        else:
            return "4+ articles"

    df["difficulty_bucket"] = df["n_gold_articles"].apply(bucket)
    return df


def summarize(df: pd.DataFrame) -> dict:
    return {
        "question_length_words": {
            "mean": float(df["question_length_words"].mean()),
            "median": float(df["question_length_words"].median()),
            "std": float(df["question_length_words"].std()),
            "min": int(df["question_length_words"].min()),
            "max": int(df["question_length_words"].max()),
        },
        "n_gold_articles": {
            "mean": float(df["n_gold_articles"].mean()),
            "median": float(df["n_gold_articles"].median()),
            "std": float(df["n_gold_articles"].std()),
        },
        "difficulty_bucket_counts": df["difficulty_bucket"].value_counts().to_dict(),
        "category_train_frequency": df.groupby("category")["category_train_frequency"].first().to_dict(),
    }


def join_with_generation_results(generation_results: dict) -> pd.DataFrame:
    """Une fois les résultats de génération disponibles, fusionne avec la table
    de difficulté pour calculer la corrélation qualité <-> difficulté par config."""
    df = build_difficulty_table().reset_index(drop=True)
    for cfg_name, cfg_data in generation_results["configs"].items():
        df[f"{cfg_name}_rouge_l"] = cfg_data["metrics"]["rouge_l_f1_per_question"]
        df[f"{cfg_name}_bertscore"] = cfg_data["metrics"]["bertscore_f1_per_question"]
    return df


def correlation_report(df_joined: pd.DataFrame, config_names: list[str]) -> dict:
    """Corrélation de Spearman entre difficulté (longueur question, nb articles gold)
    et qualité de génération (ROUGE-L), par configuration."""
    from scipy.stats import spearmanr

    report = {}
    for cfg in config_names:
        col = f"{cfg}_rouge_l"
        if col not in df_joined:
            continue
        r_len, p_len = spearmanr(df_joined["question_length_words"], df_joined[col])
        r_narts, p_narts = spearmanr(df_joined["n_gold_articles"], df_joined[col])
        report[cfg] = {
            "spearman_r_question_length": float(r_len),
            "p_value_question_length": float(p_len),
            "spearman_r_n_gold_articles": float(r_narts),
            "p_value_n_gold_articles": float(p_narts),
        }
    return report


if __name__ == "__main__":
    df = build_difficulty_table()
    summary = summarize(df)
    import json

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESULTS_DIR / "difficulty_table.csv", index=False)
    with open(config.RESULTS_DIR / "difficulty_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nÉcrit: {config.RESULTS_DIR / 'difficulty_table.csv'} et difficulty_summary.json")
