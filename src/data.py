"""Chargement et nettoyage de BSARD.

Points d'attention documentés dans config.py:
  - article_ids est stocké en string "12024" ou "947,948" -> toujours parser
  - pas de réponse rédigée gold -> reconstruite par concaténation des articles cités
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from . import config

HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{config.HF_DATASET_BSARD}/resolve/main"

_FILES = {
    "articles.csv": config.BSARD_ARTICLES_CSV,
    "questions_train.csv": config.BSARD_QUESTIONS_TRAIN_CSV,
    "questions_test.csv": config.BSARD_QUESTIONS_TEST_CSV,
    "questions_synthetic.csv": config.BSARD_QUESTIONS_SYNTHETIC_CSV,
}


def download_bsard(force: bool = False) -> None:
    """Télécharge les CSV BSARD depuis HuggingFace (dataset libre, non gated)."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for remote_name, local_path in _FILES.items():
        if local_path.exists() and not force:
            continue
        url = f"{HF_RESOLVE_BASE}/{remote_name}"
        urllib.request.urlretrieve(url, local_path)


def parse_article_ids(raw: str) -> list[int]:
    """article_ids est un string comma-separated ('947,948'), pas une liste."""
    if pd.isna(raw):
        return []
    return [int(x) for x in str(raw).split(",") if x.strip()]


def load_articles() -> pd.DataFrame:
    df = pd.read_csv(config.BSARD_ARTICLES_CSV)
    assert len(df) == config.EXPECTED_CORPUS_SIZE, (
        f"Corpus BSARD attendu à {config.EXPECTED_CORPUS_SIZE} articles, obtenu {len(df)}. "
        "Le dataset source a peut-être changé depuis l'audit du 2026-07-28."
    )
    return df


def _load_questions(path: Path, expected_size: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["article_ids"] = df["article_ids"].apply(parse_article_ids)
    if expected_size is not None:
        assert len(df) == expected_size, (
            f"{path.name} attendu à {expected_size} lignes, obtenu {len(df)}."
        )
    return df


def load_questions_train() -> pd.DataFrame:
    return _load_questions(config.BSARD_QUESTIONS_TRAIN_CSV, config.EXPECTED_TRAIN_SIZE)


def load_questions_test() -> pd.DataFrame:
    return _load_questions(config.BSARD_QUESTIONS_TEST_CSV, config.EXPECTED_TEST_SIZE)


def load_questions_synthetic() -> pd.DataFrame:
    return _load_questions(config.BSARD_QUESTIONS_SYNTHETIC_CSV)


def build_article_lookup(articles_df: pd.DataFrame) -> dict[int, dict]:
    """id -> {reference, article (texte), law_type, description, ...}"""
    return articles_df.set_index("id").to_dict(orient="index")


def build_reference_answer(article_ids: list[int], article_lookup: dict[int, dict]) -> str:
    """Concatène le texte des articles cités comme 'réponse de référence'.

    BSARD n'a pas de réponse rédigée gold: c'est une limite méthodologique
    assumée et documentée (cf. config.py, DIFFICULTES.md), pas une réponse
    idéale. Utilisée uniquement pour ROUGE-L / BERTScore, PAS pour l'éval
    de citation (qui compare des IDs, pas du texte).
    """
    parts = []
    for aid in article_ids:
        art = article_lookup.get(aid)
        if art is not None:
            parts.append(f"{art['reference']}: {art['article']}")
    return "\n".join(parts)


def build_category_index(questions_df: pd.DataFrame) -> dict[str, list[int]]:
    """category -> liste des index de lignes (pour l'analyse par sous-groupe demandée par Habrard)."""
    return questions_df.groupby("category").groups  # type: ignore[return-value]


def load_all() -> dict[str, pd.DataFrame]:
    download_bsard()
    return {
        "articles": load_articles(),
        "train": load_questions_train(),
        "test": load_questions_test(),
        "synthetic": load_questions_synthetic(),
    }


if __name__ == "__main__":
    data = load_all()
    for name, df in data.items():
        print(f"{name}: {df.shape}")
    print("Catégories (test):")
    print(data["test"]["category"].value_counts())
