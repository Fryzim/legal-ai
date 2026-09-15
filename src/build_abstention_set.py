"""Construit le jeu de 50 questions pour l'évaluation de la capacité d'abstention
(cf. brief §4.2). Trois catégories, comme demandé:
  1. questions non-juridiques (hors domaine complet)
  2. questions de droit d'un autre pays que la Belgique (le corpus BSARD est belge)
  3. questions dont l'article de référence est volontairement retiré de l'index

Sortie: data/abstention_questions.csv (id, question, abstention_type, should_abstain,
excluded_article_id [uniquement pour le type 3]).
"""
import csv
from pathlib import Path

from . import config, data

OUTPUT_PATH = config.DATA_DIR / "abstention_questions.csv"

# --- Catégorie 1: questions non-juridiques (20) ---
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

# --- Catégorie 2: droit étranger, hors du champ du corpus belge (15) ---
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

# --- Catégorie 3: questions valides du corpus BSARD, mais dont l'article de
# référence sera retiré de l'index au moment de l'éval (construit dynamiquement
# ci-dessous à partir du split test réel).
N_EXCLUDED_ARTICLE_QUESTIONS = 15


def build():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    data.download_bsard()
    test_df = data.load_questions_test()

    rows = []
    qid = 1
    for q in NON_LEGAL_QUESTIONS:
        rows.append(
            {
                "id": qid,
                "question": q,
                "abstention_type": "non_juridique",
                "should_abstain": True,
                "excluded_article_id": "",
            }
        )
        qid += 1

    for q in FOREIGN_LAW_QUESTIONS:
        rows.append(
            {
                "id": qid,
                "question": q,
                "abstention_type": "droit_etranger",
                "should_abstain": True,
                "excluded_article_id": "",
            }
        )
        qid += 1

    # Échantillon reproductible de questions BSARD réelles (1 seul article
    # cité, pour un retrait net et non ambigu de l'index).
    single_article_df = test_df[test_df["article_ids"].apply(len) == 1].sample(
        n=N_EXCLUDED_ARTICLE_QUESTIONS, random_state=config.SEED
    )
    for _, row in single_article_df.iterrows():
        rows.append(
            {
                "id": qid,
                "question": row["question"],
                "abstention_type": "article_retire_index",
                "should_abstain": True,
                "excluded_article_id": row["article_ids"][0],
            }
        )
        qid += 1

    assert len(rows) == len(NON_LEGAL_QUESTIONS) + len(FOREIGN_LAW_QUESTIONS) + N_EXCLUDED_ARTICLE_QUESTIONS

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "question", "abstention_type", "should_abstain", "excluded_article_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} questions écrites dans {OUTPUT_PATH}")
    return rows


if __name__ == "__main__":
    build()
