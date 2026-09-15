"""Calcule les analyses de l'arène humaine (accord inter-annotateurs, classement
Bradley-Terry, préférence par catégorie, distribution d'accord) à partir des
votes CSV dans results/arena_votes/, et sauvegarde le résultat consolidé.

Nécessite que les votes des 3 annotateurs existent déjà (produits par
`python -m src.arena_app --pair C2_rag:C3_finetune --annotator <prenom>`,
un par personne). Ne nécessite aucun GPU.

Usage: python -m src.compute_arena_results
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from . import arena_app, config

CONFIG_A = "C2_rag"
CONFIG_B = "C3_finetune"
ANNOTATORS = ["Bartosz", "Chaabane", "Arman"]


def compute_all() -> dict:
    return {
        "config_a": CONFIG_A,
        "config_b": CONFIG_B,
        "annotators": ANNOTATORS,
        "n_questions_per_annotator": arena_app.N_ARENA_QUESTIONS,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Doublons dédupliqués par question_index (garde le dernier vote) via "
            "arena_app._load_votes -- cf. bug découvert dans "
            "arena_C2_rag_vs_C3_finetune_Bartosz.csv (4 questions votées deux fois, "
            "probablement après un rechargement de page en cours de session)."
        ),
        "inter_annotator_agreement": arena_app.compute_inter_annotator_agreement(
            CONFIG_A, CONFIG_B, ANNOTATORS
        ),
        "bradley_terry_ranking": arena_app.compute_bradley_terry_ranking(
            [(CONFIG_A, CONFIG_B)]
        ),
        "preference_by_category": arena_app.compute_preference_by_category(
            CONFIG_A, CONFIG_B, ANNOTATORS
        ),
        "full_agreement_distribution": arena_app.compute_full_agreement_distribution(
            CONFIG_A, CONFIG_B, ANNOTATORS
        ),
    }


if __name__ == "__main__":
    results = compute_all()
    out_path = config.RESULTS_DIR / "arena_human_evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Écrit : {out_path}")
    print(json.dumps(results, indent=2, ensure_ascii=False))
