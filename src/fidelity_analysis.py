"""Calcule la fidélité (au sens Derby LLM, cf. metrics.py::fidelity_score) pour
les 4 configurations sur les 222 questions test. Aucun GPU requis (spaCy
fr_core_news_sm tourne sur CPU) -- peut s'exécuter en local ou sur RunPod.

Comme pour le juge LLM (cf. llm_judge_job.py), la fidélité est mesurée par
rapport au texte des articles réellement pertinents (reference_answers), pas
par rapport au contexte effectivement récupéré par le RAG -- même limitation
méthodologique assumée, documentée une seule fois dans DIFFICULTES.md.
"""
from __future__ import annotations

import json
import subprocess
import sys

from . import config, metrics


def _ensure_spacy_model():
    import importlib

    try:
        importlib.import_module("spacy")
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "spacy"], check=True)
    import spacy

    try:
        return spacy.load("fr_core_news_sm")
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "fr_core_news_sm"], check=True)
        return spacy.load("fr_core_news_sm")


def compute_fidelity_per_config(generation_results: dict) -> dict:
    nlp = _ensure_spacy_model()
    reference_answers = generation_results["reference_answers"]
    out = {}
    for cfg_name, cfg_data in generation_results["configs"].items():
        scores = [
            metrics.fidelity_score(ans, ref, nlp)
            for ans, ref in zip(cfg_data["answers"], reference_answers)
        ]
        import numpy as np

        valid = [s for s in scores if not np.isnan(s)]
        out[cfg_name] = {
            "fidelity_per_question": scores,
            "fidelity_mean": float(np.mean(valid)) if valid else None,
            "n_with_points_of_interest": len(valid),
            "n_total": len(scores),
        }
    return out


if __name__ == "__main__":
    gen_path = config.RESULTS_DIR / "generation_results.json"
    if not gen_path.exists():
        print(f"{gen_path} absent -- la fidélité sera calculable dès que la génération sera terminée.")
    else:
        with open(gen_path) as f:
            gen = json.load(f)
        result = compute_fidelity_per_config(gen)
        for cfg, r in result.items():
            print(f"{cfg}: fidelity_mean={r['fidelity_mean']} (n={r['n_with_points_of_interest']}/{r['n_total']})")
        out_path = config.RESULTS_DIR / "fidelity_results.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Écrit: {out_path}")
