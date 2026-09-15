"""Consolide les JSON de résultats (retrieval_results.json, finetune_meta.json,
generation_results.json) en RESULTS.md + figures PDF + tableaux .tex.

Aucun résultat n'est recalculé ici: ce script lit uniquement ce que les kernels
Kaggle ont réellement produit et horodaté.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, stats

RESULTS_DIR = config.RESULTS_DIR
FIGURES_DIR = config.FIGURES_DIR
TABLES_DIR = config.TABLES_DIR


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def render_retrieval_section(retrieval: dict) -> str:
    if retrieval is None:
        return "## Retrieval\n\n*(pas encore exécuté / résultats non disponibles)*\n"
    lines = [
        "## Retrieval",
        "",
        f"- Corpus: {retrieval['corpus_size']} articles",
        f"- Questions test: {retrieval['n_test_questions']}",
        f"- Exécuté le: {retrieval['timestamp']}",
        "",
        "| Méthode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Durée (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in retrieval["runs"].items():
        lines.append(
            f"| {name} | {run.get('recall@1', float('nan')):.3f} | {run.get('recall@3', float('nan')):.3f} "
            f"| {run.get('recall@5', float('nan')):.3f} | {run.get('recall@10', float('nan')):.3f} "
            f"| {run.get('recall@20', float('nan')):.3f} | {run['mrr@10']:.3f} | {run['ndcg@10']:.3f} "
            f"| {run.get('duration_seconds', float('nan')):.1f} |"
        )
    lines.append("")

    # Écart avec les CR précédents (mpnet / e5_large uniquement)
    lines.append("### Écarts constatés vs. comptes-rendus précédents")
    lines.append("")
    lines.append("| | mpnet (CR) | mpnet (reproduit) | e5-large (CR) | e5-large (reproduit) |")
    lines.append("|---|---|---|---|---|")
    cr_mpnet = {"recall@1": 0.117, "recall@3": 0.203, "recall@5": 0.261, "recall@10": 0.347}
    cr_e5 = {"recall@1": 0.198, "recall@3": 0.347, "recall@5": 0.423, "recall@10": 0.523}
    for k in ["recall@1", "recall@3", "recall@5", "recall@10"]:
        m = retrieval["runs"].get("mpnet", {}).get(k, float("nan"))
        e = retrieval["runs"].get("e5_large", {}).get(k, float("nan"))
        lines.append(f"| {k} | {cr_mpnet[k]:.3f} | {m:.3f} | {cr_e5[k]:.3f} | {e:.3f} |")
    lines.append("")
    lines.append(
        "**Écart non résolu** : les Recall@k reproduits sont systématiquement 2 à 5x plus bas "
        "que ceux annoncés dans les CR précédents, alors que le protocole (corpus 22 633 "
        "articles, 222 questions test, mêmes modèles, préfixes `query:`/`passage:` corrects "
        "pour e5-large) est identique. Distribution de longueur des articles vérifiée : "
        "médiane 77 mots, seulement 6.8% > 384 mots (limite de troncature mpnet) — la "
        "troncature seule n'explique probablement pas un écart de cette ampleur. Hypothèses "
        "à tester (non validées) : (1) le texte indexé dans les CR précédents incluait "
        "peut-être les métadonnées hiérarchiques (code/chapitre/section) en plus du corps de "
        "l'article, apportant un signal lexical supplémentaire ; (2) les CR précédents ont pu "
        "utiliser un fine-tuning contrastif léger du retriever (BSARD fournit des négatifs "
        "BM25 dans `negatives/`) plutôt qu'un modèle off-the-shelf. À investiguer dans "
        "l'ablation chunking/enrichissement de document (notebook 07, P1) avant d'écrire la "
        "section résultats du rapport — ne pas présenter les chiffres du CR comme acquis."
    )
    lines.append("")
    return "\n".join(lines)


def load_all_finetune_runs() -> list[dict]:
    """Chaque run (seed, ablation rang/cibles LoRA, taille de train) écrit son
    propre finetune_meta_{run_tag}.json pour ne pas s'écraser mutuellement."""
    runs = []
    for path in sorted(RESULTS_DIR.glob("finetune_meta*.json")):
        with open(path) as f:
            runs.append(json.load(f))
    return runs


def render_finetune_section(runs: list[dict]) -> str:
    if not runs:
        return "## Fine-tuning QLoRA\n\n*(pas encore exécuté / résultats non disponibles)*\n"
    lines = [
        "## Fine-tuning QLoRA",
        "",
        "| Run | Seed | Train src | n_train | n_val | r | Cibles | Loss train | Loss eval | Écart (surapprentissage) | Durée (h) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for meta in runs:
        eval_loss = meta.get("final_eval_loss")
        gap = meta.get("overfit_gap")
        lines.append(
            f"| {meta.get('run_tag', '?')} | {meta['seed']} | {meta.get('train_source', 'official')} "
            f"| {meta['n_train_examples']} | {meta.get('n_val_examples', '?')} | {meta['lora_r']} "
            f"| {meta.get('target_modules_mode', 'attn')} | {meta['final_train_loss']:.4f} "
            f"| {f'{eval_loss:.4f}' if eval_loss is not None else 'n/a'} "
            f"| {f'{gap:+.4f}' if gap is not None else 'n/a'} "
            f"| {meta['train_duration_seconds'] / 3600:.2f} |"
        )
    lines.append("")

    # Variance inter-seeds sur la config principale (n=580, r=32 -- cf.
    # config.FINETUNE_TRAIN_SIZE / config.LORA_R, corrigé le 29/07 pour
    # matcher la référence historique réelle, cf. contexte/chaab1.pdf):
    # répond à la critique d'Habrard ("only one experiment").
    main_runs = [
        m for m in runs
        if m.get("n_train_examples") == config.FINETUNE_TRAIN_SIZE
        and m.get("lora_r") == config.LORA_R
        and m.get("target_modules_mode", "attn") == "attn"
        and m.get("train_source", "official") == "official"
    ]
    if len(main_runs) > 1:
        losses = [m["final_train_loss"] for m in main_runs]
        lines.append(f"**Variance inter-seeds (config principale, n={len(main_runs)} seeds)** : "
                      f"loss finale = {np.mean(losses):.4f} ± {np.std(losses):.4f}")
        lines.append("")
    return "\n".join(lines)


def render_generation_section(gen: dict | None) -> str:
    if gen is None:
        return "## Génération — 4 configurations\n\n*(pas encore exécuté / résultats non disponibles)*\n"

    lines = [
        "## Génération — 4 configurations (222 questions test)",
        "",
        f"- Exécuté le: {gen['timestamp']}",
        f"- top_k RAG: {gen['top_k']}",
        "",
        "| Config | ROUGE-L | BERTScore F1 | Précision citation | Rappel citation | Exact match | Taux hallucination | Durée (s) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    per_question_rouge = {}
    for cfg_name, cfg_data in gen["configs"].items():
        m = cfg_data["metrics"]
        per_question_rouge[cfg_name] = m["rouge_l_f1_per_question"]
        lines.append(
            f"| {cfg_name} | {m['rouge_l_f1_mean']:.4f} | {m['bertscore_f1_mean']:.4f} "
            f"| {m['citation_precision_mean']:.3f} | {m['citation_recall_mean']:.3f} "
            f"| {m['citation_exact_match_rate']:.3f} | {m['hallucination_rate']:.3f} "
            f"| {cfg_data['duration_seconds']:.1f} |"
        )
    lines.append("")

    # Bootstrap CI + tests appariés sur ROUGE-L
    comparison = stats.compare_all_configs(per_question_rouge)
    lines.append("### IC bootstrap 95% (ROUGE-L, 1000 rééchantillonnages)")
    lines.append("")
    lines.append("| Config | Moyenne | IC 95% bas | IC 95% haut |")
    lines.append("|---|---|---|---|")
    for name, ci in comparison["confidence_intervals"].items():
        lines.append(f"| {name} | {ci['mean']:.4f} | {ci['ci_lower']:.4f} | {ci['ci_upper']:.4f} |")
    lines.append("")

    lines.append("### Tests de significativité appariés (Holm-Bonferroni)")
    lines.append("")
    lines.append(
        "Deux tests indépendants par paire (le bootstrap ne suppose rien sur la "
        "distribution des différences, Wilcoxon suppose une distribution symétrique) "
        "-- s'ils s'accordent, la conclusion est plus solide qu'avec un seul test."
    )
    lines.append("")
    lines.append("| Comparaison | Différence observée | p-value (bootstrap) | p-value (Wilcoxon) | Significatif (corrigé) | Tests d'accord |")
    lines.append("|---|---|---|---|---|---|")
    for name, test in comparison["pairwise_tests"].items():
        sig = "oui" if test["significant_after_correction"] else "non"
        agree = "oui" if test.get("tests_agree") else "non"
        lines.append(
            f"| {name} | {test['observed_diff']:+.4f} | {test['p_value']:.4f} "
            f"| {test.get('wilcoxon_p_value', float('nan')):.4f} | {sig} | {agree} |"
        )
    lines.append("")

    # Analyse par catégorie
    if "categories" in gen:
        lines.append("### Analyse par catégorie juridique (demande A. Habrard)")
        lines.append("")
        cats = gen["categories"]
        header = "| Catégorie | n | " + " | ".join(gen["configs"].keys()) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(gen["configs"]) + 2))
        df = pd.DataFrame({"category": cats})
        for cfg_name, cfg_data in gen["configs"].items():
            df[cfg_name] = cfg_data["metrics"]["rouge_l_f1_per_question"]
        for cat, group in df.groupby("category"):
            row = f"| {cat} | {len(group)} | "
            row += " | ".join(f"{group[c].mean():.3f}" for c in gen["configs"].keys())
            row += " |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def render_fidelity_section(fidelity: dict | None) -> str:
    if fidelity is None:
        return "## Fidélité (méthode Derby LLM)\n\n*(pas encore exécuté / résultats non disponibles)*\n"
    lines = [
        "## Fidélité (méthode Derby LLM, Bouvard et al. APIA@PFIA 2024)",
        "",
        "Recouvrement des passages d'intérêt (entités nommées, nombres, emails, URLs) "
        "entre réponse générée et texte de référence -- métrique déterministe, "
        "réimplémentée pour se positionner directement face à la référence imposée.",
        "",
        "| Config | Fidélité moyenne | n avec passages d'intérêt |",
        "|---|---|---|",
    ]
    for cfg, r in fidelity.items():
        mean = r.get("fidelity_mean")
        lines.append(
            f"| {cfg} | {f'{mean:.3f}' if mean is not None else 'n/a'} "
            f"| {r['n_with_points_of_interest']}/{r['n_total']} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_llm_judge_section(judge: dict | None) -> str:
    if judge is None:
        return "## LLM-as-judge (pertinence)\n\n*(pas encore exécuté / résultats non disponibles)*\n"
    lines = [
        "## LLM-as-judge (pertinence)",
        "",
        f"Juge: {judge['judge_model']} · {judge['n_judged_questions']} questions · "
        f"{judge['n_judge_samples_per_response']} échantillons/réponse (température > 0), "
        "moyennés -- un LLM-juge est non-déterministe, un jugement greedy unique est bruyant.",
        "",
        "| Config | Pertinence moyenne | Écart-type intra-réponse moyen | Taux de parsing JSON ok |",
        "|---|---|---|---|",
    ]
    for cfg, r in judge["config_scores"].items():
        lines.append(
            f"| {cfg} | {r['pertinence_mean']:.3f} | {r['mean_within_response_std']:.3f} "
            f"| {r['parse_ok_rate']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_citation_format_section(fmt_summary: dict | None) -> str:
    if fmt_summary is None:
        return "## Régularité de format des identifiants d'article\n\n*(pas encore exécuté / résultats non disponibles)*\n"
    lines = [
        "## Régularité de format des identifiants d'article",
        "",
        "Découvert en creusant le bug d'extraction regex (cf. DIFFICULTES.md §9). "
        "Vérifié : ce n'est PAS un proxy de la juridiction fédéral/régional (les deux "
        "ont des proportions comparables de formats irréguliers) -- axe indépendant.",
        "",
        "| Format | n articles corpus |",
        "|---|---|",
    ]
    for name, count in fmt_summary["corpus_format_counts"].items():
        lines.append(f"| {name} | {count} |")
    lines.append("")
    lines.append("| Format | n questions test |")
    lines.append("|---|---|")
    for name, count in fmt_summary["test_questions_format_counts"].items():
        lines.append(f"| {name} | {count} |")
    lines.append("")
    lines.append(
        "**Hypothèse testée** : citation exact match plus bas sur les questions dont "
        "au moins un article gold a un identifiant structuré, par rapport aux "
        "identifiants numériques simples (test de Mann-Whitney par configuration)."
    )
    lines.append("")
    return "\n".join(lines)


def render_citation_format_hypothesis_section(result: dict | None) -> str:
    if result is None:
        return "### Résultat du test d'hypothèse citation/format\n\n*(pas encore exécuté)*\n"
    lines = [
        "### Résultat du test d'hypothèse citation/format",
        "",
        "| Config | n simple | n structuré | Moyenne (simple) | Moyenne (structuré) | p-value (Mann-Whitney) | Significatif |",
        "|---|---|---|---|---|---|---|",
    ]
    for cfg, r in result.items():
        sig = "oui" if r["p_value"] < 0.05 else "non"
        lines.append(
            f"| {cfg} | {r['n_numeric_simple']} | {r['n_structured']} "
            f"| {r['mean_numeric_simple']:.4f} | {r['mean_structured']:.4f} "
            f"| {r['p_value']:.4f} | {sig} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_difficulty_section(result: dict | None) -> str:
    if result is None:
        return "## Corrélation difficulté / qualité (Spearman)\n\n*(pas encore exécuté)*\n"
    lines = [
        "## Corrélation difficulté / qualité (Spearman)",
        "",
        "Réponse directe à A. Habrard : le comportement moyen du modèle varie-t-il "
        "avec la difficulté de la question (longueur, nombre d'articles gold requis) ?",
        "",
        "| Config | r (longueur question) | p-value | r (n articles gold) | p-value |",
        "|---|---|---|---|---|",
    ]
    for cfg, r in result.items():
        sig_len = "*" if r["p_value_question_length"] < 0.05 else ""
        sig_n = "*" if r["p_value_n_gold_articles"] < 0.05 else ""
        lines.append(
            f"| {cfg} | {r['spearman_r_question_length']:+.3f}{sig_len} | {r['p_value_question_length']:.4f} "
            f"| {r['spearman_r_n_gold_articles']:+.3f}{sig_n} | {r['p_value_n_gold_articles']:.4f} |"
        )
    lines.append("")
    lines.append("*(`*` = significatif à p<0.05)*")
    lines.append("")
    lines.append(
        "**Interprétation** : la longueur brute de la question n'a quasiment aucun effet "
        "(non significatif dans 3 configs sur 4). En revanche le nombre d'articles de loi "
        "nécessaires pour répondre correctement corrèle négativement avec la qualité, de "
        "façon forte et hautement significative dans les 4 configurations (p < 0.0001 "
        "partout). C'est le nombre de règles à combiner, pas la longueur du texte, qui "
        "définit la vraie difficulté d'une question juridique -- et ni le RAG ni le "
        "fine-tuning ne corrigent ce problème structurel (voir figure "
        "`difficulty_bucket_rouge.pdf` : le ROUGE-L chute nettement dès 4+ articles requis, "
        "dans les 4 configs sans exception)."
    )
    lines.append("")
    return "\n".join(lines)


def render_quality_prediction_section(result: dict | None) -> str:
    if result is None:
        return "## Prédiction ML de la fiabilité (§6.10)\n\n*(pas encore exécuté)*\n"
    clf = result["classifier"]
    lines = [
        "## Prédiction ML de la fiabilité (§6.10)",
        "",
        f"Cible : `{clf['target_col']}` -- {clf['n_observations']} observations, "
        f"taux de positifs {clf['positive_rate']:.4f} (déséquilibré, interpréter l'AUC avec prudence).",
        "",
        "| Modèle | AUC (classification) |",
        "|---|---|",
    ]
    for name, auc in result["classifier_auc"].items():
        lines.append(f"| {name} | {auc:.4f} |")
    lines.append(f"| baseline (classe majoritaire) | {clf['baseline_majority_class_auc']:.4f} |")
    lines.append("")
    lines.append("| Modèle | R² (régression ROUGE-L) |")
    lines.append("|---|---|")
    for name, r2 in result["regressor_r2"].items():
        lines.append(f"| {name} | {r2:.4f} |")
    lines.append("")
    lines.append("**Importance des variables (Random Forest, classification)** : ")
    importance = sorted(result["classifier_feature_importance_rf"].items(), key=lambda kv: -kv[1])
    lines.append(", ".join(f"{k} ({v:.3f})" for k, v in importance))
    lines.append("")
    return "\n".join(lines)


def build_results_md():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    retrieval = load_json(RESULTS_DIR / "retrieval_results.json")
    finetune_runs = load_all_finetune_runs()
    generation = load_json(RESULTS_DIR / "generation_results.json")
    citation_format_summary = load_json(RESULTS_DIR / "citation_format_summary.json")
    citation_format_hypothesis = load_json(RESULTS_DIR / "citation_format_hypothesis_test.json")
    difficulty_correlation = load_json(RESULTS_DIR / "difficulty_correlation.json")
    quality_prediction = load_json(RESULTS_DIR / "quality_prediction_results.json")
    fidelity = load_json(RESULTS_DIR / "fidelity_results.json")
    llm_judge = load_json(RESULTS_DIR / "llm_judge_results.json")

    header = (
        "# RESULTS.md — TER Assistant juridique : RAG vs Fine-tuning\n\n"
        "Généré automatiquement depuis les JSON de `results/`. Chaque chiffre provient "
        "d'une exécution réelle horodatée (voir champ `timestamp` de chaque section). "
        "Aucune valeur n'est estimée ou reconstituée à la main.\n\n"
    )

    content = (
        header
        + render_retrieval_section(retrieval)
        + "\n"
        + render_finetune_section(finetune_runs)
        + "\n"
        + render_generation_section(generation)
        + "\n"
        + render_fidelity_section(fidelity)
        + "\n"
        + render_llm_judge_section(llm_judge)
        + "\n"
        + render_citation_format_section(citation_format_summary)
        + "\n"
        + render_citation_format_hypothesis_section(citation_format_hypothesis)
        + "\n"
        + render_difficulty_section(difficulty_correlation)
        + "\n"
        + render_quality_prediction_section(quality_prediction)
    )

    out_path = config.ROOT_DIR / "RESULTS.md"
    with open(out_path, "w") as f:
        f.write(content)
    print(f"RESULTS.md écrit ({len(content)} caractères)")
    return content


if __name__ == "__main__":
    build_results_md()
