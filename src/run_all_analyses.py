"""Point d'entrée unique pour TOUTES les analyses CPU post-génération.

Avant ce script, chaque analyse (corrélation difficulté, hypothèse
citation/format, prédiction ML, fidélité) était relancée à la main via des
commandes ponctuelles à chaque nouvelle génération -- fragile, et
exactement pourquoi certains résultats n'apparaissaient jamais dans
RESULTS.md malgré avoir été calculés une fois. Ce script fait tout, dans
l'ordre, à partir du seul generation_results.json, pour qu'aucune analyse
ne soit plus jamais oubliée.

Usage: python -m src.run_all_analyses
"""
from __future__ import annotations

import json

from . import citation_format_analysis, config, consolidate_results, difficulty_analysis, make_figures, make_tables


def run_all():
    gen_path = config.RESULTS_DIR / "generation_results.json"
    if not gen_path.exists():
        print(f"{gen_path} absent -- rien à analyser pour l'instant.")
        return
    with open(gen_path) as f:
        gen = json.load(f)
    cfg_names = list(gen["configs"].keys())

    print("=== 1/6 Régularité de format d'identifiant (corpus, indépendant de la génération) ===")
    article_fmt = citation_format_analysis.build_article_format_table()
    question_fmt = citation_format_analysis.build_question_format_table()
    summary = citation_format_analysis.summarize(article_fmt, question_fmt)
    with open(config.RESULTS_DIR / "citation_format_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== 2/6 Hypothèse citation/format (Mann-Whitney) ===")
    df_fmt = citation_format_analysis.join_with_generation_results(gen)
    result_fmt = citation_format_analysis.format_hypothesis_test(df_fmt, cfg_names)
    with open(config.RESULTS_DIR / "citation_format_hypothesis_test.json", "w") as f:
        json.dump(result_fmt, f, indent=2, ensure_ascii=False)
    print(json.dumps(result_fmt, indent=2, ensure_ascii=False))

    print("\n=== 3/6 Corrélation difficulté/qualité (Spearman) ===")
    df_diff = difficulty_analysis.join_with_generation_results(gen)
    result_diff = difficulty_analysis.correlation_report(df_diff, cfg_names)
    with open(config.RESULTS_DIR / "difficulty_correlation.json", "w") as f:
        json.dump(result_diff, f, indent=2, ensure_ascii=False)
    print(json.dumps(result_diff, indent=2, ensure_ascii=False))

    print("\n=== 4/6 Prédiction ML de fiabilité ===")
    try:
        from . import quality_prediction

        df_qp = quality_prediction.build_feature_matrix(gen)
        clf_results = quality_prediction.train_reliability_classifier(df_qp, "citation_exact_match")
        reg_results = quality_prediction.train_quality_regressor(df_qp, "rouge_l")
        quality_prediction.make_figures(clf_results, reg_results)
        qp_summary = {
            "classifier": {k: v for k, v in clf_results.items() if k not in ("logistic_regression", "random_forest")},
            "classifier_auc": {name: clf_results[name]["auc"] for name in ("logistic_regression", "random_forest")},
            "classifier_feature_importance_rf": clf_results["random_forest"]["feature_importance"],
            "regressor_r2": {name: reg_results[name]["r2"] for name in ("linear_regression", "random_forest")},
        }
        with open(config.RESULTS_DIR / "quality_prediction_results.json", "w") as f:
            json.dump(qp_summary, f, indent=2, ensure_ascii=False)
        print(json.dumps(qp_summary, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Prédiction ML ignorée (erreur : {e})")

    print("\n=== 5/6 Fidélité (spaCy, façon Derby LLM) ===")
    try:
        from . import fidelity_analysis

        fid_result = fidelity_analysis.compute_fidelity_per_config(gen)
        with open(config.RESULTS_DIR / "fidelity_results.json", "w") as f:
            json.dump(fid_result, f, indent=2, ensure_ascii=False)
        for cfg, r in fid_result.items():
            print(f"{cfg}: fidelity_mean={r['fidelity_mean']}")
    except Exception as e:
        print(f"Fidélité ignorée (spaCy probablement absent : {e})")

    print("\n=== 6/6 Consolidation : RESULTS.md, figures, tableaux ===")
    consolidate_results.build_results_md()
    make_figures.fig_recall_at_k()
    make_figures.fig_category_distribution()
    make_figures.fig_config_comparison_ci()
    make_figures.fig_heatmap_category()
    make_figures.fig_learning_curve()
    make_figures.fig_error_matrix()
    make_figures.fig_bradley_terry_ranking()
    make_figures.fig_difficulty_bucket()
    make_figures.fig_llm_judge_pertinence_distribution()
    make_figures.fig_error_type_distribution()
    make_tables.table_retrieval()
    make_tables.table_config_description()
    make_tables.table_citation_hallucination()
    make_tables.table_abstention()
    make_tables.table_cost_comparison()

    print("\n=== TERMINE ===")


if __name__ == "__main__":
    run_all()
