"""Génère les figures PDF vectorielles à partir des JSON de results/."""
from __future__ import annotations

import json

import matplotlib.pyplot as plt

from . import config

RESULTS_DIR = config.RESULTS_DIR
FIGURES_DIR = config.FIGURES_DIR


def fig_recall_at_k():
    path = RESULTS_DIR / "retrieval_results.json"
    if not path.exists():
        print("retrieval_results.json absent, figure ignorée")
        return
    with open(path) as f:
        retrieval = json.load(f)

    ks = config.RECALL_KS
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, run in retrieval["runs"].items():
        ys = [run.get(f"recall@{k}", float("nan")) for k in ks]
        ax.plot(ks, ys, marker="o", label=name)
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title(f"Recall@k par méthode de retrieval ({retrieval['n_test_questions']} questions test)")
    ax.legend()
    ax.grid(alpha=0.3)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "recall_at_k_comparison.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure écrite: {out}")


def fig_category_distribution():
    from . import data

    test_df = data.load_questions_test()
    counts = test_df["category"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(counts.index, counts.values, color="#4C72B0")
    ax.set_xlabel("Nombre de questions (split test, n=222)")
    ax.set_title("Distribution des catégories juridiques — BSARD test")
    for i, v in enumerate(counts.values):
        ax.text(v + 0.5, i, str(v), va="center")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "bsard_category_distribution.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure écrite: {out}")


def fig_config_comparison_ci(metric_key: str = "rouge_l_f1_per_question", metric_label: str = "ROUGE-L"):
    """Barres des 4 configurations avec IC bootstrap 95% (brief §7)."""
    path = RESULTS_DIR / "generation_results.json"
    if not path.exists():
        print("generation_results.json absent, figure ignorée")
        return
    with open(path) as f:
        gen = json.load(f)

    from . import stats

    per_question = {cfg: data["metrics"][metric_key] for cfg, data in gen["configs"].items() if metric_key in data["metrics"]}
    if not per_question:
        print(f"{metric_key} absent des configs, figure ignorée")
        return
    comparison = stats.compare_all_configs(per_question)
    cis = comparison["confidence_intervals"]

    names = list(cis.keys())
    means = [cis[n]["mean"] for n in names]
    lo = [cis[n]["mean"] - cis[n]["ci_lower"] for n in names]
    hi = [cis[n]["ci_upper"] - cis[n]["mean"] for n in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, means, yerr=[lo, hi], capsize=5, color="#4C72B0")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} par configuration (IC bootstrap 95%, 222 questions)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "config_comparison_ci.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure écrite: {out}")


def fig_heatmap_category():
    """Heatmap configuration x catégorie juridique (réponse à Habrard)."""
    path = RESULTS_DIR / "generation_results.json"
    if not path.exists():
        print("generation_results.json absent, figure ignorée")
        return
    with open(path) as f:
        gen = json.load(f)
    if "categories" not in gen:
        print("categories absent de generation_results.json, figure ignorée")
        return

    import pandas as pd

    df = pd.DataFrame({"category": gen["categories"]})
    cfg_names = list(gen["configs"].keys())
    for cfg in cfg_names:
        df[cfg] = gen["configs"][cfg]["metrics"]["rouge_l_f1_per_question"]
    pivot = df.groupby("category")[cfg_names].mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(cfg_names)), cfg_names, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("ROUGE-L moyen par configuration × catégorie juridique")
    fig.colorbar(im, label="ROUGE-L")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "heatmap_category.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure écrite: {out}")


def fig_learning_curve():
    """Courbe d'apprentissage: loss finale (train + validation) en fonction de
    la taille du train, sur tous les runs de fine-tuning disponibles à
    rang/cibles fixes (isole l'effet de la taille du train)."""
    from . import consolidate_results

    runs = consolidate_results.load_all_finetune_runs()
    main_runs = [
        r for r in runs
        if r.get("lora_r") == config.LORA_R and r.get("target_modules_mode", "attn") == "attn"
        and r.get("train_source", "official") == "official"
    ]
    if len(main_runs) < 2:
        print(f"Seulement {len(main_runs)} run(s) a r={config.LORA_R}, courbe d'apprentissage ignorée (besoin >= 2)")
        return
    main_runs = sorted(main_runs, key=lambda r: r["n_train_examples"])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        [r["n_train_examples"] for r in main_runs],
        [r["final_train_loss"] for r in main_runs],
        marker="o", label="loss train (finale)",
    )
    eval_runs = [r for r in main_runs if r.get("final_eval_loss") is not None]
    if eval_runs:
        ax.plot(
            [r["n_train_examples"] for r in eval_runs],
            [r["final_eval_loss"] for r in eval_runs],
            marker="s", label="loss validation (finale)",
        )
    ax.set_xlabel("Nombre d'exemples d'entraînement")
    ax.set_ylabel("Loss")
    ax.set_title(f"Courbe d'apprentissage QLoRA (r={config.LORA_R}, attention seule)")
    ax.legend()
    ax.grid(alpha=0.3)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "learning_curve.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure écrite: {out}")


def fig_error_matrix():
    """Matrice type d'erreur x configuration, depuis le CSV annoté manuellement."""
    path = RESULTS_DIR / "error_annotation_template.csv"
    if not path.exists():
        print("error_annotation_template.csv absent, figure ignorée")
        return
    import pandas as pd

    from . import error_analysis

    df = pd.read_csv(path)
    if (df["error_code_annotator1"] == "").all():
        print("error_annotation_template.csv pas encore annoté, figure ignorée")
        return
    matrix = error_analysis.build_error_matrix(df)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix.values, cmap="OrRd", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix.values[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Matrice type d'erreur × configuration")
    fig.colorbar(im, label="Nombre de réponses")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "error_matrix.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure écrite: {out}")


def fig_bradley_terry_ranking():
    """Classement de l'arène humaine par modèle Bradley-Terry (méthode
    Chatbot Arena/LMSYS)."""
    from . import arena_app

    result = arena_app.compute_bradley_terry_ranking()
    if "error" in result:
        print(f"{result['error']}, figure ignorée")
        return
    ranking = result["ranking"]

    fig, ax = plt.subplots(figsize=(6, 4))
    names = [r["config"] for r in ranking]
    scores = [r["score_elo_like"] for r in ranking]
    ax.barh(names, scores, color="#C44E52")
    ax.set_xlabel("Score (échelle Elo-like, Bradley-Terry)")
    ax.set_title(f"Classement arène — Bradley-Terry ({result['n_votes_used']} votes)")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "bradley_terry_ranking.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure écrite: {out}")


def fig_inter_annotator_agreement(config_a: str, config_b: str, annotators: list[str]):
    """Matrice de Kappa de Cohen par paire d'annotateurs (arène)."""
    from . import arena_app

    agreement = arena_app.compute_inter_annotator_agreement(config_a, config_b, annotators)
    if not agreement:
        print("Aucun vote d'arène commun entre annotateurs, figure ignorée")
        return

    import numpy as np

    n = len(annotators)
    matrix = np.full((n, n), np.nan)
    for i in range(n):
        matrix[i, i] = 1.0
    for key, vals in agreement.items():
        a1, a2 = key.split("_vs_")
        i, j = annotators.index(a1), annotators.index(a2)
        matrix[i, j] = matrix[j, i] = vals["cohen_kappa"]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n), annotators)
    ax.set_yticks(range(n), annotators)
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center")
    ax.set_title(f"Accord inter-annotateurs (Kappa), {config_a} vs {config_b}")
    fig.colorbar(im)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "inter_annotator_agreement.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure écrite: {out}")


def fig_agreement_distribution(config_a: str, config_b: str, annotators: list[str]):
    """Camembert de la distribution d'accord entre annotateurs (unanime /
    majoritaire / aucun accord) -- façon Derby LLM Figure 7. Jamais tracé
    jusqu'ici alors que compute_full_agreement_distribution existe déjà."""
    from . import arena_app

    dist = arena_app.compute_full_agreement_distribution(config_a, config_b, annotators)
    if "error" in dist:
        print(f"{dist['error']}, figure ignorée")
        return

    labels = ["Accord unanime", "Accord majoritaire", "Aucun accord"]
    values = [dist["pct_unanimous"], dist["pct_majority_only"], dist["pct_no_agreement"]]
    colors = ["#55A868", "#DD8452", "#C44E52"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(values, labels=[f"{l}\n({v:.1f}%)" for l, v in zip(labels, values)], colors=colors, startangle=90)
    ax.set_title(f"Accord inter-annotateurs ({dist['n_questions']} questions, {len(annotators)} annotateurs)")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "agreement_distribution.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure écrite: {out}")


def fig_difficulty_bucket():
    """Barres groupées: ROUGE-L moyen par palier de difficulté (nombre d'articles
    gold requis), pour les 4 configs -- montre que ni le RAG ni le fine-tuning
    ne compensent la chute de qualité sur les questions necessitant 4+ articles,
    alors que la longueur de la question seule n'a quasiment pas d'effet
    (cf. correlation_report dans difficulty_analysis.py)."""
    path = RESULTS_DIR / "generation_results.json"
    if not path.exists():
        print("generation_results.json absent, figure ignorée")
        return
    with open(path) as f:
        gen = json.load(f)

    from . import difficulty_analysis

    df = difficulty_analysis.join_with_generation_results(gen)
    cfg_names = list(gen["configs"].keys())
    order = ["1 article", "2-3 articles", "4+ articles"]
    counts = df["difficulty_bucket"].value_counts().reindex(order)
    grouped = df.groupby("difficulty_bucket")[[f"{cfg}_rouge_l" for cfg in cfg_names]].mean().reindex(order)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = range(len(order))
    width = 0.8 / len(cfg_names)
    colors = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"]
    for i, cfg in enumerate(cfg_names):
        vals = grouped[f"{cfg}_rouge_l"].values
        ax.bar([xi + i * width for xi in x], vals, width=width, label=cfg, color=colors[i % len(colors)])
    ax.set_xticks([xi + width * (len(cfg_names) - 1) / 2 for xi in x])
    ax.set_xticklabels([f"{b}\n(n={int(counts[b])})" for b in order])
    ax.set_ylabel("ROUGE-L moyen")
    ax.set_title("ROUGE-L selon le nombre d'articles nécessaires pour répondre")
    ax.legend(fontsize=8)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "difficulty_bucket_rouge.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure écrite: {out}")


def fig_llm_judge_pertinence_distribution():
    """Barres empilées: distribution des notes de pertinence (1-5) données par
    le LLM-juge (Phi-3.5), par configuration -- montre que la pertinence
    moyenne reste haute partout mais qu'une minorité de reponses clairement
    hors-sujet grossit progressivement de C1 vers C4."""
    path = RESULTS_DIR / "llm_judge_results.json"
    if not path.exists():
        print("llm_judge_results.json absent, figure ignorée")
        return
    with open(path) as f:
        judge = json.load(f)

    cfg_names = list(judge["config_scores"].keys())
    scores = range(1, 6)
    pct_by_cfg = {}
    for cfg in cfg_names:
        samples = [
            s["pertinence"]
            for q in judge["config_scores"][cfg]["per_question"]
            for s in q["samples"]
            if s["pertinence"] is not None
        ]
        n = len(samples)
        pct_by_cfg[cfg] = [100 * sum(1 for x in samples if x == s) / n for s in scores]

    fig, ax = plt.subplots(figsize=(8, 4))
    left = [0.0] * len(cfg_names)
    colors = {1: "#e34948", 2: "#e34948", 3: "#eda100", 4: "#eb6834", 5: "#1baf7a"}
    for s in scores:
        vals = [pct_by_cfg[cfg][s - 1] for cfg in cfg_names]
        ax.barh(cfg_names, vals, left=left, color=colors[s], label=f"Note {s}/5")
        left = [l + v for l, v in zip(left, vals)]
    ax.set_xlabel("% des échantillons jugés")
    ax.set_xlim(0, 100)
    ax.set_title("Distribution des notes de pertinence (LLM-juge Phi-3.5) par configuration")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=8)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "llm_judge_pertinence_distribution.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure écrite: {out}")


def fig_error_type_distribution():
    """Camembert de la répartition globale des types d'erreur (toutes configs
    confondues) -- vue synthétique complémentaire à la matrice type×config."""
    path = RESULTS_DIR / "error_annotation_template.csv"
    if not path.exists():
        print("error_annotation_template.csv absent, figure ignorée")
        return
    import pandas as pd

    from . import error_analysis

    df = pd.read_csv(path)
    if (df["error_code_annotator1"] == "").all():
        print("error_annotation_template.csv pas encore annoté, figure ignorée")
        return
    df = df[df["error_code_annotator1"] != ""].copy()
    df["error_type"] = df["error_code_annotator1"].astype(int).map(error_analysis.ERROR_TAXONOMY)
    counts = df["error_type"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(counts.values, labels=[f"{l}\n({v})" for l, v in zip(counts.index, counts.values)], startangle=90)
    ax.set_title("Répartition globale des types d'erreur (toutes configs)")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "error_type_distribution.pdf"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Figure écrite: {out}")


if __name__ == "__main__":
    fig_recall_at_k()
    fig_category_distribution()
    fig_config_comparison_ci()
    fig_heatmap_category()
    fig_learning_curve()
    fig_error_matrix()
    fig_bradley_terry_ranking()
    fig_difficulty_bucket()
    fig_llm_judge_pertinence_distribution()
    fig_error_type_distribution()
