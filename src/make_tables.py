"""Génère les tableaux .tex (booktabs) à partir des JSON de results/."""
from __future__ import annotations

import json

from . import config

RESULTS_DIR = config.RESULTS_DIR
TABLES_DIR = config.TABLES_DIR


def table_retrieval():
    path = RESULTS_DIR / "retrieval_results.json"
    if not path.exists():
        print("retrieval_results.json absent, tableau ignoré")
        return
    with open(path) as f:
        retrieval = json.load(f)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparaison des méthodes de retrieval sur les " + str(retrieval["n_test_questions"]) + r" questions test de BSARD.}",
        r"\label{tab:retrieval-comparison}",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Méthode & Recall@1 & Recall@3 & Recall@5 & Recall@10 & Recall@20 & MRR@10 & nDCG@10 \\",
        r"\midrule",
    ]
    name_map = {
        "bm25": "BM25",
        "mpnet": "mpnet (baseline)",
        "e5_large": "multilingual-e5-large",
        "hybrid_bm25_e5": "Hybride (RRF)",
    }
    for key, run in retrieval["runs"].items():
        name = name_map.get(key, key)
        lines.append(
            f"{name} & {run.get('recall@1', float('nan')):.3f} & {run.get('recall@3', float('nan')):.3f} "
            f"& {run.get('recall@5', float('nan')):.3f} & {run.get('recall@10', float('nan')):.3f} "
            f"& {run.get('recall@20', float('nan')):.3f} & {run['mrr@10']:.3f} & {run['ndcg@10']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "retrieval_comparison.tex"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Tableau écrit: {out}")


def table_config_description():
    """Description statique des 4 configurations (brief §6.1) -- ne dépend
    d'aucun résultat, peut être générée à tout moment."""
    rows = [
        ("C1", "Mistral-7B-Instruct zero-shot", "Non", "Non"),
        ("C2", "RAG (retrieval e5-large + reranker)", "Oui", "Non"),
        ("C3", "Fine-tuning QLoRA seul", "Non", "Oui"),
        ("C4", "Fine-tuning QLoRA + RAG", "Oui", "Oui"),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Description des quatre configurations comparées.}",
        r"\label{tab:config-description}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Config & Description & Retrieval & Fine-tuning \\",
        r"\midrule",
    ]
    for cfg, desc, rag, ft in rows:
        lines.append(f"{cfg} & {desc} & {rag} & {ft} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "config_description.tex"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Tableau écrit: {out}")


def table_citation_hallucination():
    path = RESULTS_DIR / "generation_results.json"
    if not path.exists():
        print("generation_results.json absent, tableau ignoré")
        return
    with open(path) as f:
        gen = json.load(f)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Fiabilité des réponses : exactitude de citation et hallucination (222 questions test).}",
        r"\label{tab:citation-hallucination}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Config & Précision citation & Rappel citation & Exact match & Taux hallucination \\",
        r"\midrule",
    ]
    for cfg_name, cfg_data in gen["configs"].items():
        m = cfg_data["metrics"]
        lines.append(
            f"{cfg_name} & {m.get('citation_precision_mean', float('nan')):.3f} "
            f"& {m.get('citation_recall_mean', float('nan')):.3f} "
            f"& {m.get('citation_exact_match_rate', float('nan')):.3f} "
            f"& {m.get('hallucination_rate', float('nan')):.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "citation_hallucination.tex"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Tableau écrit: {out}")


def table_abstention():
    path = RESULTS_DIR / "abstention_results.json"
    if not path.exists():
        print("abstention_results.json absent, tableau ignoré")
        return
    with open(path) as f:
        abst = json.load(f)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Capacité d'abstention (" + str(abst["n_questions"]) + r" questions hors-domaine/sans réponse).}",
        r"\label{tab:abstention}",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Config & Taux d'abstention correcte \\",
        r"\midrule",
    ]
    for cfg_name, cfg_data in abst["configs"].items():
        lines.append(f"{cfg_name} & {cfg_data['correct_abstention_rate']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "abstention.tex"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Tableau écrit: {out}")


def table_cost_comparison():
    """Coût pratique (brief §4.2/§7): temps d'entraînement, temps d'inférence,
    taille des artefacts -- dimension que Derby LLM valorise aussi."""
    from . import consolidate_results

    finetune_runs = consolidate_results.load_all_finetune_runs()
    gen_path = RESULTS_DIR / "generation_results.json"
    gen = None
    if gen_path.exists():
        with open(gen_path) as f:
            gen = json.load(f)

    main_ft = next(
        (r for r in finetune_runs if r.get("lora_r") == config.LORA_R and r.get("n_train_examples") == config.FINETUNE_TRAIN_SIZE),
        None,
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Coût pratique par configuration.}",
        r"\label{tab:cost-comparison}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Config & Durée génération (s, 222 q.) & Notes \\",
        r"\midrule",
    ]
    if gen is not None:
        for cfg_name, cfg_data in gen["configs"].items():
            lines.append(f"{cfg_name} & {cfg_data.get('duration_seconds', float('nan')):.1f} & \\\\")
    if main_ft is not None:
        lines.append(
            f"Fine-tuning (entraînement) & {main_ft['train_duration_seconds'] / 3600:.2f} h & "
            f"{main_ft['pct_trainable']:.3f}\\% des paramètres, adaptateur \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / "cost_comparison.tex"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Tableau écrit: {out}")


if __name__ == "__main__":
    table_retrieval()
    table_config_description()
    table_citation_hallucination()
    table_abstention()
    table_cost_comparison()
