"""Arène d'évaluation humaine (brief §4.5), interface Gradio locale.

Réplique le protocole Derby LLM: question + contexte + 2 réponses anonymisées
et présentées dans un ordre aléatoire, 4 boutons (A meilleure / B meilleure /
Match nul / Aucune). Comparaisons prioritaires : C2 (RAG) vs C3 (fine-tune
seul) -- le duel central de la question de recherche -- puis C4 vs C2.

Nécessite generation_results.json (produit par le kernel de génération). Ne
nécessite AUCUN GPU: tourne en local sur les réponses déjà générées.

Affiche aussi le contexte de référence (texte de loi réellement pertinent)
à côté des deux réponses, pour que l'annotateur puisse juger si une réponse
s'écarte des faits, pas seulement si elle "sonne bien" (façon Derby LLM
Fig. 4). Le prénom de l'annotateur se saisit dans l'interface elle-même
(un seul lien partageable pour Chaabane et le 3e annotateur, pas besoin de
relancer le script avec un argument différent pour chacun).

Usage: python -m src.arena_app --pair C2_rag:C3_finetune [--annotator bartosz]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from . import config, metrics

N_ARENA_QUESTIONS = 60
PRIORITY_PAIRS = [
    ("C2_rag", "C3_finetune"),  # duel central de la question de recherche
    ("C4_finetune_rag", "C2_rag"),
]


def load_generation_results() -> dict:
    path = config.RESULTS_DIR / "generation_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} absent. L'arène nécessite les réponses générées par le kernel de génération."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _water_fill_allocation(sizes: dict[str, int], total: int) -> dict[str, int]:
    """Répartition équilibrée par catégorie (pas proportionnelle) : chaque
    catégorie reçoit une part égale du budget restant, plafonnée par sa
    taille réelle -- les catégories les plus petites (ex. Protection sociale,
    4 questions sur 222) donnent tout ce qu'elles ont plutôt que de recevoir
    une part proportionnelle qui les rendrait inanalysables (1 question sur 60).
    """
    remaining_cats = sorted(sizes.keys(), key=lambda c: sizes[c])
    allocation = {}
    budget = total
    n_remaining = len(remaining_cats)
    for c in remaining_cats:
        share = budget / n_remaining
        take = min(sizes[c], round(share))
        allocation[c] = take
        budget -= take
        n_remaining -= 1
    leftover = total - sum(allocation.values())
    if leftover != 0:
        largest = max(allocation, key=lambda c: sizes[c])
        allocation[largest] += leftover
    return allocation


def build_stratified_sample(categories: list[str], n_total: int = N_ARENA_QUESTIONS, seed: int = config.SEED) -> list[int]:
    """Tirage stratifié par catégorie juridique : répartition équilibrée
    (cf. _water_fill_allocation), pas un simple tirage aléatoire uniforme qui
    reproduirait le déséquilibre du corpus (Famille/Logement sur-représentées,
    Protection sociale quasi absente). Permet ensuite d'analyser par catégorie
    quelles configurations sont préférées humainement, pas seulement le total.
    """
    rng = random.Random(seed)
    by_cat: dict[str, list[int]] = {}
    for i, c in enumerate(categories):
        by_cat.setdefault(c, []).append(i)
    for c in by_cat:
        rng.shuffle(by_cat[c])

    sizes = {c: len(idx) for c, idx in by_cat.items()}
    allocation = _water_fill_allocation(sizes, min(n_total, len(categories)))

    sample = []
    for c, n in allocation.items():
        sample.extend(by_cat[c][:n])
    return sorted(sample)


MAX_CHARS_PER_ARTICLE = 220
MAX_ARTICLES_SHOWN = 3


def _shorten_reference_context(raw: str) -> str:
    """Le contexte de référence (concaténation du texte des articles gold,
    cf. data.py::build_reference_answer) peut faire plusieurs milliers de
    caractères sur les questions à articles longs ou multiples -- illisible
    dans l'arène. Raccourci par article (pas une coupure globale aveugle, qui
    perdrait la référence des articles suivants) : garde la référence de
    chaque article (ex. "Art. 7, Code Wallon...") en entier, tronque son
    texte a MAX_CHARS_PER_ARTICLE, et limite a MAX_ARTICLES_SHOWN articles."""
    lines = [l for l in raw.split("\n") if l.strip()]
    shortened = []
    for line in lines[:MAX_ARTICLES_SHOWN]:
        if ":" in line:
            ref, _, text = line.partition(":")
            text = text.strip()
            if len(text) > MAX_CHARS_PER_ARTICLE:
                text = text[:MAX_CHARS_PER_ARTICLE].rsplit(" ", 1)[0] + " (...)"
            shortened.append(f"{ref.strip()} : {text}")
        else:
            shortened.append(line[:MAX_CHARS_PER_ARTICLE])
    if len(lines) > MAX_ARTICLES_SHOWN:
        shortened.append(f"(+ {len(lines) - MAX_ARTICLES_SHOWN} autre(s) article(s) de référence non affiché(s))")
    return "\n\n".join(shortened)


def build_arena_items(gen: dict, config_a: str, config_b: str, seed: int = config.SEED) -> list[dict]:
    questions = gen["questions"]
    categories = gen.get("categories", [""] * len(questions))
    rng = random.Random(seed)
    sample_idx = build_stratified_sample(categories, N_ARENA_QUESTIONS, seed)

    answers_a = gen["configs"][config_a]["answers"]
    answers_b = gen["configs"][config_b]["answers"]

    reference_answers = gen.get("reference_answers", [""] * len(questions))

    items = []
    for i in sample_idx:
        # Randomise l'ordre d'affichage (anonymisation gauche/droite)
        swap = rng.random() < 0.5
        left, right = (answers_a[i], answers_b[i]) if not swap else (answers_b[i], answers_a[i])
        left_cfg, right_cfg = (config_a, config_b) if not swap else (config_b, config_a)
        items.append(
            {
                "question_index": i,
                "question": questions[i],
                "category": categories[i],
                # Texte des articles réellement pertinents (référence) -- affiché à
                # l'annotateur pour qu'il puisse juger si une réponse s'écarte des
                # faits, pas seulement si elle "sonne bien" (façon Derby LLM Fig. 4,
                # qui affiche le contexte à côté des deux réponses à comparer).
                "reference_context": _shorten_reference_context(reference_answers[i]),
                # Nettoye pour l'affichage uniquement (motif "Question:...Reponse:..."
                # recopie par C2 sans fine-tuning, cf. metrics.has_question_echo) --
                # le texte brut reste dans generation_results.json pour les metriques.
                "left_answer": metrics.strip_question_echo(left),
                "right_answer": metrics.strip_question_echo(right),
                "left_config": left_cfg,  # non affiché à l'annotateur, utilisé pour désanonymiser après export
                "right_config": right_cfg,
            }
        )
    return items


def run_app(config_a: str, config_b: str, annotator: str | None = None, share: bool = False):
    import gradio as gr

    gen = load_generation_results()
    items = build_arena_items(gen, config_a, config_b)

    out_dir = config.RESULTS_DIR / "arena_votes"
    out_dir.mkdir(parents=True, exist_ok=True)

    def render(idx):
        if idx >= len(items):
            return (
                "Terminé — merci ! Toutes les comparaisons ont été votées.",
                "", "", "",
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(interactive=False), gr.update(interactive=False),
            )
        item = items[idx]
        progress = f"Question {idx + 1}/{len(items)} (catégorie : {item['category']})"
        q_text = f"**{progress}**\n\n**Question :** {item['question']}"
        ref_text = item["reference_context"] or "(aucun texte de référence disponible pour cette question)"
        return (
            q_text, ref_text, item["left_answer"], item["right_answer"],
            gr.update(interactive=True), gr.update(interactive=True),
            gr.update(interactive=True), gr.update(interactive=True),
        )

    def start(prenom, sess):
        prenom = (prenom or "").strip()
        if not prenom:
            return (
                sess, gr.update(), gr.update(visible=True),
                "⚠️ Merci d'indiquer ton prénom avant de commencer.",
                "", "", "",
                gr.update(interactive=False), gr.update(interactive=False),
                gr.update(interactive=False), gr.update(interactive=False),
            )
        sess = {"idx": 0, "annotator": prenom}
        q_text, ref_text, left, right, *btns = render(sess["idx"])
        return (sess, gr.update(visible=False), gr.update(visible=True), q_text, ref_text, left, right, *btns)

    def vote(choice, sess):
        # sess est propre a CHAQUE session de navigateur (gr.State), pas un dict
        # global partage -- sinon plusieurs annotateurs connectes en meme temps
        # sur le meme lien ecraseraient/melangeraient la progression des autres.
        if not sess or sess.get("annotator") is None:
            return sess, *render(0)
        idx = sess["idx"]
        if idx >= len(items):
            return sess, *render(idx)
        item = items[idx]
        row = {
            "annotator": sess["annotator"],
            "question_index": item["question_index"],
            "category": item["category"],
            "config_a": config_a,
            "config_b": config_b,
            "left_config": item["left_config"],
            "right_config": item["right_config"],
            "choice": choice,  # "left", "right", "tie", "neither"
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        out_path = out_dir / f"arena_{config_a}_vs_{config_b}_{sess['annotator']}.csv"
        # Sauvegarde incrémentale (résistant à une fermeture accidentelle du navigateur)
        write_header = not out_path.exists()
        with open(out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        sess["idx"] += 1
        return sess, *render(sess["idx"])

    with gr.Blocks(title="Arène TER — évaluation humaine") as demo:
        session_state = gr.State(value={"idx": 0, "annotator": None})
        gr.Markdown(f"## Arène : {config_a} vs {config_b}")
        with gr.Row(visible=True) as login_row:
            prenom_box = gr.Textbox(label="Ton prénom", placeholder="ex: Bartosz", value=annotator or "")
            start_btn = gr.Button("Commencer")
        with gr.Column(visible=False) as arena_col:
            question_md = gr.Markdown()
            ref_box = gr.Textbox(label="Contexte (texte de loi réellement pertinent -- pour juger si une réponse s'écarte des faits)", lines=6, interactive=False)
            with gr.Row():
                left_box = gr.Textbox(label="Réponse A", lines=10, interactive=False)
                right_box = gr.Textbox(label="Réponse B", lines=10, interactive=False)
            with gr.Row():
                btn_a = gr.Button("A est meilleure")
                btn_b = gr.Button("B est meilleure")
                btn_tie = gr.Button("Match nul")
                btn_neither = gr.Button("Aucune (les deux mauvaises)")

        start_btn.click(
            start, inputs=[prenom_box, session_state],
            outputs=[session_state, login_row, arena_col, question_md, ref_box, left_box, right_box, btn_a, btn_b, btn_tie, btn_neither],
        )
        btn_a.click(lambda s: vote("left", s), inputs=[session_state], outputs=[session_state, question_md, ref_box, left_box, right_box, btn_a, btn_b, btn_tie, btn_neither])
        btn_b.click(lambda s: vote("right", s), inputs=[session_state], outputs=[session_state, question_md, ref_box, left_box, right_box, btn_a, btn_b, btn_tie, btn_neither])
        btn_tie.click(lambda s: vote("tie", s), inputs=[session_state], outputs=[session_state, question_md, ref_box, left_box, right_box, btn_a, btn_b, btn_tie, btn_neither])
        btn_neither.click(lambda s: vote("neither", s), inputs=[session_state], outputs=[session_state, question_md, ref_box, left_box, right_box, btn_a, btn_b, btn_tie, btn_neither])

    demo.launch(share=share)


def _load_votes(path: Path):
    """Charge un CSV de votes en dédupliquant par question_index (garde le
    dernier). La sauvegarde incrémentale (cf. vote()) ajoute une ligne à
    chaque clic sans vérifier si la question a déjà été votée -- un
    rechargement de page en cours de session peut donc laisser plusieurs
    votes pour la même question ; seul le plus récent (dernier ajouté) est
    considéré valide."""
    import pandas as pd

    df = pd.read_csv(path)
    return df.drop_duplicates(subset="question_index", keep="last")


def compute_inter_annotator_agreement(config_a: str, config_b: str, annotators: list[str]) -> dict:
    """Cohen's Kappa par paire d'annotateurs sur les votes de l'arène."""
    from sklearn.metrics import cohen_kappa_score

    out_dir = config.RESULTS_DIR / "arena_votes"
    dfs = {}
    for ann in annotators:
        path = out_dir / f"arena_{config_a}_vs_{config_b}_{ann}.csv"
        if path.exists():
            dfs[ann] = _load_votes(path).set_index("question_index")["choice"]

    results = {}
    from itertools import combinations

    for a1, a2 in combinations(dfs.keys(), 2):
        common = dfs[a1].index.intersection(dfs[a2].index)
        if len(common) == 0:
            continue
        kappa = cohen_kappa_score(dfs[a1].loc[common], dfs[a2].loc[common])
        agreement = float((dfs[a1].loc[common] == dfs[a2].loc[common]).mean())
        results[f"{a1}_vs_{a2}"] = {"cohen_kappa": float(kappa), "exact_agreement": agreement, "n_common": len(common)}
    return results


def compute_bradley_terry_ranking(pairs: list[tuple[str, str]] = PRIORITY_PAIRS) -> dict:
    """Classement des configurations à partir de TOUS les votes d'arène
    disponibles (toutes paires, tous annotateurs), par modèle de Bradley-Terry
    -- la méthode utilisée par Chatbot Arena/LMSYS (dont Derby LLM et nous nous
    inspirons tous deux pour le protocole d'arène), réimplémentée ici via une
    estimation par maximum de vraisemblance (pas copiée).

    P(i bat j) = exp(s_i) / (exp(s_i) + exp(s_j)). Donne un score de force
    relative sur une échelle unique, plus interprétable qu'un taux de victoire
    par paire isolée quand plusieurs configurations sont en jeu. Un match nul
    compte pour 0.5 victoire de chaque côté ; "aucune" (neither) est exclu
    (ne renseigne pas sur la force relative des deux réponses).
    """
    import numpy as np
    import pandas as pd
    from scipy.optimize import minimize

    out_dir = config.RESULTS_DIR / "arena_votes"
    configs_involved = sorted({c for pair in pairs for c in pair})
    idx = {c: i for i, c in enumerate(configs_involved)}
    n = len(configs_involved)
    wins = np.zeros((n, n))  # wins[i, j] = nb de victoires (ponderees) de i sur j

    n_votes_used = 0
    for config_a, config_b in pairs:
        for path in out_dir.glob(f"arena_{config_a}_vs_{config_b}_*.csv"):
            df = _load_votes(path)
            for _, row in df.iterrows():
                if row["choice"] == "neither":
                    continue
                winner_cfg = None
                if row["choice"] == "tie":
                    wins[idx[row["left_config"]], idx[row["right_config"]]] += 0.5
                    wins[idx[row["right_config"]], idx[row["left_config"]]] += 0.5
                elif row["choice"] == "left":
                    winner_cfg = row["left_config"]
                elif row["choice"] == "right":
                    winner_cfg = row["right_config"]
                if winner_cfg is not None:
                    loser_cfg = row["right_config"] if winner_cfg == row["left_config"] else row["left_config"]
                    wins[idx[winner_cfg], idx[loser_cfg]] += 1.0
                n_votes_used += 1

    if n_votes_used == 0:
        return {"error": "Aucun vote d'arène trouvé dans results/arena_votes/"}

    def neg_log_likelihood(strengths):
        s = np.concatenate([[0.0], strengths])  # ancre la premiere config a 0 (identifiabilite)
        ll = 0.0
        for i in range(n):
            for j in range(n):
                if wins[i, j] > 0:
                    ll += wins[i, j] * (s[i] - np.logaddexp(s[i], s[j]))
        return -ll

    x0 = np.zeros(n - 1)
    result = minimize(neg_log_likelihood, x0, method="BFGS")
    strengths = np.concatenate([[0.0], result.x])
    # Normalise pour lisibilite (comme un score Elo, echelle arbitraire)
    strengths_display = 1000 + 400 * (strengths - strengths.mean())

    ranking = sorted(
        [{"config": c, "strength_bt": float(strengths[idx[c]]), "score_elo_like": float(strengths_display[idx[c]])} for c in configs_involved],
        key=lambda r: -r["strength_bt"],
    )
    return {
        "ranking": ranking,
        "n_votes_used": n_votes_used,
        "converged": bool(result.success),
        "note": "config non presente dans une comparaison d'arene = absente du classement (ex: C1 si jamais oppose)",
    }


def compute_preference_by_category(config_a: str, config_b: str, annotators: list[str]) -> dict:
    """Pour chaque catégorie juridique, taux de victoire de config_a / config_b
    / match nul, tous annotateurs confondus. Répond directement à l'objectif du
    tirage stratifié : voir humainement quelles catégories semblent le mieux
    générées par quelle configuration, pas seulement le taux global."""
    import pandas as pd

    out_dir = config.RESULTS_DIR / "arena_votes"
    rows = []
    for ann in annotators:
        path = out_dir / f"arena_{config_a}_vs_{config_b}_{ann}.csv"
        if path.exists():
            rows.append(_load_votes(path))
    if not rows:
        return {"error": f"aucun vote trouve pour {config_a} vs {config_b}"}
    df = pd.concat(rows, ignore_index=True)

    def winner(row):
        if row["choice"] == "tie":
            return "tie"
        if row["choice"] == "neither":
            return "neither"
        winning_cfg = row["left_config"] if row["choice"] == "left" else row["right_config"]
        return "a" if winning_cfg == config_a else "b"

    df["winner"] = df.apply(winner, axis=1)

    report = {}
    for cat, group in df.groupby("category"):
        counts = group["winner"].value_counts(normalize=True).to_dict()
        report[cat] = {
            "n_votes": len(group),
            f"pct_{config_a}": 100 * counts.get("a", 0.0),
            f"pct_{config_b}": 100 * counts.get("b", 0.0),
            "pct_tie": 100 * counts.get("tie", 0.0),
            "pct_neither": 100 * counts.get("neither", 0.0),
        }
    return report


def compute_full_agreement_distribution(config_a: str, config_b: str, annotators: list[str]) -> dict:
    """Distribution d'accord entre TOUS les annotateurs sur chaque question
    (unanime / majoritaire / aucun accord), pas seulement par paire.
    S'inspire de la Figure 7 de Derby LLM (Bouvard et al., APIA@PFIA 2024) --
    répliquée ici parce qu'avec exactement 3 annotateurs comme eux, le Kappa
    par paire seul masque si le désaccord est généralisé ou concentré sur
    quelques questions limites."""
    import pandas as pd

    out_dir = config.RESULTS_DIR / "arena_votes"
    dfs = {}
    for ann in annotators:
        path = out_dir / f"arena_{config_a}_vs_{config_b}_{ann}.csv"
        if path.exists():
            dfs[ann] = _load_votes(path).set_index("question_index")["choice"]

    if len(dfs) < 3:
        return {"error": f"besoin de {len(annotators)} annotateurs, {len(dfs)} disponibles"}

    common_idx = None
    for s in dfs.values():
        common_idx = s.index if common_idx is None else common_idx.intersection(s.index)

    unanimous = majority = none = 0
    per_question = {}
    for q in common_idx:
        votes = [dfs[ann].loc[q] for ann in dfs]
        counts = pd.Series(votes).value_counts()
        top_count = counts.iloc[0]
        if top_count == len(votes):
            unanimous += 1
            level = "unanime"
        elif top_count >= 2:
            majority += 1
            level = "majoritaire"
        else:
            none += 1
            level = "aucun"
        per_question[int(q)] = {"votes": votes, "agreement_level": level}

    n = len(common_idx)
    return {
        "n_questions": n,
        "pct_unanimous": 100 * unanimous / n if n else None,
        "pct_majority_only": 100 * majority / n if n else None,
        "pct_no_agreement": 100 * none / n if n else None,
        "per_question": per_question,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", default="C2_rag:C3_finetune", help="config_a:config_b")
    parser.add_argument("--annotator", default=None, help="Pre-remplit le prenom dans l'interface (optionnel, modifiable)")
    parser.add_argument("--share", action="store_true", help="Cree un lien public temporaire Gradio (annotateurs a distance)")
    args = parser.parse_args()
    ca, cb = args.pair.split(":")
    run_app(ca, cb, args.annotator, share=args.share)
