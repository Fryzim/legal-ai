"""Prédiction supervisée de la fiabilité d'une réponse (§6.10 du plan de rapport).

Question de recherche : peut-on prédire, à partir de caractéristiques observables
AVANT lecture de la réponse (longueur de la question, nombre d'articles gold,
catégorie, régularité de format d'identifiant, configuration), le risque qu'une
réponse soit peu fiable (citation incorrecte) ?

Inspiré de deux courants de la littérature (réimplémenté pour notre tâche, pas
copié) :
  - "Quality Estimation" en TAL (à la COMET-QE) : estimer la qualité d'une
    traduction/génération sans réponse de référence, à partir de features du
    contexte -- ici appliqué à la fiabilité de citation plutôt qu'à un score
    de traduction.
  - "Selective prediction" / option de rejet : un classifieur de risque peut
    servir de base à une règle d'abstention automatique (si risque prédit
    élevé, préférer "je ne sais pas"), au-delà du jeu d'abstention fixe déjà
    construit (build_abstention_set.py).

H0 : un classifieur entraîné sur ces caractéristiques ne fait pas mieux qu'un
modèle qui prédit toujours la classe majoritaire (AUC = 0.5).

Validation croisée à 5 plis stratifiée (peu d'observations: ~222 questions x
4 configs = 888 lignes), pas de split train/test unique qui serait trop
bruyant à ce volume.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import citation_format_analysis, config, difficulty_analysis

FIGURES_DIR = config.FIGURES_DIR
RESULTS_DIR = config.RESULTS_DIR

FEATURE_COLUMNS = [
    "question_length_words",
    "n_gold_articles",
    "category_train_frequency",
    "has_structured_gold",
]


def build_feature_matrix(generation_results: dict) -> pd.DataFrame:
    """Une ligne par (question, config). Features connues avant génération +
    la configuration elle-même (variable catégorielle, encodée one-hot)."""
    difficulty_df = difficulty_analysis.build_difficulty_table().reset_index(drop=True)
    format_df = citation_format_analysis.build_question_format_table().reset_index(drop=True)

    base = difficulty_df[["question_length_words", "n_gold_articles", "category_train_frequency", "category"]].copy()
    base["has_structured_gold"] = format_df["has_structured_gold"].astype(int)

    rows = []
    for cfg_name, cfg_data in generation_results["configs"].items():
        m = cfg_data["metrics"]
        if "citation_exact_match_per_question" not in m:
            continue
        cfg_df = base.copy()
        cfg_df["config"] = cfg_name
        cfg_df["citation_exact_match"] = m["citation_exact_match_per_question"]
        cfg_df["hallucination"] = m["hallucination_per_question"]
        cfg_df["rouge_l"] = m["rouge_l_f1_per_question"]
        rows.append(cfg_df)

    if not rows:
        raise ValueError(
            "Aucune config n'a de metriques par question (citation_exact_match_per_question). "
            "Necessite generation_job.py execute avec le correctif du 30/07 (per_question persiste)."
        )
    return pd.concat(rows, ignore_index=True)


def _prepare_xy(df: pd.DataFrame, target_col: str):
    X = pd.get_dummies(df[FEATURE_COLUMNS + ["config"]], columns=["config"], drop_first=False)
    y = df[target_col].values
    return X, y


def train_reliability_classifier(df: pd.DataFrame, target_col: str = "citation_exact_match") -> dict:
    """Régression logistique (interprétable) + Random Forest (interactions),
    validation croisée à 5 plis stratifiée. Retourne AUC, matrice de confusion
    (au seuil 0.5, agrégée sur les plis), importance des variables, et les
    points de la courbe ROC / calibration pour le tracé."""
    from sklearn.calibration import calibration_curve
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    X, y = _prepare_xy(df, target_col)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.SEED)

    results = {}
    for name, model in [
        ("logistic_regression", LogisticRegression(max_iter=1000)),
        ("random_forest", RandomForestClassifier(n_estimators=200, max_depth=6, random_state=config.SEED)),
    ]:
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        auc = roc_auc_score(y, proba)
        fpr, tpr, _ = roc_curve(y, proba)
        cm = confusion_matrix(y, pred).tolist()
        frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")

        model.fit(X, y)
        if name == "logistic_regression":
            importance = dict(zip(X.columns, model.coef_[0].tolist()))
        else:
            importance = dict(zip(X.columns, model.feature_importances_.tolist()))

        results[name] = {
            "auc": float(auc),
            "confusion_matrix": cm,
            "roc_fpr": fpr.tolist(),
            "roc_tpr": tpr.tolist(),
            "calibration_mean_predicted": mean_pred.tolist(),
            "calibration_fraction_positive": frac_pos.tolist(),
            "feature_importance": importance,
        }

    results["baseline_majority_class_auc"] = 0.5
    results["target_col"] = target_col
    results["n_observations"] = len(df)
    results["positive_rate"] = float(np.mean(y))
    return results


def train_quality_regressor(df: pd.DataFrame, target_col: str = "rouge_l") -> dict:
    """Régression (linéaire + Random Forest) sur une métrique continue
    (ROUGE-L), même protocole de validation croisée. Sert de contrepoint au
    classifieur binaire: la config domine-t-elle aussi la variance d'une
    métrique de surface continue, ou est-ce spécifique à la fiabilité ?"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import KFold, cross_val_predict

    X, y = _prepare_xy(df, target_col)
    cv = KFold(n_splits=5, shuffle=True, random_state=config.SEED)

    results = {"target_col": target_col, "n_observations": len(df)}
    for name, model in [
        ("linear_regression", LinearRegression()),
        ("random_forest", RandomForestRegressor(n_estimators=200, max_depth=6, random_state=config.SEED)),
    ]:
        pred = cross_val_predict(model, X, y, cv=cv)
        r2 = r2_score(y, pred)
        model.fit(X, y)
        importance = (
            dict(zip(X.columns, model.coef_.tolist()))
            if name == "linear_regression"
            else dict(zip(X.columns, model.feature_importances_.tolist()))
        )
        results[name] = {
            "r2": float(r2),
            "predicted": pred.tolist(),
            "actual": y.tolist(),
            "feature_importance": importance,
        }
    return results


def make_figures(classifier_results: dict, regressor_results: dict) -> None:
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ROC curve (les deux modeles)
    fig, ax = plt.subplots(figsize=(5, 5))
    for name in ("logistic_regression", "random_forest"):
        r = classifier_results[name]
        ax.plot(r["roc_fpr"], r["roc_tpr"], label=f"{name} (AUC={r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="hasard (AUC=0.5)")
    ax.set_xlabel("Taux de faux positifs")
    ax.set_ylabel("Taux de vrais positifs")
    ax.set_title(f"ROC — prédiction de {classifier_results['target_col']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quality_prediction_roc.pdf")
    plt.close(fig)

    # Matrice de confusion (random forest)
    cm = np.array(classifier_results["random_forest"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1], ["prédit: incorrect", "prédit: exact"])
    ax.set_yticks([0, 1], ["réel: incorrect", "réel: exact"])
    ax.set_title("Matrice de confusion (Random Forest)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quality_prediction_confusion_matrix.pdf")
    plt.close(fig)

    # Importance des variables (random forest)
    importance = classifier_results["random_forest"]["feature_importance"]
    sorted_items = sorted(importance.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh([k for k, _ in sorted_items], [v for _, v in sorted_items], color="#55A868")
    ax.set_xlabel("Importance (Random Forest)")
    ax.set_title("Quelle variable prédit le mieux la fiabilité de citation ?")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quality_prediction_feature_importance.pdf")
    plt.close(fig)

    # Courbe de calibration
    fig, ax = plt.subplots(figsize=(5, 5))
    for name in ("logistic_regression", "random_forest"):
        r = classifier_results[name]
        ax.plot(r["calibration_mean_predicted"], r["calibration_fraction_positive"], marker="o", label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="calibration parfaite")
    ax.set_xlabel("Probabilité prédite moyenne")
    ax.set_ylabel("Fraction réellement positive")
    ax.set_title("Calibration du classifieur de risque")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quality_prediction_calibration.pdf")
    plt.close(fig)

    # Predit vs reel (regression ROUGE-L, random forest)
    r = regressor_results["random_forest"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(r["actual"], r["predicted"], alpha=0.3, s=10)
    lims = [0, max(max(r["actual"]), max(r["predicted"])) * 1.05]
    ax.plot(lims, lims, "--", color="gray")
    ax.set_xlabel("ROUGE-L réel")
    ax.set_ylabel("ROUGE-L prédit (Random Forest, CV 5 plis)")
    ax.set_title(f"Prédit vs réel — ROUGE-L (R²={r['r2']:.3f})")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "quality_prediction_rouge_predicted_vs_actual.pdf")
    plt.close(fig)

    print(f"5 figures écrites dans {FIGURES_DIR}")


if __name__ == "__main__":
    gen_path = RESULTS_DIR / "generation_results.json"
    if not gen_path.exists():
        print(f"{gen_path} absent -- la prédiction de fiabilité sera calculable dès que la génération sera terminée.")
    else:
        with open(gen_path) as f:
            gen = json.load(f)
        df = build_feature_matrix(gen)
        clf_results = train_reliability_classifier(df, "citation_exact_match")
        reg_results = train_quality_regressor(df, "rouge_l")
        make_figures(clf_results, reg_results)

        summary = {
            "classifier": {k: v for k, v in clf_results.items() if k not in ("logistic_regression", "random_forest")},
            "classifier_auc": {name: clf_results[name]["auc"] for name in ("logistic_regression", "random_forest")},
            "classifier_feature_importance_rf": clf_results["random_forest"]["feature_importance"],
            "regressor_r2": {name: reg_results[name]["r2"] for name in ("linear_regression", "random_forest")},
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        with open(RESULTS_DIR / "quality_prediction_results.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Écrit: {RESULTS_DIR / 'quality_prediction_results.json'}")
