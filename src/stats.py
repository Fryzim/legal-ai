"""Statistiques: bootstrap CI, tests appariés, correction Holm-Bonferroni.

Aucun GPU requis — tourne en local sur les prédictions déjà générées.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from . import config


def bootstrap_ci(
    values: list[float],
    n_resamples: int = config.BOOTSTRAP_N_RESAMPLES,
    confidence: float = config.CONFIDENCE_LEVEL,
    seed: int = config.SEED,
    statistic=np.mean,
) -> dict:
    """IC bootstrap non-paramétrique sur une métrique par-question."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    boot_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = values[rng.integers(0, n, size=n)]
        boot_stats[i] = statistic(sample)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(boot_stats, [alpha, 1 - alpha])
    return {
        "mean": float(statistic(values)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n": int(n),
        "n_resamples": n_resamples,
    }


def paired_bootstrap_test(
    values_a: list[float],
    values_b: list[float],
    n_resamples: int = config.BOOTSTRAP_N_RESAMPLES,
    seed: int = config.SEED,
) -> dict:
    """Test de significativité par bootstrap apparié sur les différences (a - b).

    H0: la différence moyenne est nulle. p-value = proportion de rééchantillons
    où le signe de la différence s'inverse par rapport à l'observé.
    """
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diffs = a - b
    observed_diff = diffs.mean()

    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot_diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = diffs[rng.integers(0, n, size=n)]
        boot_diffs[i] = sample.mean()

    if observed_diff >= 0:
        p_value = 2 * min((boot_diffs <= 0).mean(), 0.5)
    else:
        p_value = 2 * min((boot_diffs >= 0).mean(), 0.5)
    p_value = min(p_value, 1.0)

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "n": int(n),
        "n_resamples": n_resamples,
    }


def wilcoxon_signed_rank(values_a: list[float], values_b: list[float]) -> dict:
    from scipy.stats import wilcoxon

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    try:
        stat, p = wilcoxon(a, b)
    except ValueError:
        # toutes les différences sont nulles
        stat, p = 0.0, 1.0
    return {"statistic": float(stat), "p_value": float(p), "n": int(len(a))}


def holm_bonferroni_correction(p_values: dict[str, float], alpha: float = 0.05) -> dict:
    """p_values: {nom_comparaison: p_value}. Retourne le statut significatif après correction."""
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    results = {}
    reject_all_after = False
    for i, (name, p) in enumerate(items):
        threshold = alpha / (m - i)
        significant = (not reject_all_after) and (p <= threshold)
        if not significant:
            reject_all_after = True
        results[name] = {
            "p_value": p,
            "threshold": threshold,
            "significant_after_correction": significant,
        }
    return results


def compare_all_configs(
    per_question_scores: dict[str, list[float]],
    n_resamples: int = config.BOOTSTRAP_N_RESAMPLES,
    seed: int = config.SEED,
) -> dict:
    """Compare toutes les paires de configurations sur une métrique donnée,
    avec correction de Holm-Bonferroni pour comparaisons multiples.
    """
    config_names = list(per_question_scores.keys())
    pairwise = {}
    p_values = {}
    for a, b in combinations(config_names, 2):
        test = paired_bootstrap_test(
            per_question_scores[a], per_question_scores[b], n_resamples=n_resamples, seed=seed
        )
        # Wilcoxon signe en complement du bootstrap: deux tests independants,
        # d'hypotheses differentes (le bootstrap ne suppose rien sur la
        # distribution, Wilcoxon suppose une distribution symetrique des
        # differences) -- si les deux s'accordent, la conclusion est plus
        # solide. Code depuis le debut (wilcoxon_signed_rank) mais jamais
        # cable ici jusqu'a present -- trouve en verifiant l'inventaire des
        # tests reellement utilises.
        wilcoxon = wilcoxon_signed_rank(per_question_scores[a], per_question_scores[b])
        test["wilcoxon_p_value"] = wilcoxon["p_value"]
        test["wilcoxon_statistic"] = wilcoxon["statistic"]
        test["tests_agree"] = (test["p_value"] < 0.05) == (wilcoxon["p_value"] < 0.05)
        key = f"{a}_vs_{b}"
        pairwise[key] = test
        p_values[key] = test["p_value"]

    corrected = holm_bonferroni_correction(p_values)
    for key in pairwise:
        pairwise[key]["significant_after_correction"] = corrected[key]["significant_after_correction"]
        pairwise[key]["holm_threshold"] = corrected[key]["threshold"]

    cis = {name: bootstrap_ci(scores, n_resamples=n_resamples, seed=seed) for name, scores in per_question_scores.items()}

    return {"confidence_intervals": cis, "pairwise_tests": pairwise}
