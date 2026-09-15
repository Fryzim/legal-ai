# Arène humaine — C2_rag vs C3_finetune

Résultats calculés via `src/arena_app.py` (fonctions `compute_inter_annotator_agreement`, `compute_bradley_terry_ranking`, `compute_preference_by_category`, `compute_full_agreement_distribution`) sur les votes de 3 annotateurs (Bartosz, Chaabane, Arman), 60 questions chacun.

Détail des chiffres et JSON brut : `results/arena_human_evaluation_results.json`.

**Note méthodologique** : le CSV de Bartosz contenait 4 doublons (questions votées deux fois, probablement après un rechargement de page pendant la session — la sauvegarde est incrémentale et n'empêche pas de revoter). Dédupliqués en gardant le dernier vote avant tout calcul.

## 1. Accord inter-annotateurs (Cohen's Kappa)

| Paire | Kappa | Accord exact | n questions communes |
|---|---|---|---|
| Bartosz vs Chaabane | 0.389 | 56.7% | 60 |
| Bartosz vs Arman | 0.471 | 66.7% | 60 |
| Chaabane vs Arman | 0.331 | 53.3% | 60 |

Accord "modéré" (0.21–0.60) sur les 3 paires, sans paire aberrante.

## 2. Classement Bradley-Terry (global, 3 annotateurs)

| Config | Score (échelle Elo) |
|---|---|
| C2_rag | 1368.7 |
| C3_finetune | 631.3 |

103 → 161 votes utilisés (hors choix "neither"), convergence atteinte. **C2_rag est nettement préféré à C3_finetune**, de façon cohérente sur les 3 annotateurs.

## 3. Préférence par catégorie juridique (3 annotateurs combinés)

| Catégorie | n votes | % C2_rag | % C3_finetune | % tie | % neither |
|---|---|---|---|---|---|
| Famille | 30 | 86.7% | 6.7% | 0.0% | 6.7% |
| Argent | 30 | 76.7% | 3.3% | 10.0% | 10.0% |
| Justice | 30 | 70.0% | 3.3% | 16.7% | 10.0% |
| Travail | 18 | 72.2% | 0.0% | 11.1% | 16.7% |
| Logement | 30 | 66.7% | 6.7% | 6.7% | 20.0% |
| Etrangers | 30 | 60.0% | 20.0% | 16.7% | 3.3% |
| Protection sociale | 12 | 50.0% | 8.3% | 8.3% | 33.3% |

C2_rag l'emporte dans **toutes** les catégories, avec la marge la plus faible sur Protection sociale (n=12, échantillon réduit).

## 4. Distribution d'accord complète (3 annotateurs, 60 questions)

| Niveau d'accord | % |
|---|---|
| Unanime (3/3) | 45.0% |
| Majoritaire (2/3) | 41.7% |
| Aucun accord | 13.3% |

## À retenir

L'arène humaine converge clairement en faveur de **C2 (RAG)** sur **C3 (fine-tuning seul)**, dans toutes les catégories juridiques, avec un accord inter-annotateurs raisonnable (Kappa 0.33–0.47). Ce résultat est plus tranché que celui des métriques automatiques (ROUGE-L ne montrait pas de différence significative entre C2 et C3 dans `RESULTS.md`) — un écart intéressant à discuter entre métrique de surface et préférence humaine réelle.
