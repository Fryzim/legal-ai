# Assistant IA juridique — RAG vs Fine-tuning (QLoRA)

Travail d'étude et de recherche (M1 MLDM, Université Jean Monnet) évaluant deux stratégies d'adaptation d'un LLM à un domaine de niche : la recherche documentaire augmentée (RAG) et le fine-tuning par QLoRA, appliquées à des questions de droit belge en français.

## Contexte

Un modèle généraliste (Mistral-7B-Instruct) répond mal aux questions juridiques précises car il n'a pas mémorisé le corpus de lois pertinent et ne peut pas citer ses sources. Deux approches concurrentes existent pour corriger cela : lui donner accès aux textes au moment de la requête (RAG), ou le spécialiser sur le domaine via fine-tuning. Ce projet compare les deux, seules et combinées, avec un protocole d'évaluation complet plutôt qu'une seule métrique.

## Protocole

- **Dataset** : [BSARD](https://huggingface.co/datasets/maastrichtlawtech/bsard) — 22 633 articles de loi belges, 222 questions de test annotées.
- **Retrieval** : BM25, embeddings denses (mpnet, e5-large), et un hybride BM25+e5, comparés sur Recall@k, MRR et nDCG.
- **Fine-tuning** : QLoRA sur Mistral-7B-Instruct, plusieurs configurations (rang LoRA, cibles d'attention, taille du jeu d'entraînement, données synthétiques), moyennées sur 3 seeds pour estimer la variance.
- **Génération** : 4 configurations comparées sur 222 questions — zero-shot, RAG seul, fine-tuning seul, fine-tuning + RAG — via ROUGE-L, BERTScore, exactitude de citation, taux d'hallucination, fidélité (méthode Derby LLM) et jugement humain à l'aveugle (arena).
- **Analyse** : intervalles de confiance bootstrap, tests de significativité appariés (bootstrap + Wilcoxon, correction Holm-Bonferroni), corrélation difficulté/qualité, et un modèle prédisant la fiabilité d'une réponse à partir de ses métadonnées.

## Résultat principal

La combinaison fine-tuning + RAG obtient le meilleur ROUGE-L (0.179 contre 0.110 en zero-shot), mais aucune configuration ne résout le vrai facteur de difficulté : le nombre d'articles de loi à combiner pour répondre correctement, qui corrèle négativement et significativement (p < 0.0001) avec la qualité dans les 4 configurations.

## Reporting

- **Live dashboard :** [fryzim.github.io/legal-ai](https://fryzim.github.io/legal-ai/) — Recall@k par méthode de retrieval, ROUGE-L par configuration (avec IC 95%), écart par catégorie juridique, corrélation difficulté/qualité (source dans `docs/index.html`, chiffres repris de `RESULTS.md`).
- **Power Query :** `reporting/power_query.m` — charge `results/difficulty_table.csv` (une ligne par question de test) pour Power BI/Excel.

## Contenu du dépôt

```
src/                  pipeline (retrieval, fine-tuning, génération, analyses statistiques)
kaggle_kernels/       jobs exécutés sur GPU Kaggle/RunPod (un par étape du pipeline)
results/              sorties JSON/CSV brutes de chaque expérience
figures/, tables/     figures et tableaux générés pour le rapport
report/               rapport LaTeX complet
docs/index.html       dashboard de reporting (GitHub Pages)
reporting/power_query.m   script Power Query (M) pour Power BI / Excel
RESULTS.md            résultats chiffrés consolidés, générés automatiquement depuis results/
DIFFICULTES.md        problèmes techniques rencontrés et solutions retenues
```

## Stack

Python, PyTorch, Transformers, PEFT/QLoRA, sentence-transformers, rank_bm25, scikit-learn.
