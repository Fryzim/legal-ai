# RESULTS.md — TER Assistant juridique : RAG vs Fine-tuning

Généré automatiquement depuis les JSON de `results/`. Chaque chiffre provient d'une exécution réelle horodatée (voir champ `timestamp` de chaque section). Aucune valeur n'est estimée ou reconstituée à la main.

## Retrieval

- Corpus: 22633 articles
- Questions test: 222
- Exécuté le: 2026-07-28T18:40:24Z

| Méthode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Recall@20 | MRR@10 | nDCG@10 | Durée (s) |
|---|---|---|---|---|---|---|---|---|
| bm25 | 0.063 | 0.140 | 0.184 | 0.224 | 0.289 | 0.215 | 0.176 | 20.4 |
| mpnet | 0.021 | 0.045 | 0.074 | 0.097 | 0.124 | 0.105 | 0.076 | 311.9 |
| e5_large | 0.083 | 0.169 | 0.211 | 0.296 | 0.377 | 0.288 | 0.230 | 900.9 |
| hybrid_bm25_e5 | 0.106 | 0.196 | 0.250 | 0.322 | 0.390 | 0.312 | 0.258 | 0.0 |

### Écarts constatés vs. comptes-rendus précédents

| | mpnet (CR) | mpnet (reproduit) | e5-large (CR) | e5-large (reproduit) |
|---|---|---|---|---|
| recall@1 | 0.117 | 0.021 | 0.198 | 0.083 |
| recall@3 | 0.203 | 0.045 | 0.347 | 0.169 |
| recall@5 | 0.261 | 0.074 | 0.423 | 0.211 |
| recall@10 | 0.347 | 0.097 | 0.523 | 0.296 |

**Écart non résolu** : les Recall@k reproduits sont systématiquement 2 à 5x plus bas que ceux annoncés dans les CR précédents, alors que le protocole (corpus 22 633 articles, 222 questions test, mêmes modèles, préfixes `query:`/`passage:` corrects pour e5-large) est identique. Distribution de longueur des articles vérifiée : médiane 77 mots, seulement 6.8% > 384 mots (limite de troncature mpnet) — la troncature seule n'explique probablement pas un écart de cette ampleur. Hypothèses à tester (non validées) : (1) le texte indexé dans les CR précédents incluait peut-être les métadonnées hiérarchiques (code/chapitre/section) en plus du corps de l'article, apportant un signal lexical supplémentaire ; (2) les CR précédents ont pu utiliser un fine-tuning contrastif léger du retriever (BSARD fournit des négatifs BM25 dans `negatives/`) plutôt qu'un modèle off-the-shelf. À investiguer dans l'ablation chunking/enrichissement de document (notebook 07, P1) avant d'écrire la section résultats du rapport — ne pas présenter les chiffres du CR comme acquis.

## Fine-tuning QLoRA

| Run | Seed | Train src | n_train | n_val | r | Cibles | Loss train | Loss eval | Écart (surapprentissage) | Durée (h) |
|---|---|---|---|---|---|---|---|---|---|---|
| ? | 42 | official | 580 | ? | 16 | attn | 0.8411 | n/a | n/a | 9.52 |
| seed123_n580_r32_attn | 123 | official | 580 | 100 | 32 | attn | 0.8439 | 0.6894 | +0.1076 | 0.25 |
| seed2026_n580_r32_attn | 2026 | official | 580 | 100 | 32 | attn | 0.8496 | 0.6889 | +0.1409 | 0.25 |
| seed42_n190_r32_attn | 42 | official | 190 | 100 | 32 | attn | 1.1224 | 1.0981 | +0.1020 | 0.09 |
| seed42_n331_r32_attn_CodeCivil | 42 | official | 331 | 100 | 32 | attn | 0.8319 | 1.0153 | +0.4623 | 0.15 |
| seed42_n380_r32_attn | 42 | official | 380 | 100 | 32 | attn | 0.9656 | 0.8604 | +0.1371 | 0.17 |
| seed42_n580_r32_attn | 42 | official | 580 | 100 | 32 | attn | 0.8344 | 0.6793 | +0.1423 | 0.26 |
| seed42_n580_r32_attn_mlp | 42 | official | 580 | 100 | 32 | attn_mlp | 0.5269 | 0.4158 | +0.2136 | 0.33 |
| seed42_n580_r32_attn_synth | 42 | official_plus_synthetic | 2580 | 100 | 32 | attn | 0.9401 | 0.5161 | -0.2350 | 0.86 |
| seed42_n580_r8_attn | 42 | official | 580 | 100 | 8 | attn | 0.8504 | 0.7101 | +0.1352 | 0.26 |
| seed42_n786_r32_attn | 42 | official | 786 | 100 | 32 | attn | 0.7359 | 0.5607 | +0.1352 | 0.34 |

**Variance inter-seeds (config principale, n=3 seeds)** : loss finale = 0.8426 ± 0.0063

## Génération — 4 configurations (222 questions test)

- Exécuté le: 2026-07-30T17:02:32Z
- top_k RAG: 5

| Config | ROUGE-L | BERTScore F1 | Précision citation | Rappel citation | Exact match | Taux hallucination | Durée (s) |
|---|---|---|---|---|---|---|---|
| C1_zero_shot | 0.1100 | 0.7788 | 0.016 | 0.011 | 0.005 | 0.144 | 258.3 |
| C2_rag | 0.1395 | 0.7947 | 0.176 | 0.132 | 0.036 | 0.234 | 622.2 |
| C3_finetune | 0.1467 | 0.8009 | 0.023 | 0.012 | 0.000 | 0.050 | 343.4 |
| C4_finetune_rag | 0.1790 | 0.7978 | 0.069 | 0.053 | 0.005 | 0.167 | 705.5 |

### IC bootstrap 95% (ROUGE-L, 1000 rééchantillonnages)

| Config | Moyenne | IC 95% bas | IC 95% haut |
|---|---|---|---|
| C1_zero_shot | 0.1100 | 0.1037 | 0.1163 |
| C2_rag | 0.1395 | 0.1292 | 0.1511 |
| C3_finetune | 0.1467 | 0.1360 | 0.1598 |
| C4_finetune_rag | 0.1790 | 0.1565 | 0.2017 |

### Tests de significativité appariés (Holm-Bonferroni)

Deux tests indépendants par paire (le bootstrap ne suppose rien sur la distribution des différences, Wilcoxon suppose une distribution symétrique) -- s'ils s'accordent, la conclusion est plus solide qu'avec un seul test.

| Comparaison | Différence observée | p-value (bootstrap) | p-value (Wilcoxon) | Significatif (corrigé) | Tests d'accord |
|---|---|---|---|---|---|
| C1_zero_shot_vs_C2_rag | -0.0295 | 0.0000 | 0.0000 | oui | oui |
| C1_zero_shot_vs_C3_finetune | -0.0368 | 0.0000 | 0.0000 | oui | oui |
| C1_zero_shot_vs_C4_finetune_rag | -0.0691 | 0.0000 | 0.0000 | oui | oui |
| C2_rag_vs_C3_finetune | -0.0072 | 0.2960 | 0.0796 | non | oui |
| C2_rag_vs_C4_finetune_rag | -0.0395 | 0.0000 | 0.0002 | oui | oui |
| C3_finetune_vs_C4_finetune_rag | -0.0323 | 0.0040 | 0.3066 | oui | non |

### Analyse par catégorie juridique (demande A. Habrard)

| Catégorie | n | C1_zero_shot | C2_rag | C3_finetune | C4_finetune_rag |
|---|---|---|---|---|---|
| Argent | 36 | 0.112 | 0.167 | 0.128 | 0.191 |
| Etrangers | 13 | 0.096 | 0.118 | 0.126 | 0.117 |
| Famille | 67 | 0.117 | 0.146 | 0.179 | 0.182 |
| Justice | 30 | 0.082 | 0.099 | 0.105 | 0.116 |
| Logement | 66 | 0.118 | 0.140 | 0.149 | 0.216 |
| Protection sociale | 4 | 0.087 | 0.126 | 0.112 | 0.137 |
| Travail | 6 | 0.121 | 0.153 | 0.154 | 0.138 |

## Fidélité (méthode Derby LLM, Bouvard et al. APIA@PFIA 2024)

Recouvrement des passages d'intérêt (entités nommées, nombres, emails, URLs) entre réponse générée et texte de référence -- métrique déterministe, réimplémentée pour se positionner directement face à la référence imposée.

| Config | Fidélité moyenne | n avec passages d'intérêt |
|---|---|---|
| C1_zero_shot | 0.054 | 221/222 |
| C2_rag | 0.295 | 219/222 |
| C3_finetune | 0.298 | 222/222 |
| C4_finetune_rag | 0.324 | 210/222 |

## LLM-as-judge (pertinence)

Juge: microsoft/Phi-3.5-mini-instruct · 50 questions · 10 échantillons/réponse (température > 0), moyennés -- un LLM-juge est non-déterministe, un jugement greedy unique est bruyant.

| Config | Pertinence moyenne | Écart-type intra-réponse moyen | Taux de parsing JSON ok |
|---|---|---|---|
| C1_zero_shot | 5.000 | 0.000 | 0.922 |
| C2_rag | 4.850 | 0.048 | 0.878 |
| C3_finetune | 4.731 | 0.121 | 0.906 |
| C4_finetune_rag | 4.617 | 0.107 | 0.924 |

## Régularité de format des identifiants d'article

Découvert en creusant le bug d'extraction regex (cf. DIFFICULTES.md §9). Vérifié : ce n'est PAS un proxy de la juridiction fédéral/régional (les deux ont des proportions comparables de formats irréguliers) -- axe indépendant.

| Format | n articles corpus |
|---|---|
| structured | 13199 |
| numeric_simple | 9434 |

| Format | n questions test |
|---|---|
| numeric_simple | 118 |
| structured | 104 |

**Hypothèse testée** : citation exact match plus bas sur les questions dont au moins un article gold a un identifiant structuré, par rapport aux identifiants numériques simples (test de Mann-Whitney par configuration).

### Résultat du test d'hypothèse citation/format

| Config | n simple | n structuré | Moyenne (simple) | Moyenne (structuré) | p-value (Mann-Whitney) | Significatif |
|---|---|---|---|---|---|---|
| C1_zero_shot | 118 | 104 | 0.0085 | 0.0000 | 0.3525 | non |
| C2_rag | 118 | 104 | 0.0508 | 0.0192 | 0.2094 | non |
| C3_finetune | 118 | 104 | 0.0000 | 0.0000 | 1.0000 | non |
| C4_finetune_rag | 118 | 104 | 0.0085 | 0.0000 | 0.3525 | non |

## Corrélation difficulté / qualité (Spearman)

Réponse directe à A. Habrard : le comportement moyen du modèle varie-t-il avec la difficulté de la question (longueur, nombre d'articles gold requis) ?

| Config | r (longueur question) | p-value | r (n articles gold) | p-value |
|---|---|---|---|---|
| C1_zero_shot | -0.107 | 0.1125 | -0.452* | 0.0000 |
| C2_rag | -0.076 | 0.2584 | -0.336* | 0.0000 |
| C3_finetune | -0.064 | 0.3403 | -0.510* | 0.0000 |
| C4_finetune_rag | -0.152* | 0.0236 | -0.353* | 0.0000 |

*(`*` = significatif à p<0.05)*

**Interprétation** : la longueur brute de la question n'a quasiment aucun effet (non significatif dans 3 configs sur 4). En revanche le nombre d'articles de loi nécessaires pour répondre correctement corrèle négativement avec la qualité, de façon forte et hautement significative dans les 4 configurations (p < 0.0001 partout). C'est le nombre de règles à combiner, pas la longueur du texte, qui définit la vraie difficulté d'une question juridique -- et ni le RAG ni le fine-tuning ne corrigent ce problème structurel (voir figure `difficulty_bucket_rouge.pdf` : le ROUGE-L chute nettement dès 4+ articles requis, dans les 4 configs sans exception).

## Prédiction ML de la fiabilité (§6.10)

Cible : `citation_exact_match` -- 888 observations, taux de positifs 0.0113 (déséquilibré, interpréter l'AUC avec prudence).

| Modèle | AUC (classification) |
|---|---|
| logistic_regression | 0.8782 |
| random_forest | 0.8384 |
| baseline (classe majoritaire) | 0.5000 |

| Modèle | R² (régression ROUGE-L) |
|---|---|
| linear_regression | 0.1364 |
| random_forest | 0.1662 |

**Importance des variables (Random Forest, classification)** : 
question_length_words (0.389), category_train_frequency (0.185), n_gold_articles (0.158), config_C2_rag (0.127), has_structured_gold (0.066), config_C3_finetune (0.027), config_C1_zero_shot (0.025), config_C4_finetune_rag (0.022)
