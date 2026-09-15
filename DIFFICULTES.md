# Difficultés techniques rencontrées et recommandations pratiques

Ce document liste, au fil de l'eau, les problèmes techniques réellement rencontrés
pendant le développement, avec leur diagnostic exact et la solution appliquée.
Format demandé par A. Habrard pour l'annexe du rapport MLDM : des recommandations
d'usage des bibliothèques, pas un roman des péripéties.

---

## 1. Kaggle assigne par défaut un GPU Tesla P100, incompatible avec le PyTorch préinstallé

**Contexte** : premier lancement du kernel de retrieval (`kaggle kernels push`, `enable_gpu: true`,
sans précision du type d'accélérateur).

**Symptôme exact** :
```
Tesla P100-PCIE-16GB with CUDA capability sm_60 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120.
...
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

**Diagnostic** : l'image Docker Kaggle embarque désormais une version de PyTorch compilée
sans support de l'architecture Pascal (sm_60, celle des P100). Comme la sélection
d'accélérateur n'est pas indiquée dans `kernel-metadata.json` par défaut, Kaggle
attribue silencieusement un P100 — le job échoue dès le premier appel `.encode()`
sur le GPU, après plusieurs minutes de setup (perte de temps + de quota).

**Solution** : forcer explicitement un T4 au push, via le champ `machine_shape`
(sous-documenté dans le README du CLI mais accepté par l'API) :

```json
// kernel-metadata.json
{
  "enable_gpu": true,
  "machine_shape": "NvidiaTeslaT4"
}
```
ou en ligne de commande :
```bash
kaggle kernels push -p mon_dossier --accelerator NvidiaTeslaT4
```

**Recommandation** : ne jamais lancer un kernel GPU sur Kaggle sans fixer `machine_shape`
explicitement à `NvidiaTeslaT4` (P100 est aujourd'hui un accélérateur legacy à éviter
pour tout stack PyTorch récent). Le bug ne se manifeste qu'après le début de l'exécution,
donc invisible tant qu'on ne relit pas les logs — vérifier systématiquement
`kaggle kernels status` puis les logs avant de supposer qu'un job tourne correctement.

---

## 2. `rank_bm25` absent de l'image Docker Kaggle par défaut

**Symptôme** : `ModuleNotFoundError: No module named 'rank_bm25'` dès le début du script,
alors que `sentence-transformers`, `torch`, `transformers` sont préinstallés.

**Solution** : installer les dépendances manquantes en tout début de script via
`subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rank_bm25"])`
plutôt que de supposer l'environnement Kaggle complet. Toujours lister explicitement
les paquets non-standards utilisés (rank_bm25, peft, trl, bitsandbytes selon les
versions d'image) et les installer en tête de script, avant les imports qui en dépendent.

**Recommandation générale** : pour tout job Kaggle poussé via API (pas testé
interactivement dans l'UI au préalable), traiter l'environnement comme "transformers +
torch + pandas/numpy seulement" et installer explicitement le reste.

---

## 3. `article_ids` de BSARD est un string, pas une liste

**Symptôme** : bugs silencieux lors du calcul de Recall@k — comparer un `str` complet
(`"947,948"`) à des IDs entiers ne matche jamais, donnant un Recall@k artificiellement
proche de 0 sans erreur explicite.

**Solution** : parser systématiquement avec `[int(x) for x in str(raw).split(",") if x.strip()]`
dès le chargement (cf. `src/data.py::parse_article_ids`), jamais en aval dans le pipeline.

**Recommandation** : ajouter une assertion de type au chargement de tout dataset externe
(`assert isinstance(row["article_ids"], list)`) plutôt que de découvrir le bug via des
métriques anormalement basses en fin de pipeline.

---

## 4. Modèle Mistral-7B-Instruct-v0.3 : gating supposé à tort

**Contexte** : le brief initial du projet supposait que `mistralai/Mistral-7B-Instruct-v0.3`
nécessitait une acceptation de licence manuelle sur HuggingFace (comme les modèles Llama),
ce qui aurait contrevenu à la contrainte "ressources gratuites sans demande d'accès".

**Vérification effectuée** : test de téléchargement anonyme (`curl` sans token) du fichier
`config.json` du dépôt officiel → `HTTP 200`. Le dépôt est en réalité public, sous licence
Apache 2.0, sans gating.

**Recommandation** : ne jamais supposer le statut d'accès d'un modèle HuggingFace à partir
de sa famille ou de sa réputation — toujours vérifier avec une requête anonyme réelle avant
de bâtir un plan de contournement (miroir, token, etc.) qui pourrait être inutile.

---

## 5. BSARD ne contient pas de réponse rédigée (limite méthodologique, pas un bug)

**Constat** : les colonnes de `questions_test.csv`/`questions_train.csv` sont
`id, category, subcategory, question, extra_description, article_ids` — aucun champ
de réponse en langage naturel. BSARD est nativement un dataset de *retrieval*, pas de QA
générative.

**Conséquence pour le protocole** : la "réponse de référence" utilisée pour ROUGE-L et
BERTScore est construite par concaténation du texte des articles cités
(`src/data.py::build_reference_answer`). Ce n'est pas une réponse rédigée idéale, donc
les scores de similarité de surface (ROUGE-L, BERTScore) doivent être interprétés comme
une mesure de *recouvrement lexical avec le texte de loi*, pas de qualité rédactionnelle.
Cette limite doit être assumée explicitement dans le rapport, pas dissimulée — elle explique
en grande partie pourquoi tous les ROUGE-L observés restent bas (0.08–0.16 dans nos CR
précédents).

**Recommandation** : pour toute réutilisation de BSARD en génération (et non en retrieval
pur), documenter cette reconstruction de référence dès la section méthodologie, avant de
présenter les résultats.

---

## 6. `trl` récent : `SFTTrainer` n'accepte plus `TrainingArguments` ni `dataset_text_field`/`max_seq_length` directement

**Symptôme** : `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'dataset_text_field'`
alors que ce paramètre est documenté dans de nombreux tutoriels QLoRA trouvés en ligne.

**Diagnostic** : les versions récentes de `trl` (installées via `pip install -U trl`) ont
remplacé `transformers.TrainingArguments` par une classe dédiée `trl.SFTConfig`, qui
regroupe désormais les hyperparamètres d'entraînement ET les paramètres spécifiques au
SFT (`dataset_text_field`, et `max_length` — qui a lui-même remplacé `max_seq_length`).

**Solution** :
```python
from trl import SFTConfig, SFTTrainer

args = SFTConfig(
    output_dir=..., num_train_epochs=..., per_device_train_batch_size=...,
    dataset_text_field="text", max_length=1024,
)
trainer = SFTTrainer(model=model, args=args, train_dataset=ds)
```

**Recommandation** : pour toute bibliothèque à évolution rapide (`trl`, `peft`,
`bitsandbytes`), ne pas se fier à des tutoriels/exemples externes sans vérifier la
signature réelle de la version installée (`inspect.signature(...)` ou lecture directe
du fichier source dans `site-packages`) — l'API change entre versions mineures sans
rétrocompatibilité systématique.

## 7. `trl` récent : `loss_type="chunked_nll"` (nouveau défaut) casse sur modèle quantifié + PEFT

**Symptôme** : `AttributeError: 'functools.partial' object has no attribute '__func__'` levé
dans `SFTTrainer.__init__` → `_patch_chunked_ce_lm_head`, avant même le début de l'entraînement.

**Diagnostic** : la version de `trl` installée (`pip install -U trl`) a introduit un nouveau
mode de loss par défaut, `loss_type="chunked_nll"`, qui patche dynamiquement le `forward` du
modèle pour économiser de la mémoire sur le calcul de la cross-entropy. Ce patch suppose une
structure de `lm_head` non modifiée ; avec un modèle chargé en 4-bit (bitsandbytes) et enveloppé
par un adaptateur PEFT (LoRA), le `lm_head` est manipulé différemment et le patch casse.

**Solution** : forcer explicitement `loss_type="nll"` (loss standard, comportement historique)
dans `SFTConfig` :
```python
args = SFTConfig(..., loss_type="nll")
```

**Recommandation** : pour du QLoRA (4-bit + PEFT), ne jamais laisser les valeurs par défaut
d'une bibliothèque encore en évolution rapide sur les optimisations de loss/mémoire — les
"améliorations" par défaut (chunked CE, Liger kernel, etc.) sont souvent testées d'abord sur
des modèles pleine précision non quantifiés. Toujours tester avec `loss_type="nll"` en premier
sur un stack quantifié + PEFT, puis n'activer les optimisations que si elles sont explicitement
validées pour ce cas d'usage.

## 8. Développement local sur Mac Intel (x86_64) : `torch` plafonné à 2.2.2

**Constat** : `pip index versions torch` sur la machine de développement locale (macOS Intel)
ne propose rien au-delà de la 2.2.2 — PyTorch a arrêté de publier des wheels pour macOS x86_64
après cette version. Or `transformers` récent exige `torch>=2.4`, ce qui bloque l'usage de
`bert_score`/`AutoModel` en local (sans toucher au GPU).

**Impact réel** : aucun sur le pipeline final — tout le calcul GPU (retrieval, fine-tuning,
génération, BERTScore) tourne sur Kaggle (Linux + CUDA), où `torch`/`transformers` récents
et mutuellement compatibles sont installés nativement. La logique pure Python (parsing,
regex de citation, ROUGE-L, bootstrap, abstention) a été validée en local avec succès ; seul
le chargement d'un modèle `AutoModel` local est impossible sur cette machine.

**Recommandation** : sur Mac Intel, ne pas chercher à répliquer localement l'environnement
GPU — utiliser le local uniquement pour la logique CPU-only (regex, stats, parsing de
données) et déléguer tout ce qui charge un modèle `transformers` à l'environnement cible
(Kaggle/Colab). Tenter de forcer une version de `torch` non publiée pour sa plateforme est
une perte de temps.

## 9. Identifiants d'article BSARD : ~32% ne sont pas purement numériques

**Constat** : un premier regex d'extraction de numéro d'article (`Art\.\s*(\d+[a-zA-Z]?)`)
ne matchait que 68% des références du corpus. Les régions belges utilisent des systèmes de
numérotation hétérogènes selon le code : `Art. 959` (numérique simple), mais aussi
`Art. N1.1` (Bruxelles, Air/Climat), `Art. L1122-9` (Wallonie, Démocratie locale),
`Art. R.II.21-7` (Wallonie, réglementaire), `Art. D382` (Wallonie, décret), `Art. 259bis7`
(suffixe alphabétique après un nombre).

**Impact potentiel si non corrigé** : 32% des articles de référence auraient une citation
"gold" vide, gonflant artificiellement le taux d'hallucination détecté et faussant
precision/recall de citation. Détecté avant le lancement du job de génération (qui aurait
sinon consommé du quota GPU sur des métriques silencieusement fausses) — et avant que le job
de fine-tuning déjà lancé ne termine avec des exemples d'entraînement dont la citation cible
dupliquait la référence complète (`"Article Art. 959, Code Judiciaire (Livre II...)"` au lieu
de `"Article 959"`), repéré à ~15-20 min d'un run de plusieurs heures → job relancé avant que
le coût du re-lancement ne devienne prohibitif.

**Solution** : regex généralisé `Art\.\s*([^,]+)` (tout jusqu'à la virgule séparant
l'identifiant du nom du code), vérifié à 100% de couverture sur `articles.csv['reference']`
via une assertion explicite dans le pipeline (`match_rate > 0.99`).

**Recommandation** : sur un corpus juridique multi-juridictionnel (ici: droit fédéral +
régional belge), ne jamais supposer un format d'identifiant homogène. Toujours mesurer le
taux de couverture d'un pattern d'extraction sur l'intégralité du corpus avant de l'utiliser
pour construire des métriques, et ajouter une assertion de couverture minimale qui fait
échouer le job tôt plutôt que de produire des métriques silencieusement biaisées.

## 10. Kaggle : limite de 2 sessions GPU "batch" simultanées, et échecs de push qui créent des brouillons orphelins

**Symptôme 1** : `kaggle kernels push` avec `machine_shape` défini échoue avec
`Maximum batch GPU session count of 2 reached` dès qu'une deuxième session GPU
(lancée via l'API OU via une session interactive ouverte dans le navigateur)
est déjà active sur le compte.

**Symptôme 2, plus trompeur** : quand cette limite est atteinte, certains échecs
de push renvoient un message différent et non explicite, `Notebook not found`,
au lieu du message de quota clair. Dans les deux cas, **le kernel est quand
même créé côté serveur en tant que brouillon** ("[Private Notebook]" dans
`kaggle kernels list --mine`, sans `ref` exploitable), alors que le push a
échoué. Recommencer le push sans nettoyer ces brouillons crée un nouveau
brouillon à chaque tentative, qui vient lui-même occuper un slot de quota —
un cercle vicieux qui aggrave le blocage au lieu de le résoudre.

**Diagnostic** : la limite de 2 sessions batch GPU simultanées est une
contrainte du compte gratuit, indépendante du nombre de kernels *terminés*
(`COMPLETE`) — seuls les kernels réellement en file d'attente ou en cours
comptent. Les sessions interactives ouvertes dans l'éditeur web du navigateur
comptent dans le même quota que les kernels lancés par API, et ne sont ni
visibles ni contrôlables depuis la CLI.

**Solution appliquée** :
1. `kaggle kernels list --mine --csv` pour repérer les brouillons `[Private
   Notebook]` orphelins.
2. `kaggle kernels delete -y <owner>/<slug-devine>` pour les supprimer (le
   `ref` n'étant pas affiché, il faut deviner le slug prévu au moment du push
   raté — généralement il correspond à l'`id` du `kernel-metadata.json`
   utilisé lors de la tentative).
3. Si le quota reste bloqué après nettoyage, la cause est externe (session
   interactive ouverte ailleurs sur le compte) — vérifier manuellement sur
   kaggle.com/work.

**Recommandation** : ne jamais relancer un `kaggle kernels push` en boucle
après un échec sans d'abord vérifier `kaggle kernels list --mine --csv` pour
détecter et nettoyer les brouillons orphelins. Sur un compte gratuit
partagé/actif, prévoir que le quota de 2 sessions GPU simultanées peut être
consommé par une activité invisible depuis la CLI (session interactive
navigateur) — le message d'erreur ne le précise pas.

## 11. `kaggle kernels push` d'une nouvelle version NE stoppe PAS l'ancienne session en cours

**Découverte (avec capture d'écran de l'utilisateur à l'appui)** : chaque `kaggle kernels
push` sur un kernel déjà lancé crée une nouvelle *version* et la lance — mais la session
GPU de la version précédente continue de tourner en parallèle si elle n'a pas terminé
(succès ou erreur) au moment du nouveau push. Dans ce projet, la Version #3 du kernel de
fine-tuning (poussée avant la correction du bug de citation, cf. §9) tournait encore
"Running: 1h" en même temps que la Version #4 (avec le correctif) — **les deux consommaient
chacune un des 2 slots GPU batch du compte**, expliquant à elles seules toute la saga de
blocage "Maximum batch GPU session count of 2 reached" des sections précédentes. Aucune
session "mystère" externe n'était en cause.

**Solution** : avant de pousser une nouvelle version d'un kernel pour corriger un bug,
**arrêter explicitement la version précédente** depuis l'interface web
(kaggle.com/work → "..." sur la version en cours → Stop/Cancel) si elle est encore en
cours d'exécution. `kaggle kernels status` ne renvoie que l'état de la dernière version
poussée — il ne signale PAS qu'une version antérieure tourne encore en parallèle et
consomme du quota. `kaggle kernels list --mine` ne montre pas non plus les sessions actives
par version.

**Recommandation** : sur Kaggle, un correctif de bug en cours d'exécution == deux sessions
actives, pas une seule, tant que l'ancienne n'est pas arrêtée manuellement. Vérifier
systématiquement l'interface web après un re-push corrigeant un bug détecté en cours de
run, plutôt que de supposer que la nouvelle version remplace proprement l'ancienne.

## 12. Reranker cross-encoder : OOM CUDA sur des articles de loi très longs

**Symptôme** : `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.20 GiB`
pendant le passage avant du `CrossEncoder("BAAI/bge-reranker-v2-m3")`, sur la 4e variante
de l'ablation retrieval (après 3 variantes réussies).

**Diagnostic** : la distribution de longueur des articles BSARD a une très longue traîne
(médiane 77 mots, max 5790 mots). `CrossEncoder.predict()` sans `max_length` ni
`batch_size` explicites tente d'encoder des paires (question, article-très-long) en un
batch, et la mémoire attention (quadratique en longueur de séquence) explose sur les
articles les plus longs du top-20.

**Solution** : `CrossEncoder(..., max_length=512)` (troncature tokenizer) + troncature
brute du texte en amont (`text[:2000]`) + `batch_size=8` explicite dans `.predict()` +
vidage du cache CUDA tous les 50 questions.

**Leçon plus générale, déjà appliquée à ce stade** : sauvegarde incrémentale du JSON de
résultats après CHAQUE variante d'ablation (pas seulement à la toute fin du script) —
la première exécution a perdu 3 variantes déjà terminées avec succès parce que le crash
de la 4e a empêché le `json.dump` final de s'exécuter. Récupérées manuellement depuis les
logs stdout du kernel, mais une sauvegarde incrémentale aurait évité ce risque.

**Recommandation** : sur tout corpus au contenu de longueur très variable (ici: textes de
loi), toujours fixer explicitement `max_length` pour un cross-encoder/reranker, ne jamais
laisser une valeur par défaut. Et plus généralement: dans un job long avec plusieurs étapes
indépendantes, sauvegarder les résultats intermédiaires après chaque étape, pas seulement
à la fin.

## 13. `k_ablation_job` : même leçon que §12 (OOM sans sauvegarde incrémentale), reproduite

**Symptôme** : OOM CUDA à k=10 (`Tried to allocate 4.00 GiB`, batch_size=8 fixe quel que
soit k), qui a fait perdre les résultats de k=1 et k=3 déjà calculés avec succès, faute
de sauvegarde intermédiaire — exactement le problème du §12, corrigé une fois puis
reproduit dans un script différent.

**Diagnostic** : chaque script Kaggle de ce projet est volontairement autonome (pas
d'import du package `src/` local, pour rester poussable indépendamment) — un correctif
appliqué à un script n'est donc jamais propagé automatiquement aux autres.

**Solution** : `batch_size` désormais adapté par valeur de `k` (8 pour k=1/3, 4 pour
k=10) + retry automatique avec batch réduit en cas d'OOM + sauvegarde du JSON de
résultats après chaque `k`, pas seulement à la fin.

**Recommandation** : sur un ensemble de scripts autonomes dupliqués par nécessité
(contrainte de déploiement, pas par choix), tenir une checklist explicite des correctifs
"universels" (sauvegarde incrémentale, gestion OOM) à réappliquer partout, plutôt que de
supposer qu'un correctif isolé suffit.

## 14. Identifiants d'article BSARD : suffixe d'annotation parasite sur ~2,5% du corpus

**Symptôme** : en creusant plus loin le problème d'hétérogénéité de format du §9, ~555
articles sur 22 633 ont un identifiant de la forme `"1714bis_REGION_DE_BRUXELLES-CAPITALE"`
ou `"275_DROIT_FUTUR"` — un suffixe de note (variante régionale/temporelle) collé
directement à l'identifiant par le scraping de la source, sans séparateur.

**Impact potentiel si non corrigé** : (1) la cible d'entraînement du fine-tuning apprenait
à générer des citations non-naturelles (`"Article 1714bis_REGION_DE_BRUXELLES-CAPITALE"`)
qu'aucun humain n'écrirait ; (2) les métriques de citation (exact match, hallucination)
pénalisaient un modèle qui cite correctement l'article sous sa forme lisible
(`"Article 1714bis"`), car cette forme ne matche jamais le "gold" contaminé. 16 questions
sur les 222 du split test sont concernées (7,2%) — pas négligeable.

**Vérification faite avant de généraliser une hypothèse** : ce défaut n'est PAS corrélé à
la juridiction fédéral/régional (contrairement à une première intuition) — le fédéral a
même plus d'identifiants "structurés" en absolu (5552) que le régional (4731). L'étude de
la régularité de format (`src/citation_format_analysis.py`) est donc un axe d'analyse
indépendant de la juridiction, pas redondant.

**Solution** : `ARTICLE_ID_ANNOTATION_SUFFIX_REGEX = r"_[A-Z][A-Z_\-]+$"`, appliqué
systématiquement après extraction de l'identifiant, avant tout usage comme citation gold
(cible d'entraînement ou métrique). Reste 7 cas limites non couverts (ex.
`"D_VIII.27.2"`, un préfixe légitime, ou un cas de contamination du champ référence par
du texte d'article) — documentés mais non corrigés, l'effort de les gérer un par un
n'étant pas justifié pour 0,03% du corpus.

**Recommandation** : sur un corpus scrapé automatiquement, ne jamais faire confiance à un
champ "identifiant" sans vérifier qu'il ne contient QUE l'identifiant — chercher
spécifiquement les motifs de contamination (mots en majuscules, séparateurs inhabituels,
longueur anormale) avant de l'utiliser comme cible d'entraînement ou de comparaison exacte.

## 15. Fine-tuning QLoRA sans split de validation

**Constat** : jusqu'ici, le job de fine-tuning ne suivait que la loss d'entraînement
finale — aucun jeu de validation n'était réservé. Sur 580-886 exemples et 3 epochs,
impossible de distinguer un modèle qui généralise d'un modèle qui mémorise, et impossible
de comparer objectivement l'ablation cibles LoRA (attention seule vs attention+MLP, qui
ajoute de la capacité — donc du risque de surapprentissage sur peu de données).

**Solution** : 100 questions réservées avec un seed fixe (999, indépendant du seed
d'entraînement) — disjointes et identiques quelle que soit la taille de train ou le seed
testés, pour rester comparables sur toute la courbe d'apprentissage. `SFTConfig` évalue
désormais à chaque epoch (`eval_strategy="epoch"`) et la courbe train/eval complète est
persistée dans `finetune_meta_{run_tag}.json` (`train_loss_history`, `eval_loss_history`,
`overfit_gap`).

**Recommandation** : sur un fine-tuning à très peu d'exemples, ne jamais se fier à la
seule loss d'entraînement finale — réserver une validation fixe dès le premier run, même
petite, pour que la comparaison entre configurations (rang LoRA, cibles, taille de train)
soit sur un signal de généralisation, pas de mémorisation.

<!-- Entrées suivantes à compléter au fil des prochains runs (génération, LLM-as-judge,
     arène humaine). -->
