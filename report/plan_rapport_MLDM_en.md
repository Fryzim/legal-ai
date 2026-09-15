# Report Plan — M1 MLDM (Chaabane Ouammou, supervisor A. Habrard)

Convention: each section lists figures/tables to insert (`figures/`, `tables/`) and
`.bib` references to cite (`report/references.bib`). Written in English, LaTeX/Overleaf.

## 1. Introduction
- Motivation, research questions (same three as the joint proposal)
- Contributions: extended experimental protocol vs. a single-experiment baseline
  (direct response to the June/July feedback), category-level analysis, bootstrap CIs

## 2. Related Work
- Parameter-efficient fine-tuning: LoRA (@hu2021lora), QLoRA (@dettmers2023qlora)
- Retrieval-Augmented Generation: @lewis2020rag, survey @gao2023ragsurvey
- RAG vs. fine-tuning comparisons, explicit positioning against @bouvard2024derbyllm
  (Derby LLM: same arena/faithfulness-style metrics, legal domain, extended protocol)
- Evaluation of generative QA systems: @zhang2020bertscore, @es2023ragas, @chiang2024chatbotarena
- Legal NLP benchmarks: @louis2022bsard (BSARD), @guha2023legalbench, @hendrycks2021cuad,
  @chalkidis2022lexglue
- Base models: @jiang2023mistral7b, @touvron2023llama2

## 3. Dataset and Task
- BSARD: verified structure (22,633 articles; splits 886/222/113,165 train/test/synthetic),
  native `category`/`subcategory` fields (7 legal categories) — directly enables the
  subgroup analysis suggested in supervisor feedback
- **Explicit limitation**: BSARD has no gold free-text answer; reference answers for
  ROUGE-L/BERTScore are reconstructed by concatenating cited article text
  (cf. DIFFICULTES.md #5) — stated as a methodological limitation, not hidden
- Justification for Belgian French-speaking law as an acceptable substitute for French law:
  the object of study is the *comparison of adaptation methods*, not legal content itself;
  French language is preserved; LLeQA access blocker documented

## 4. Method
### 4.1 Four configurations (C1-C4): zero-shot / RAG / fine-tune / fine-tune+RAG
- Shared generation prompt (fairness across configs) — Table `tables/config_description.tex`
### 4.2 Retrieval
- Dense (mpnet baseline, multilingual-e5-large), BM25, hybrid RRF, cross-encoder reranking
### 4.3 QLoRA fine-tuning
- r=16, alpha=32, NF4 4-bit, 3 epochs, 580 examples (config aligned with earlier progress
  reports for comparability)

## 5. Experimental Setup — Addressing "Single Experiment" Feedback
**This section directly answers the supervisor's June/July comment** ("comparisons may
appear limited... extend the experimental setup to make tests on different test
data/sampling to evaluate the average behavior"):
- Full 222-question test split (not a 50-question subsample) for all core metrics
- Bootstrap resampling (1,000 resamples) → 95% confidence intervals on every metric
- Paired bootstrap significance tests between all configuration pairs, Holm-Bonferroni
  correction for multiple comparisons
- Subgroup analysis by legal category (Famille, Logement, Argent, Justice, Étrangers,
  Travail, Protection sociale) — directly implements "you can focus on particular
  subgroups of laws"
- *(Reduced-scope note, P0 vs. P1)*: due to a compressed timeline, the main fine-tuning
  run uses 1 seed rather than 3; this is documented as a limitation, with a plan to extend
  to multi-seed replication before the final defense if GPU budget allows

## 6. Results
### 6.1 Retrieval evaluation — Fig. `figures/recall_at_k_comparison.pdf`
### 6.2 Generation quality with confidence intervals — Fig. `figures/config_comparison_ci.pdf`
### 6.3 Citation accuracy and hallucination rate — `tables/citation_hallucination.tex`
### 6.4 Category-level breakdown — Fig. `figures/heatmap_category.pdf`
### 6.5 Abstention capability — `tables/abstention.tex`
### 6.6 Practical cost comparison — `tables/cost_comparison.tex`
### 6.7 Ablations (P1): retrieval variants, k, chunking, LoRA rank, learning curve
### 6.8 Human evaluation arena (P1): inter-annotator agreement (Cohen's Kappa)

## 7. Error Analysis
- 9-category taxonomy (hallucinated citation, wrong-but-existing article, jurisdiction
  confusion, missing citation, missing/excessive abstention, language error, incomplete
  answer, degeneration) — matrix error-type × configuration
- 5-8 annotated qualitative examples

## 8. Discussion
- Answering the three research questions with evidence
- Explicit comparison with Derby LLM's findings (@bouvard2024derbyllm)
- Limitations: Belgian (not French) law corpus, reconstructed references, reduced seed count

## 9. Conclusion and Future Work
- Multi-seed replication, extended ablations, second base model (Qwen2.5-7B) for
  generalization check — as time/GPU budget allows post-deadline

## Appendix A — Technical Difficulties (DIFFICULTES.md, practical library recommendations format)
Per supervisor's explicit request: documented as practical recommendations, not narrative.

## Appendix B — Implementation Details
Full QLoRA hyperparameters, prompts, kernel-level reproducibility notes (seeds, versions)
