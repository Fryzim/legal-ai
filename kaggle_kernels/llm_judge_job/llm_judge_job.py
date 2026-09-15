"""Kaggle/RunPod (GPU): LLM-as-judge (Phi-3.5-mini-instruct, local, gratuit)
notant la PERTINENCE des réponses des 4 configurations, sur un sous-échantillon
de 50 questions (les évaluations coûteuses type LLM-judge sont volontairement
limitées à ce sous-ensemble, cf. brief §4.5-4.6 — le Recall@k et ROUGE-L/
BERTScore restent calculés sur les 222 questions complètes ailleurs).

Ne juge PAS la fidélité (contrairement à la version précédente): un LLM-juge
est un outil coûteux et non-déterministe, à réserver à ce qu'aucune métrique
déterministe ne peut mesurer. La fidélité a une métrique déterministe dédiée,
inspirée de Derby LLM (Bouvard et al., APIA@PFIA 2024) -- recouvrement des
"passages d'intérêt" (entités nommées, nombres, emails, URLs) entre réponse et
texte de référence, cf. `src/metrics.py::fidelity_score` et
`src/fidelity_analysis.py`. Reproduit leur choix méthodologique explicite:
juger par LLM UNIQUEMENT ce qui est intrinsèquement subjectif (la pertinence),
et mesurer par une règle déterministe ce qui peut l'être (la fidélité) --
plutôt que de dupliquer un même jugement de deux façons différentes.

Pour la même raison qu'eux (un LLM-juge est non-déterministe), la pertinence
est échantillonnée N_JUDGE_SAMPLES fois par réponse (température > 0) et
moyennée, au lieu d'un jugement greedy unique.

Dépend du kernel ter-bsard-generation-eval (kernel_sources).
Sortie: /kaggle/working/llm_judge_results.json
"""
import glob
import json
import re
import subprocess
import os
import sys
import time
from pathlib import Path

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U", "bitsandbytes", "accelerate"],
    check=True,
)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Chemins portables: par defaut Kaggle, surchargeables via variables
# d'environnement pour tourner ailleurs (RunPod, etc.) sans modifier le script.
WORK_DIR = os.environ.get("WORK_DIR", "/kaggle/working")
INPUT_DIR = os.environ.get("INPUT_DIR", "/kaggle/input")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Qwen2.5-7B-Instruct est aussi testé comme second modèle de base
# (other_llm_job, généralisation des conclusions RAG vs FT) -- il ne peut donc
# pas être le juge sans biais d'auto-préférence. Phi-3.5-mini-instruct: non
# gated (licence MIT), absent des 4 configurations comparées et du second
# modèle de base testé.
JUDGE_MODEL = "microsoft/Phi-3.5-mini-instruct"
N_JUDGED_QUESTIONS = 50
N_JUDGE_SAMPLES = 10  # meme ordre de grandeur que Derby LLM (10 echantillons pour la pertinence)
JUDGE_SAMPLING_TEMPERATURE = 0.7
MAX_NEW_TOKENS = 150

JUDGE_PROMPT_TEMPLATE = """Tu es un juge expert en droit belge, chargé d'évaluer la qualité d'une réponse générée par un assistant juridique.

Question posée : {question}

Réponse générée par l'assistant à évaluer :
{answer}

Évalue si la réponse répond effectivement à la question posée (pertinence), sur une échelle de 1 (très mauvais) à 5 (excellent). Ne juge PAS l'exactitude factuelle du contenu, uniquement si la réponse traite bien ce qui est demandé.

Réponds UNIQUEMENT avec un objet JSON de la forme :
{{"pertinence": <entier 1-5>, "justification": "<une phrase courte>"}}
"""


def find_file(pattern):
    matches = glob.glob(f"{INPUT_DIR}/**/{pattern}", recursive=True)
    if not matches:
        raise FileNotFoundError(f"No match for {pattern} under {INPUT_DIR}")
    return matches[0]


gen_results_path = find_file("generation_results.json")
with open(gen_results_path) as f:
    gen = json.load(f)

questions = gen["questions"]
n_total = len(questions)
assert n_total == 222

rng = np.random.default_rng(SEED)
judged_indices = sorted(rng.choice(n_total, size=min(N_JUDGED_QUESTIONS, n_total), replace=False).tolist())
print(f"Questions jugées: {len(judged_indices)} / {n_total}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(JUDGE_MODEL, quantization_config=bnb_config, device_map="auto")


def judge_one_sample(question, answer):
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, answer=answer)
    messages = [{"role": "user", "content": prompt}]
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=JUDGE_SAMPLING_TEMPERATURE,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group(0))
            return {"pertinence": int(parsed.get("pertinence", -1)), "justification": str(parsed.get("justification", "")), "raw": text, "parse_ok": True}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return {"pertinence": None, "justification": None, "raw": text, "parse_ok": False}


def judge_one(question, answer):
    """Repete le jugement N_JUDGE_SAMPLES fois (temperature > 0, comme Derby LLM
    pour leur pertinence) et retourne la moyenne + ecart-type -- un LLM-juge
    est non-deterministe, un jugement greedy unique est un point de mesure
    bruyant, pas une estimation robuste."""
    samples = [judge_one_sample(question, answer) for _ in range(N_JUDGE_SAMPLES)]
    valid = [s["pertinence"] for s in samples if s["pertinence"] is not None]
    return {
        "pertinence_mean": float(np.mean(valid)) if valid else None,
        "pertinence_std": float(np.std(valid)) if valid else None,
        "n_valid_samples": len(valid),
        "n_samples": N_JUDGE_SAMPLES,
        "samples": samples,
    }


results = {
    "seed": SEED,
    "judge_model": JUDGE_MODEL,
    "n_judged_questions": len(judged_indices),
    "n_judge_samples_per_response": N_JUDGE_SAMPLES,
    "judged_indices": judged_indices,
    "config_scores": {},
}

for cfg_name, cfg_data in gen["configs"].items():
    print(f"=== judging {cfg_name} (pertinence, {N_JUDGE_SAMPLES} echantillons/reponse) ===")
    t0 = time.time()
    answers = cfg_data["answers"]
    per_question = []
    for count, i in enumerate(judged_indices):
        judged = judge_one(questions[i], answers[i])
        judged["question_index"] = i
        per_question.append(judged)
        if count % 10 == 0:
            print(f"  {count}/{len(judged_indices)}")
    parse_ok_rate = float(np.mean([q["n_valid_samples"] / q["n_samples"] for q in per_question]))
    question_means = [q["pertinence_mean"] for q in per_question if q["pertinence_mean"] is not None]
    results["config_scores"][cfg_name] = {
        "per_question": per_question,
        "parse_ok_rate": parse_ok_rate,
        "pertinence_mean": float(np.mean(question_means)) if question_means else None,
        # ecart-type INTRA-reponse moyen: a quel point le juge est instable sur une meme reponse
        "mean_within_response_std": float(np.mean([q["pertinence_std"] for q in per_question if q["pertinence_std"] is not None])),
        "duration_seconds": time.time() - t0,
    }
    print(results["config_scores"][cfg_name]["pertinence_mean"], "within-response std:", results["config_scores"][cfg_name]["mean_within_response_std"])

results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

with open(f"{WORK_DIR}/llm_judge_results.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("DONE")
