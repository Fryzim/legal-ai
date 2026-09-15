"""QLoRA fine-tuning de Mistral-7B-Instruct-v0.3 sur BSARD.

Conçu pour tourner sur GPU (Kaggle T4 16Go), cf. kaggle_kernels/finetune_job.py.
Configuration alignée sur les comptes-rendus déjà envoyés (r=32, alpha=32, NF4 4-bit,
3 epochs) pour que les résultats reproduits restent comparables (r=32 corrige le
29/07, la référence historique 580 exemples utilisait r=32 pas r=16, cf.
contexte/chaab1.pdf).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import config


def build_training_examples(questions_df, article_lookup, n_examples: int = config.FINETUNE_TRAIN_SIZE, seed: int = config.SEED):
    from . import data, generation

    df = questions_df.sample(n=min(n_examples, len(questions_df)), random_state=seed).reset_index(drop=True)
    examples = []
    for _, row in df.iterrows():
        ref_answer = data.build_reference_answer(row["article_ids"], article_lookup)
        article_refs = []
        for aid in row["article_ids"]:
            art = article_lookup.get(aid)
            if art is None:
                continue
            m = re.search(config.ARTICLE_REFERENCE_REGEX, str(art["reference"]))
            if m:
                article_refs.append(config.clean_article_ref_id(m.group(1).strip()))
        examples.append(generation.format_finetune_example(row["question"], ref_answer, article_refs))
    return examples


def load_base_model_4bit(model_name: str = config.MODEL_MISTRAL_BASE):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.QLORA_QUANT_TYPE,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto"
    )
    return model, tokenizer


def build_lora_model(base_model, r: int = config.LORA_R, alpha: int = config.LORA_ALPHA, target_modules=None):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    base_model = prepare_model_for_kbit_training(base_model)
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules or config.LORA_TARGET_MODULES,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    pct = 100 * trainable / total
    return model, {"trainable_params": trainable, "total_params": total, "pct_trainable": pct}


def run_finetune(
    train_examples: list[dict],
    output_dir: str,
    seed: int = config.SEED,
    epochs: int = config.FINETUNE_EPOCHS,
    r: int = config.LORA_R,
    alpha: int = config.LORA_ALPHA,
) -> dict:
    """Lance le fine-tuning QLoRA et retourne un dict de métadonnées (durée, config, etc)."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    config.set_all_seeds(seed)
    t0 = time.time()

    model, tokenizer = load_base_model_4bit()
    model, lora_stats = build_lora_model(model, r=r, alpha=alpha)

    def format_fn(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}

    ds = Dataset.from_list(train_examples).map(format_fn)

    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=config.FINETUNE_BATCH_SIZE,
        gradient_accumulation_steps=config.FINETUNE_GRAD_ACCUM,
        learning_rate=config.FINETUNE_LR,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        seed=seed,
        report_to=[],
        dataset_text_field="text",
        max_length=1024,
        loss_type="nll",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=ds,
    )
    train_result = trainer.train()
    trainer.save_model(output_dir)

    duration_s = time.time() - t0
    meta = {
        "seed": seed,
        "epochs": epochs,
        "lora_r": r,
        "lora_alpha": alpha,
        "n_train_examples": len(train_examples),
        "trainable_params": lora_stats["trainable_params"],
        "total_params": lora_stats["total_params"],
        "pct_trainable": lora_stats["pct_trainable"],
        "duration_seconds": duration_s,
        "final_train_loss": train_result.training_loss,
        "output_dir": output_dir,
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "finetune_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_finetuned_model(base_model_name: str, adapter_dir: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.QLORA_QUANT_TYPE,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config, device_map="auto")
    model = PeftModel.from_pretrained(base, adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    return model, tokenizer
