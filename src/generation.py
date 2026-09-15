"""Génération: construction de prompt (commun aux 4 configs C1-C4), inférence,
parsing de citation. Le modèle (base ou avec adaptateur LoRA) est chargé en amont
et passé à `generate_answer`.
"""
from __future__ import annotations

from . import config


def build_prompt(question: str, context_articles: list[dict] | None = None) -> str:
    """context_articles: liste de dicts {reference, article} récupérés par le retriever.
    None ou [] => pas de RAG (configs C1/C3).
    """
    parts = [config.SYSTEM_PROMPT_FR, ""]
    if context_articles:
        parts.append("Contexte juridique (articles potentiellement pertinents) :")
        for art in context_articles:
            parts.append(f"- {art['reference']} : {art['article']}")
        parts.append("")
    parts.append(f"Question : {question}")
    parts.append("Réponse :")
    return "\n".join(parts)


def build_chat_messages(question: str, context_articles: list[dict] | None = None) -> list[dict]:
    """Format chat pour tokenizer.apply_chat_template (Mistral-Instruct)."""
    user_content = []
    if context_articles:
        ctx = "\n".join(f"- {a['reference']} : {a['article']}" for a in context_articles)
        user_content.append(f"Contexte juridique :\n{ctx}\n")
    user_content.append(f"Question : {question}")
    return [
        {"role": "system", "content": config.SYSTEM_PROMPT_FR},
        {"role": "user", "content": "\n".join(user_content)},
    ]


def generate_answer(
    model,
    tokenizer,
    question: str,
    context_articles: list[dict] | None = None,
    max_new_tokens: int = config.GENERATION_MAX_NEW_TOKENS,
    temperature: float = config.GENERATION_TEMPERATURE,
) -> str:
    import torch

    messages = build_chat_messages(question, context_articles)
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(
        model.device
    )
    do_sample = temperature > 0.0
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
    with torch.no_grad():
        output_ids = model.generate(input_ids, **gen_kwargs)
    generated = output_ids[0][input_ids.shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def format_finetune_example(question: str, reference_answer: str, article_refs: list[str]) -> dict:
    """Format d'exemple d'entraînement QLoRA (question -> réponse + citation)."""
    citation = ", ".join(f"Article {r}" for r in article_refs) if article_refs else ""
    completion = f"{reference_answer}\n\n{citation}".strip()
    return {
        "messages": [
            {"role": "system", "content": config.SYSTEM_PROMPT_FR},
            {"role": "user", "content": f"Question : {question}"},
            {"role": "assistant", "content": completion},
        ]
    }
