"""Probe 2: Language Modeling -- Causal (CLM) vs Masked (MLM).

Reference: Ch1.4 -- "Transformers are language models"

Demonstrates the two pretraining objectives:
  - Causal LM (GPT): "predicting the next word having read the n previous words"
  - Masked LM (BERT): "predicts a masked word in the sentence"

Key investigation: How does context directionality affect predictions?
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from src.architecture_deepdive.data import LM_INPUTS


def run_causal_lm_probe(model_name: str = "gpt2", device: str = "cpu") -> dict:
    """Causal Language Modeling: predict the next token left-to-right.

    From the course:
      "This is called causal language modeling because the output depends
       on the past and present inputs, but not the future ones."

    Args:
        model_name: HuggingFace model checkpoint name.
        device: Device string.

    Returns:
        Dict with model info and per-prompt top-5 predictions.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device).eval()

    results = []
    for prompt in LM_INPUTS["causal_prompts"]:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # logits shape: (1, seq_len, vocab_size)
            # Last position predicts the next token
            next_token_logits = outputs.logits[0, -1, :]
            probs = torch.softmax(next_token_logits, dim=-1)

            # Top 5 predictions
            top5_probs, top5_ids = torch.topk(probs, 5)
            top5_tokens = [tokenizer.decode(tid) for tid in top5_ids]

        results.append(
            {
                "prompt": prompt,
                "top_5_predictions": [
                    {
                        "token": tok.strip(),
                        "probability": round(prob.item(), 4),
                    }
                    for tok, prob in zip(top5_tokens, top5_probs, strict=True)
                ],
                "context_direction": "left-to-right only (causal mask)",
            }
        )

    return {
        "model": model_name,
        "objective": "Causal LM (CLM)",
        "results": results,
    }


def run_masked_lm_probe(model_name: str = "bert-base-uncased", device: str = "cpu") -> dict:
    """Masked Language Modeling: predict masked tokens using FULL context.

    From the course:
      "Another example is masked language modeling, in which the model
       predicts a masked word in the sentence."

    Args:
        model_name: HuggingFace model checkpoint name.
        device: Device string.

    Returns:
        Dict with model info and per-sentence top-5 predictions.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.to(device).eval()

    results = []
    for sentence in LM_INPUTS["masked_sentences"]:
        inputs = tokenizer(sentence, return_tensors="pt").to(device)

        # Find [MASK] position
        mask_token_id = tokenizer.mask_token_id
        mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[1]

        with torch.no_grad():
            outputs = model(**inputs)

            for pos in mask_positions:
                mask_logits = outputs.logits[0, pos, :]
                probs = torch.softmax(mask_logits, dim=-1)
                top5_probs, top5_ids = torch.topk(probs, 5)
                top5_tokens = [tokenizer.decode(tid).strip() for tid in top5_ids]

                results.append(
                    {
                        "sentence": sentence,
                        "mask_position": pos.item(),
                        "top_5_predictions": [
                            {
                                "token": tok,
                                "probability": round(prob.item(), 4),
                            }
                            for tok, prob in zip(top5_tokens, top5_probs, strict=True)
                        ],
                        "context_direction": ("bidirectional" " (sees left AND right of [MASK])"),
                    }
                )

    return {
        "model": model_name,
        "objective": "Masked LM (MLM)",
        "results": results,
    }


def run_experiment(device: str = "cpu") -> dict:
    """Compare CLM vs MLM on equivalent prompts.

    Args:
        device: Device string ("cpu" or "cuda").

    Returns:
        Dict with causal_lm results, masked_lm results, and comparison.
    """
    results: dict = {"task": "language_modeling_comparison"}

    results["causal_lm"] = run_causal_lm_probe(device=device)
    results["masked_lm"] = run_masked_lm_probe(device=device)

    # Direct comparison on shared concept
    results["comparison_analysis"] = {
        "shared_test": ("Predicting 'France' in 'The capital of France is ___'"),
        "clm_approach": ("GPT sees: 'The capital of France is' -> predicts next token"),
        "mlm_approach": (
            "BERT sees: 'The capital of France is [MASK] .'"
            " -> predicts masked token using FULL sentence including '.'"
        ),
        "key_difference": (
            "CLM can only use LEFT context (past tokens). "
            "MLM uses BOTH left and right context. "
            "This is why BERT is better for understanding tasks "
            "and GPT is better for generation tasks."
        ),
    }

    return results
