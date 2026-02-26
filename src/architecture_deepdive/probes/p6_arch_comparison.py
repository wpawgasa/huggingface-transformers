"""Probe 6: Encoder vs Decoder vs Encoder-Decoder Comparison.

Reference: Ch1.4 -- "General Transformer architecture"

From the course:
  "Encoder-only models: Good for tasks that require understanding of the input"
  "Decoder-only models: Good for generative tasks such as text generation"
  "Encoder-decoder models: Good for generative tasks that require an input"

This probe runs all three architecture types on the same input and
compares their hidden state representations, output formats, and
task-specific behaviors.
"""

import numpy as np
import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from src.architecture_deepdive.data import MODEL_REGISTRY

SHARED_INPUT = "The Transformer architecture was introduced in 2017."


def probe_encoder_only(device: str = "cpu") -> dict:
    """BERT (Encoder-Only): Produces contextualized token embeddings.

    Bidirectional attention -- every token sees every other token.
    Best for: classification, NER, QA.

    Args:
        device: Device string.

    Returns:
        Dict with model info, hidden state shape, attention analysis.
    """
    model_name = MODEL_REGISTRY["encoder_only"]["primary"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.to(device).eval()

    inputs = tokenizer(SHARED_INPUT, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state  # (1, seq_len, hidden_size)
    attentions = outputs.attentions  # tuple of (1, heads, seq, seq)

    # Verify bidirectional: attention matrix should be full (no zeros)
    attn_matrix = attentions[0][0, 0].cpu().numpy()  # layer 0, head 0
    is_bidirectional = bool(np.all(attn_matrix > 0))

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return {
        "model": model_name,
        "architecture": "encoder-only",
        "hidden_state_shape": list(hidden.shape),
        "num_tokens": len(tokens),
        "tokens": tokens,
        "hidden_dim": hidden.shape[-1],
        "attention_is_bidirectional": is_bidirectional,
        "attention_matrix_shape": list(attn_matrix.shape),
        "output_type": "Contextualized embeddings per token",
        "typical_use": ("Add task-specific head (classifier, QA span predictor)"),
        "cls_embedding_norm": round(float(hidden[0, 0].norm()), 4),
    }


def probe_decoder_only(device: str = "cpu") -> dict:
    """GPT-2 (Decoder-Only): Generates next tokens autoregressively.

    Causal attention -- each token only sees past tokens.
    Best for: text generation.

    Args:
        device: Device string.

    Returns:
        Dict with model info, logits shape, causal verification, predictions.
    """
    model_name = MODEL_REGISTRY["decoder_only"]["primary"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
    model.to(device).eval()

    inputs = tokenizer(SHARED_INPUT, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # (1, seq_len, vocab_size)
    attentions = outputs.attentions  # tuple of (1, heads, seq, seq)

    # Verify causal: upper triangle of attention should be ~0
    attn_matrix = attentions[0][0, 0].cpu().numpy()
    upper_triangle_sum = float(np.triu(attn_matrix, k=1).sum())
    is_causal = upper_triangle_sum < 1e-6

    # Predict next token
    next_token_logits = logits[0, -1, :]
    probs = torch.softmax(next_token_logits, dim=-1)
    top5_probs, top5_ids = torch.topk(probs, 5)
    top5_tokens = [tokenizer.decode(tid).strip() for tid in top5_ids]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return {
        "model": model_name,
        "architecture": "decoder-only",
        "logits_shape": list(logits.shape),
        "num_tokens": len(tokens),
        "tokens": tokens,
        "vocab_size": logits.shape[-1],
        "attention_is_causal": is_causal,
        "upper_triangle_attn_sum": round(upper_triangle_sum, 8),
        "next_token_predictions": [
            {"token": t, "prob": round(p.item(), 4)}
            for t, p in zip(top5_tokens, top5_probs, strict=True)
        ],
        "output_type": "Next-token probability distribution",
        "typical_use": "Autoregressive text generation",
    }


def probe_encoder_decoder(device: str = "cpu") -> dict:
    """T5 (Encoder-Decoder): Encodes input, then generates output sequence.

    Encoder has bidirectional attention, decoder has causal + cross-attention.
    Best for: translation, summarization, question answering.

    Args:
        device: Device string.

    Returns:
        Dict with model info, encoder hidden shape, generated text.
    """
    model_name = MODEL_REGISTRY["encoder_decoder"]["primary"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device).eval()

    # T5 uses "text-to-text" format
    input_text = f"summarize: {SHARED_INPUT}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=30,
            num_beams=2,
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Inspect encoder outputs
    encoder_outputs = model.get_encoder()(**inputs)
    encoder_hidden = encoder_outputs.last_hidden_state

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return {
        "model": model_name,
        "architecture": "encoder-decoder",
        "encoder_hidden_shape": list(encoder_hidden.shape),
        "num_input_tokens": len(tokens),
        "tokens": tokens,
        "generated_text": generated_text,
        "generated_token_count": len(generated_ids[0]),
        "output_type": ("Generated sequence (encoder context -> decoder generation)"),
        "typical_use": ("Translation, summarization, text-to-text tasks"),
        "key_feature": (
            "Decoder has TWO attention layers per block: "
            "(1) causal self-attention over generated tokens, "
            "(2) cross-attention over encoder hidden states. "
            "This allows the decoder to 'look at' the full input."
        ),
    }


def run_experiment(device: str = "cpu") -> dict:
    """Run all three architecture probes and compare.

    Args:
        device: Device string ("cpu" or "cuda").

    Returns:
        Dict with per-architecture results and synthesis.
    """
    results: dict = {"task": "architecture_comparison"}

    results["encoder_only"] = probe_encoder_only(device)
    results["decoder_only"] = probe_decoder_only(device)
    results["encoder_decoder"] = probe_encoder_decoder(device)

    results["shared_input"] = SHARED_INPUT

    results["synthesis"] = {
        "attention_patterns": {
            "encoder_only": ("Full bidirectional matrix" " (every token sees every token)"),
            "decoder_only": ("Lower-triangular causal matrix" " (only see past tokens)"),
            "encoder_decoder": ("Encoder=bidirectional," " Decoder=causal+cross-attention"),
        },
        "output_format": {
            "encoder_only": "Per-token embeddings (needs task head)",
            "decoder_only": ("Next-token logits (autoregressive generation)"),
            "encoder_decoder": "Generated token sequence",
        },
        "task_suitability": {
            "encoder_only": ("Classification, NER, QA, sentence similarity"),
            "decoder_only": ("Text generation, code completion, chatbots"),
            "encoder_decoder": ("Translation, summarization, question answering"),
        },
        "course_quote": (
            "Each of these parts can be used independently," " depending on the task."
        ),
    }

    return results
