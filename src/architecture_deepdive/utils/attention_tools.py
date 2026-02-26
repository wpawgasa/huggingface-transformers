"""Extract, process, and prepare attention weights for visualization.

Core utility for Probe 5 (Attention Visualization).

Reference:
  - Ch1.4: "Attention layers tell the model to pay specific attention
    to certain words and more or less ignore the others"
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def extract_attention_weights(
    model_name: str,
    text: str,
    device: str = "cpu",
) -> dict:
    """Run a forward pass and extract attention weights from all layers/heads.

    Args:
        model_name: HuggingFace model checkpoint name.
        text: Input text to process.
        device: Device string ("cpu" or "cuda").

    Returns:
        Dict with keys: tokens, attentions (ndarray), num_layers, num_heads,
        seq_len.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.to(device).eval()

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions is a tuple of (num_layers,) tensors
    # each tensor: (batch=1, num_heads, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions).squeeze(1)  # (L, H, S, S)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return {
        "tokens": tokens,
        "attentions": attentions.cpu().numpy(),
        "num_layers": attentions.shape[0],
        "num_heads": attentions.shape[1],
        "seq_len": attentions.shape[2],
    }


def get_attention_to_token(
    attentions: np.ndarray,
    tokens: list,
    target_token: str,
    layer: int = -1,
) -> dict:
    """For a given target token, get how much attention each other token pays to it.

    This directly tests the course's claim:
      "the model pays specific attention to certain words"

    Args:
        attentions: Array of shape (layers, heads, seq, seq).
        tokens: List of token strings.
        target_token: The token to analyze.
        layer: Which layer (-1 = last).

    Returns:
        Dict mapping each token to its attention weight toward target_token.
    """
    target_idx = None
    for i, tok in enumerate(tokens):
        if target_token.lower() in tok.lower():
            target_idx = i
            break

    if target_idx is None:
        return {"error": f"Token '{target_token}' not found in {tokens}"}

    # Average across heads, select layer
    attn_matrix = attentions[layer].mean(axis=0)  # (seq, seq)
    # Column = how much attention flows TO target_idx
    attention_to_target = attn_matrix[:, target_idx]

    return {tok: round(float(attention_to_target[i]), 4) for i, tok in enumerate(tokens)}


def compare_causal_vs_bidirectional_mask(seq_len: int) -> dict:
    """Generate and compare attention masks for causal and bidirectional models.

    Directly illustrates the key architectural difference from Ch1.4.

    Args:
        seq_len: Sequence length to generate masks for.

    Returns:
        Dict with bidirectional mask, causal mask, difference, and stats.
    """
    # Bidirectional: full attention (encoder / BERT)
    bidirectional_mask = np.ones((seq_len, seq_len))

    # Causal: lower triangular (decoder / GPT)
    causal_mask = np.tril(np.ones((seq_len, seq_len)))

    return {
        "bidirectional": bidirectional_mask,
        "causal": causal_mask,
        "difference": bidirectional_mask - causal_mask,
        "causal_masked_positions": int(np.sum(bidirectional_mask - causal_mask)),
        "note": (
            f"For seq_len={seq_len}: bidirectional has {seq_len**2} attention"
            f" pairs, causal has {seq_len * (seq_len + 1) // 2}"
            " (upper triangle masked)"
        ),
    }
