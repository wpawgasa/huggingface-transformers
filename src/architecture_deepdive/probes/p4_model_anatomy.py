"""Probe 4: General Transformer Architecture -- Model Anatomy.

Reference: Ch1.4 -- "General Transformer architecture" + "Architectures vs. checkpoints"

From the course:
  "Architecture: the skeleton of the model -- the definition of each layer
   and each operation that happens within the model."
  "Checkpoints: the weights that will be loaded in a given architecture."

This probe inspects one representative model from each family:
  - BERT (encoder-only)
  - GPT-2 (decoder-only)
  - T5 (encoder-decoder)
"""

from src.architecture_deepdive.data import MODEL_REGISTRY
from src.architecture_deepdive.utils.model_inspector import (
    format_param_count,
    inspect_model,
)
from src.architecture_deepdive.utils.plotting import plot_parameter_comparison


def run_experiment(device: str = "cpu") -> dict:
    """Inspect model anatomy across all three architecture families.

    Args:
        device: Device string (unused for this probe, kept for API consistency).

    Returns:
        Dict with per-family anatomy details and cross-architecture comparison.
    """
    results: dict = {"task": "model_anatomy"}

    model_data_for_plot = []

    for family_key, info in MODEL_REGISTRY.items():
        model_name = info["primary"]
        anatomy = inspect_model(model_name, family_key)

        results[family_key] = {
            "model": model_name,
            "architecture_class": anatomy.architecture_class,
            "family_description": info["family"],
            "pretraining_objective": info["objective"],
            "attention_type": info["attention"],
            "parameters": format_param_count(anatomy.num_parameters),
            "num_parameters_raw": anatomy.num_parameters,
            "num_layers": anatomy.num_layers,
            "hidden_size": anatomy.hidden_size,
            "num_attention_heads": anatomy.num_attention_heads,
            "intermediate_size": anatomy.intermediate_size,
            "vocab_size": anatomy.vocab_size,
            "max_position_embeddings": anatomy.max_position_embeddings,
            "has_encoder": anatomy.has_encoder,
            "has_decoder": anatomy.has_decoder,
            "layer_breakdown": {
                k: {
                    "num_weight_tensors": v["count"],
                    "total_params": format_param_count(v["params"]),
                }
                for k, v in anatomy.layer_breakdown.items()
            },
        }

        model_data_for_plot.append(
            {
                "name": model_name,
                "family": family_key,
                "params": anatomy.num_parameters,
            }
        )

    # Generate parameter comparison chart
    plot_parameter_comparison(model_data_for_plot)

    # Cross-architecture comparison
    results["comparison"] = {
        "architecture_vs_checkpoint_note": (
            "BERT is an *architecture*. bert-base-uncased is a *checkpoint*."
            " Both GPT-2 and gpt2 refer to different things: one is the"
            " architecture design, the other is a specific set of trained"
            " weights."
        ),
        "structural_comparison": {
            "encoder_only (BERT)": {
                "components": ("Embeddings -> N x [Self-Attention + FFN] -> Pooler"),
                "attention_mask": "Full (bidirectional)",
                "output": "Contextualized token representations",
            },
            "decoder_only (GPT-2)": {
                "components": ("Embeddings -> N x [Masked Self-Attention + FFN]" " -> LM Head"),
                "attention_mask": "Causal (lower triangular)",
                "output": "Next-token logits",
            },
            "encoder_decoder (T5)": {
                "components": (
                    "Encoder: Embeddings -> N x [Self-Attention + FFN]\n"
                    "Decoder: Embeddings -> N x [Self-Attn + Cross-Attn"
                    " + FFN] -> LM Head"
                ),
                "attention_mask": (
                    "Encoder: bidirectional, Decoder: causal" " + cross-attention to encoder"
                ),
                "output": "Generated sequence",
            },
        },
    }

    return results
