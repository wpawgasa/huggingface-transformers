"""Utilities for inspecting transformer model internals.

Counts parameters, lists layers, extracts attention masks.

References:
  - Ch1.4: "Architecture: the skeleton of the model"
  - Ch1.4: "Checkpoints: the weights loaded in a given architecture"
"""

from collections import OrderedDict
from dataclasses import dataclass

from transformers import AutoConfig, AutoModel


@dataclass
class ModelAnatomy:
    """Complete architectural profile of a transformer model."""

    name: str
    architecture_class: str  # e.g., "BertModel", "GPT2Model"
    family: str  # encoder-only / decoder-only / encoder-decoder
    num_parameters: int
    num_trainable_parameters: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int  # FFN hidden dim
    vocab_size: int
    max_position_embeddings: int
    has_encoder: bool
    has_decoder: bool
    layer_breakdown: dict  # parameter count per layer type


def inspect_model(model_name: str, family: str) -> ModelAnatomy:
    """Load a model and extract its full architectural profile.

    Demonstrates Ch1.4's distinction between architecture and checkpoint.

    Args:
        model_name: HuggingFace model checkpoint name.
        family: "encoder_only", "decoder_only", or "encoder_decoder".

    Returns:
        ModelAnatomy dataclass with all extracted information.
    """
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Layer-by-layer parameter breakdown
    layer_breakdown: OrderedDict[str, dict] = OrderedDict()
    for name, param in model.named_parameters():
        top_module = name.split(".")[0]
        if top_module not in layer_breakdown:
            layer_breakdown[top_module] = {"count": 0, "params": 0}
        layer_breakdown[top_module]["count"] += 1
        layer_breakdown[top_module]["params"] += param.numel()

    # Extract config values (handles different config field names)
    num_layers = getattr(
        config,
        "num_hidden_layers",
        getattr(config, "n_layer", getattr(config, "num_layers", 0)),
    )
    hidden_size = getattr(
        config,
        "hidden_size",
        getattr(config, "n_embd", getattr(config, "d_model", 0)),
    )
    num_heads = getattr(
        config,
        "num_attention_heads",
        getattr(config, "n_head", getattr(config, "num_heads", 0)),
    )
    intermediate_size = getattr(
        config,
        "intermediate_size",
        getattr(config, "n_inner", getattr(config, "d_ff", 0)),
    )

    return ModelAnatomy(
        name=model_name,
        architecture_class=model.__class__.__name__,
        family=family,
        num_parameters=total_params,
        num_trainable_parameters=trainable_params,
        num_layers=num_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        intermediate_size=intermediate_size,
        vocab_size=config.vocab_size,
        max_position_embeddings=getattr(
            config,
            "max_position_embeddings",
            getattr(config, "n_positions", 0),
        ),
        has_encoder="encoder" in family,
        has_decoder="decoder" in family,
        layer_breakdown=dict(layer_breakdown.items()),
    )


def format_param_count(n: int) -> str:
    """Human-readable parameter count (e.g., 110M, 1.5B)."""
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)
