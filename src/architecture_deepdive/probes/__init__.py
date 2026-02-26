"""Architecture deep dive probe modules.

Each module implements one probe from the HuggingFace LLM Course Chapter 1.4
and exports a run_experiment(device) -> dict function.
"""

from src.architecture_deepdive.probes import (
    p1_model_timeline,
    p2_language_modeling,
    p3_transfer_learning,
    p4_model_anatomy,
    p5_attention_viz,
    p6_arch_comparison,
)

__all__ = [
    "p1_model_timeline",
    "p2_language_modeling",
    "p3_transfer_learning",
    "p4_model_anatomy",
    "p5_attention_viz",
    "p6_arch_comparison",
]
