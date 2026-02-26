"""Utility modules for architecture deep dive.

- model_inspector: Count params, list layers, extract config
- attention_tools: Extract and process attention weights
- plotting: Matplotlib/seaborn visualization helpers
"""

from src.architecture_deepdive.utils import (
    attention_tools,
    model_inspector,
    plotting,
)

__all__ = [
    "attention_tools",
    "model_inspector",
    "plotting",
]
