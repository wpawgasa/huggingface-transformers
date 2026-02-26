"""Visualization helpers for attention matrices, parameter charts, etc.

Outputs saved as PNG to results/figures/.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

matplotlib.use("Agg")  # non-interactive backend

FIGURE_DIR = Path("results/figures")


def _ensure_figure_dir() -> None:
    """Create the figure output directory if it doesn't exist."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def plot_attention_matrix(
    attention: np.ndarray,
    tokens: list,
    title: str = "Attention Weights",
    save_name: str = "attention_matrix.png",
    layer: int = 0,
    head: int = 0,
) -> None:
    """Plot a single attention head's weight matrix as a heatmap.

    Rows = query tokens, Columns = key tokens.

    Args:
        attention: Array of shape (layers, heads, seq, seq).
        tokens: Token strings for axis labels.
        title: Plot title.
        save_name: Filename to save under FIGURE_DIR.
        layer: Which layer to plot.
        head: Which head to plot.
    """
    _ensure_figure_dir()
    attn = attention[layer, head]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        annot=len(tokens) <= 12,
        fmt=".2f",
        ax=ax,
    )
    ax.set_title(f"{title}\nLayer {layer}, Head {head}")
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending from)")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / save_name, dpi=150)
    plt.close()


def plot_attention_mask_comparison(
    bidirectional: np.ndarray,
    causal: np.ndarray,
    tokens: list,
    save_name: str = "mask_comparison.png",
) -> None:
    """Side-by-side plot of bidirectional vs causal attention masks.

    Directly illustrates the course's encoder vs decoder distinction.

    Args:
        bidirectional: Full attention mask array.
        causal: Lower-triangular causal mask array.
        tokens: Token labels for axes.
        save_name: Filename to save under FIGURE_DIR.
    """
    _ensure_figure_dir()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, mask, mask_title in zip(
        axes,
        [bidirectional, causal],
        ["Bidirectional (Encoder / BERT)", "Causal (Decoder / GPT)"],
        strict=True,
    ):
        sns.heatmap(
            mask,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
        )
        ax.set_title(mask_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")

    plt.suptitle(
        "Attention Mask Patterns — The Key Architectural Difference",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / save_name, dpi=150, bbox_inches="tight")
    plt.close()


def plot_parameter_comparison(
    model_data: list[dict],
    save_name: str = "parameter_comparison.png",
) -> None:
    """Bar chart comparing parameter counts across architecture families.

    Args:
        model_data: List of dicts with "name", "family", "params" keys.
        save_name: Filename to save under FIGURE_DIR.
    """
    _ensure_figure_dir()
    colors = {
        "encoder_only": "#4CAF50",
        "decoder_only": "#2196F3",
        "encoder_decoder": "#FF9800",
    }
    names = [d["name"] for d in model_data]
    params = [d["params"] / 1e6 for d in model_data]
    bar_colors = [colors.get(d["family"], "#999") for d in model_data]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(names, params, color=bar_colors, edgecolor="white")
    ax.set_xlabel("Parameters (Millions)")
    ax.set_title("Parameter Count by Architecture Family")

    for bar, p in zip(bars, params, strict=True):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{p:.1f}M",
            va="center",
            fontsize=9,
        )

    legend_elements = [
        Patch(facecolor=c, label=label)
        for label, c in [
            ("Encoder-only", "#4CAF50"),
            ("Decoder-only", "#2196F3"),
            ("Encoder-decoder", "#FF9800"),
        ]
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / save_name, dpi=150)
    plt.close()


def plot_transfer_learning_comparison(
    pretrained_scores: list,
    scratch_scores: list,
    epochs: list,
    save_name: str = "transfer_learning.png",
) -> None:
    """Learning curves: pretrained vs from-scratch fine-tuning.

    Illustrates Ch1.4's transfer learning concept.

    Args:
        pretrained_scores: Accuracy per epoch for pretrained model.
        scratch_scores: Accuracy per epoch for from-scratch model.
        epochs: Epoch numbers.
        save_name: Filename to save under FIGURE_DIR.
    """
    _ensure_figure_dir()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        epochs,
        pretrained_scores,
        "o-",
        color="#4CAF50",
        label="Pretrained + fine-tuned",
        linewidth=2,
    )
    ax.plot(
        epochs,
        scratch_scores,
        "s--",
        color="#F44336",
        label="Trained from scratch",
        linewidth=2,
    )

    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title("Transfer Learning: Pretrained vs From-Scratch\n" "(Ch1.4 Core Concept)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / save_name, dpi=150)
    plt.close()
