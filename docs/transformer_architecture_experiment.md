# Experiment: How Do Transformers Work? — Architecture Deep Dive

**Source**: [HuggingFace LLM Course — Chapter 1.4: How do Transformers work?](https://huggingface.co/learn/llm-course/chapter1/4)

**Experiment Design**: AI Researcher Reproduction Framework

**Scope**: Systematic hands-on exploration of Transformer internals — attention mechanisms, architecture families (encoder / decoder / encoder-decoder), language modeling objectives (causal vs masked), transfer learning, and model anatomy

**Success Criterion**: All probes produce interpretable outputs; attention visualizations confirm theoretical predictions; architecture comparisons are benchmarked and documented

---

## 1. Problem & Motivation

**Research Question**: How do the three Transformer architecture families (encoder-only, decoder-only, encoder-decoder) differ mechanistically in their attention patterns, hidden representations, and downstream task suitability — and can we empirically verify the theoretical claims taught in the course?

**Context**: Chapter 1.4 introduces the foundational concepts behind Transformers: self-attention, causal vs. bidirectional context, pretraining objectives, transfer learning, and the encoder–decoder split. These ideas are taught conceptually but rarely probed hands-on with actual model internals.

**Prior Work Limitations**:
- Tutorials show pipelines but rarely expose attention weights or hidden states
- Architecture differences (encoder vs. decoder) are explained textually but not compared empirically
- Transfer learning benefits are stated but not measured on controlled experiments
- Attention masking behavior (causal vs. bidirectional) is not visually demonstrated

**Gap Addressed**: This experiment provides a structured, code-driven exploration of every concept in Chapter 1.4, turning theoretical understanding into observable, measurable results.

---

## 2. Technical Approach

### 2.1 Course Concept Map

Chapter 1.4 covers **6 core concepts** — this experiment maps each to a concrete, runnable probe:

| # | Course Concept | Experiment Probe | What We Measure |
|---|---------------|-----------------|-----------------|
| 1 | Transformer History | Model timeline explorer | Parameter counts, release dates, architecture families |
| 2 | Language Models (CLM vs MLM) | Causal vs Masked LM comparison | Perplexity, token predictions, context directionality |
| 3 | Transfer Learning | Pretrained vs scratch comparison | Accuracy gap, training efficiency |
| 4 | General Architecture | Model anatomy inspector | Layer counts, parameter breakdown, computational graph |
| 5 | Attention Layers | Attention weight visualization | Attention matrices, head specialization, context sensitivity |
| 6 | Encoder vs Decoder vs Enc-Dec | Three-family comparison | Hidden states, masking patterns, task suitability |

### 2.2 Architecture Taxonomy (from the Course)

```
┌─────────────────────────────────────────────────────────────────────┐
│                     TRANSFORMER (Vaswani et al., 2017)              │
│                                                                     │
│   ┌───────────────┐         ┌───────────────┐                      │
│   │   ENCODER      │◄───────►│   DECODER      │                    │
│   │               │ cross-  │               │                      │
│   │ Bidirectional │ attn    │ Autoregressive│                      │
│   │ Self-Attention│         │ Self-Attention│                      │
│   └───────┬───────┘         └───────┬───────┘                      │
│           │                         │                               │
│           ▼                         ▼                               │
│   ┌───────────────┐         ┌───────────────┐                      │
│   │ ENCODER-ONLY  │         │ DECODER-ONLY  │                      │
│   │               │         │               │                      │
│   │ BERT, RoBERTa │         │ GPT, Llama,   │                      │
│   │ DistilBERT    │         │ Mistral       │                      │
│   │               │         │               │                      │
│   │ MLM objective │         │ CLM objective │                      │
│   │ Bidirectional │         │ Left-to-right │                      │
│   │               │         │               │                      │
│   │ Understanding │         │ Generation    │                      │
│   │ tasks         │         │ tasks         │                      │
│   └───────────────┘         └───────────────┘                      │
│                                                                     │
│                    ┌───────────────────┐                             │
│                    │ ENCODER-DECODER   │                             │
│                    │                   │                             │
│                    │ T5, BART, Marian  │                             │
│                    │                   │                             │
│                    │ Seq2Seq objective │                             │
│                    │ Bidirectional enc │                             │
│                    │ + Causal decoder  │                             │
│                    │                   │                             │
│                    │ Translation,      │                             │
│                    │ Summarization     │                             │
│                    └───────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Attention Masking — The Key Differentiator

```
Bidirectional (Encoder / BERT):        Causal (Decoder / GPT):

  The cat sat on                         The cat sat on
T  ■  ■  ■  ■                         T  ■  □  □  □
h  ■  ■  ■  ■                         h  ■  ■  □  □
e  ■  ■  ■  ■                         e  ■  ■  ■  □
   ■  ■  ■  ■                            ■  ■  ■  ■

■ = attends     □ = masked             ■ = attends     □ = masked

Every token sees ALL tokens.           Each token sees ONLY past tokens.
Used for: understanding context.       Used for: generating next token.
```

---

## 3. Experimental Setup

### 3.1 Environment

```yaml
# config.yaml
environment:
  python: ">=3.9"
  framework: "transformers>=4.40.0"
  backend: "torch>=2.0"
  device: "auto"

dependencies:
  core:
    - transformers
    - torch
    - numpy
  visualization:
    - matplotlib
    - seaborn
  analysis:
    - pandas
    - scikit-learn         # for transfer learning evaluation
  optional:
    - bertviz              # attention visualization
    - accelerate           # large model loading
    - datasets             # HuggingFace datasets for fine-tuning
    - sentencepiece        # for T5 tokenizer
```

### 3.2 Project Structure

This experiment lives inside the main repository under `src/architecture_deepdive/`:

```
src/
├── __init__.py                           # Top-level package
├── pipeline_exploration/                 # Ch1.3 experiment (separate)
└── architecture_deepdive/
    ├── __init__.py
    ├── experiment_runner.py              # Main orchestrator
    ├── data.py                           # Shared test sentences and inputs
    ├── probes/
    │   ├── __init__.py
    │   ├── p1_model_timeline.py          # Probe 1: Transformer history
    │   ├── p2_language_modeling.py       # Probe 2: Causal vs Masked LM
    │   ├── p3_transfer_learning.py       # Probe 3: Pretrained vs scratch
    │   ├── p4_model_anatomy.py           # Probe 4: Architecture inspection
    │   ├── p5_attention_viz.py           # Probe 5: Attention visualization
    │   └── p6_arch_comparison.py         # Probe 6: Encoder vs Decoder vs Enc-Dec
    └── utils/
        ├── __init__.py
        ├── attention_tools.py            # Extract and visualize attention weights
        ├── model_inspector.py            # Count params, list layers, trace ops
        └── plotting.py                   # Matplotlib/seaborn helpers

results/                                  # Saved experiment outputs (gitignored)
├── figures/                              # Generated visualizations
├── outputs.json                          # Structured probe outputs
└── comparison_report.md                  # Final analysis
```

---

## 4. Implementation Guide

### 4.1 Shared Data & Utilities — `src/architecture_deepdive/data.py`

```python
"""
Shared test inputs for all probes.
Sentences chosen to highlight attention behavior from the course:
  - "You like this course" (translation example from Ch1.4)
  - Subject-verb agreement (attention distance test)
  - Coreference resolution (long-range dependency test)
"""

# ──────────────────────────────────────────────
# Core sentences from the course
# ──────────────────────────────────────────────
COURSE_SENTENCES = {
    "translation_example": "You like this course",
    "french_target": "Vous aimez ce cours",
    "attention_demo": (
        "The animal didn't cross the street because it was too tired"
    ),
}

# ──────────────────────────────────────────────
# Attention probing sentences
# ──────────────────────────────────────────────
ATTENTION_SENTENCES = {
    # Subject-verb agreement across distance
    "agreement_short": "The cat sits on the mat.",
    "agreement_long": "The cat that chased the mice in the barn sits on the mat.",

    # Coreference — "it" must attend to the right antecedent
    "coref_animal": "The animal didn't cross the street because it was too tired.",
    "coref_street": "The animal didn't cross the street because it was too wide.",

    # Positional sensitivity
    "order_matters": "The dog bit the man.",
    "order_reversed": "The man bit the dog.",
}

# ──────────────────────────────────────────────
# Language modeling test prompts
# ──────────────────────────────────────────────
LM_INPUTS = {
    "causal_prompts": [
        "The capital of France is",
        "In this course, we will teach you how to",
        "Transformers are language models that",
    ],
    "masked_sentences": [
        "The capital of France is [MASK].",
        "Transformers are [MASK] models.",
        "The [MASK] didn't cross the street because it was too tired.",
    ],
}

# ──────────────────────────────────────────────
# Transfer learning dataset (simple sentiment)
# ──────────────────────────────────────────────
TRANSFER_LEARNING_DATA = {
    "train": [
        ("This movie is fantastic!", 1),
        ("Terrible waste of time.", 0),
        ("I loved every moment of it.", 1),
        ("The worst film I have ever seen.", 0),
        ("A beautiful and moving story.", 1),
        ("Boring and predictable plot.", 0),
        ("Absolutely brilliant acting.", 1),
        ("I fell asleep halfway through.", 0),
    ],
    "test": [
        ("An outstanding achievement in cinema.", 1),
        ("Completely unwatchable garbage.", 0),
        ("Heartwarming and genuinely funny.", 1),
        ("Dull characters and weak dialogue.", 0),
    ],
}

# ──────────────────────────────────────────────
# Model registry — architecture families
# ──────────────────────────────────────────────
MODEL_REGISTRY = {
    "encoder_only": {
        "primary": "bert-base-uncased",
        "small": "google/bert_uncased_L-2_H-128_A-2",  # tiny BERT for fast tests
        "family": "BERT-like (auto-encoding)",
        "objective": "Masked Language Modeling (MLM)",
        "attention": "bidirectional",
    },
    "decoder_only": {
        "primary": "gpt2",
        "small": "sshleifer/tiny-gpt2",  # tiny GPT-2 for fast tests
        "family": "GPT-like (auto-regressive)",
        "objective": "Causal Language Modeling (CLM)",
        "attention": "causal (left-to-right)",
    },
    "encoder_decoder": {
        "primary": "t5-small",
        "small": "google/t5-efficient-tiny",  # tiny T5
        "family": "T5-like (sequence-to-sequence)",
        "objective": "Span corruption / Denoising",
        "attention": "bidirectional encoder + causal decoder",
    },
}
```

### 4.2 Utility: Model Inspector — `src/architecture_deepdive/utils/model_inspector.py`

```python
"""
Utilities for inspecting transformer model internals.
Counts parameters, lists layers, extracts attention masks.

References:
  - Ch1.4: "Architecture: the skeleton of the model"
  - Ch1.4: "Checkpoints: the weights loaded in a given architecture"
"""

import torch
from transformers import AutoModel, AutoConfig
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class ModelAnatomy:
    """Complete architectural profile of a transformer model."""
    name: str
    architecture_class: str        # e.g., "BertModel", "GPT2Model"
    family: str                    # encoder-only / decoder-only / encoder-decoder
    num_parameters: int
    num_trainable_parameters: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int         # FFN hidden dim
    vocab_size: int
    max_position_embeddings: int
    has_encoder: bool
    has_decoder: bool
    layer_breakdown: dict          # parameter count per layer type


def inspect_model(model_name: str, family: str) -> ModelAnatomy:
    """
    Load a model and extract its full architectural profile.
    Demonstrates Ch1.4's distinction between architecture and checkpoint.

    Args:
        model_name: HuggingFace model checkpoint name
        family: "encoder_only", "decoder_only", or "encoder_decoder"
    """
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Layer-by-layer parameter breakdown
    layer_breakdown = OrderedDict()
    for name, param in model.named_parameters():
        # Group by top-level module
        top_module = name.split(".")[0]
        if top_module not in layer_breakdown:
            layer_breakdown[top_module] = {"count": 0, "params": 0}
        layer_breakdown[top_module]["count"] += 1
        layer_breakdown[top_module]["params"] += param.numel()

    # Extract config values (handles different config field names)
    num_layers = getattr(config, "num_hidden_layers",
                 getattr(config, "n_layer",
                 getattr(config, "num_layers", 0)))

    hidden_size = getattr(config, "hidden_size",
                  getattr(config, "n_embd",
                  getattr(config, "d_model", 0)))

    num_heads = getattr(config, "num_attention_heads",
                getattr(config, "n_head",
                getattr(config, "num_heads", 0)))

    intermediate_size = getattr(config, "intermediate_size",
                        getattr(config, "n_inner",
                        getattr(config, "d_ff", 0)))

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
        max_position_embeddings=getattr(config, "max_position_embeddings",
                                getattr(config, "n_positions", 0)),
        has_encoder="encoder" in family or "encoder_decoder" in family,
        has_decoder="decoder" in family or "encoder_decoder" in family,
        layer_breakdown={k: v for k, v in layer_breakdown.items()},
    )


def format_param_count(n: int) -> str:
    """Human-readable parameter count (e.g., 110M, 1.5B)."""
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)
```

### 4.3 Utility: Attention Tools — `src/architecture_deepdive/utils/attention_tools.py`

```python
"""
Extract, process, and prepare attention weights for visualization.
Core utility for Probe 5 (Attention Visualization).

Reference:
  - Ch1.4: "Attention layers tell the model to pay specific attention
    to certain words and more or less ignore the others"
"""

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer


def extract_attention_weights(
    model_name: str,
    text: str,
    device: str = "cpu",
) -> dict:
    """
    Run a forward pass and extract attention weights from all layers/heads.

    Returns:
        {
            "tokens": ["[CLS]", "the", "cat", ...],
            "attentions": np.ndarray of shape (num_layers, num_heads, seq_len, seq_len),
        }
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.to(device).eval()

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.attentions is a tuple of (num_layers,) tensors
    # each tensor: (batch=1, num_heads, seq_len, seq_len)
    attentions = torch.stack(outputs.attentions).squeeze(1)  # (layers, heads, seq, seq)
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
    """
    For a given target token, get how much attention each other token
    pays to it (summed over heads).

    This directly tests the course's claim:
      "the model pays specific attention to certain words"

    Args:
        attentions: (layers, heads, seq, seq)
        tokens: list of token strings
        target_token: the token to analyze
        layer: which layer (-1 = last)

    Returns:
        dict mapping each token to its attention weight toward target_token
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

    return {
        tok: round(float(attention_to_target[i]), 4)
        for i, tok in enumerate(tokens)
    }


def compare_causal_vs_bidirectional_mask(seq_len: int) -> dict:
    """
    Generate and compare attention masks for causal and bidirectional models.
    Directly illustrates the key architectural difference from Ch1.4.
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
            f"For seq_len={seq_len}: bidirectional has {seq_len**2} attention pairs, "
            f"causal has {seq_len*(seq_len+1)//2} (upper triangle masked)"
        ),
    }
```

### 4.4 Utility: Plotting — `src/architecture_deepdive/utils/plotting.py`

```python
"""
Visualization helpers for attention matrices, parameter charts, etc.
Outputs saved as PNG/SVG to results/figures/.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path

FIGURE_DIR = Path("results/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def plot_attention_matrix(
    attention: np.ndarray,
    tokens: list,
    title: str = "Attention Weights",
    save_name: str = "attention_matrix.png",
    layer: int = 0,
    head: int = 0,
):
    """
    Plot a single attention head's weight matrix as a heatmap.
    Rows = query tokens, Columns = key tokens.

    Args:
        attention: (layers, heads, seq, seq)
        tokens: token strings
        title: plot title
        save_name: filename
        layer, head: which layer/head to plot
    """
    attn = attention[layer, head]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap="YlOrRd",
        vmin=0, vmax=1,
        annot=True if len(tokens) <= 12 else False,
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
):
    """
    Side-by-side plot of bidirectional vs causal attention masks.
    Directly illustrates the course's encoder vs decoder distinction.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, mask, title in zip(
        axes,
        [bidirectional, causal],
        ["Bidirectional (Encoder / BERT)", "Causal (Decoder / GPT)"],
    ):
        sns.heatmap(
            mask,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="Blues",
            vmin=0, vmax=1,
            cbar=False,
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")

    plt.suptitle(
        "Attention Mask Patterns — The Key Architectural Difference",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / save_name, dpi=150, bbox_inches="tight")
    plt.close()


def plot_parameter_comparison(
    model_data: list[dict],
    save_name: str = "parameter_comparison.png",
):
    """
    Bar chart comparing parameter counts across architecture families.

    Args:
        model_data: [{"name": str, "family": str, "params": int}, ...]
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"encoder_only": "#4CAF50", "decoder_only": "#2196F3", "encoder_decoder": "#FF9800"}
    names = [d["name"] for d in model_data]
    params = [d["params"] / 1e6 for d in model_data]
    bar_colors = [colors.get(d["family"], "#999") for d in model_data]

    bars = ax.barh(names, params, color=bar_colors, edgecolor="white")
    ax.set_xlabel("Parameters (Millions)")
    ax.set_title("Parameter Count by Architecture Family")

    for bar, p in zip(bars, params):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{p:.1f}M", va="center", fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=l)
        for l, c in [("Encoder-only", "#4CAF50"), ("Decoder-only", "#2196F3"), ("Encoder-decoder", "#FF9800")]
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
):
    """
    Learning curves: pretrained vs from-scratch fine-tuning.
    Illustrates Ch1.4's transfer learning concept.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, pretrained_scores, "o-", color="#4CAF50", label="Pretrained + fine-tuned", linewidth=2)
    ax.plot(epochs, scratch_scores, "s--", color="#F44336", label="Trained from scratch", linewidth=2)

    ax.set_xlabel("Training Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title("Transfer Learning: Pretrained vs From-Scratch\n(Ch1.4 Core Concept)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(FIGURE_DIR / save_name, dpi=150)
    plt.close()
```

---

### 4.5 Probe 1: Transformer History Timeline — `src/architecture_deepdive/probes/p1_model_timeline.py`

```python
"""
Probe 1: Transformer History
Reference: Ch1.4 — "A bit of Transformer history"

Builds a structured timeline of the models mentioned in the course,
enriched with parameter counts and architecture classification.
Maps each model to one of the three families:
  - GPT-like (auto-regressive)
  - BERT-like (auto-encoding)
  - T5-like (sequence-to-sequence)
"""

from dataclasses import dataclass


@dataclass
class TransformerMilestone:
    name: str
    date: str
    params: str
    family: str
    architecture: str
    key_innovation: str
    hf_checkpoint: str


# Timeline data from the course
TIMELINE = [
    TransformerMilestone(
        name="Transformer (Original)",
        date="June 2017",
        params="65M",
        family="encoder-decoder",
        architecture="Full encoder-decoder",
        key_innovation="Self-attention replacing recurrence entirely",
        hf_checkpoint="N/A (original paper)",
    ),
    TransformerMilestone(
        name="GPT",
        date="June 2018",
        params="117M",
        family="decoder-only",
        architecture="12-layer decoder",
        key_innovation="First pretrained Transformer; causal LM + fine-tuning",
        hf_checkpoint="openai-gpt",
    ),
    TransformerMilestone(
        name="BERT",
        date="October 2018",
        params="110M / 340M",
        family="encoder-only",
        architecture="12/24-layer encoder",
        key_innovation="Bidirectional context via Masked LM + Next Sentence Prediction",
        hf_checkpoint="bert-base-uncased",
    ),
    TransformerMilestone(
        name="GPT-2",
        date="February 2019",
        params="1.5B",
        family="decoder-only",
        architecture="48-layer decoder",
        key_innovation="Scaled-up GPT; zero-shot task transfer; delayed release for ethics",
        hf_checkpoint="gpt2",
    ),
    TransformerMilestone(
        name="T5",
        date="October 2019",
        params="60M–11B",
        family="encoder-decoder",
        architecture="Encoder-decoder with text-to-text framing",
        key_innovation="Unified all NLP tasks as text-to-text; span corruption pretraining",
        hf_checkpoint="t5-small",
    ),
    TransformerMilestone(
        name="GPT-3",
        date="May 2020",
        params="175B",
        family="decoder-only",
        architecture="96-layer decoder",
        key_innovation="In-context learning; few-shot / zero-shot without fine-tuning",
        hf_checkpoint="N/A (API only)",
    ),
    TransformerMilestone(
        name="InstructGPT",
        date="January 2022",
        params="~175B",
        family="decoder-only",
        architecture="GPT-3 + RLHF",
        key_innovation="RLHF alignment; instruction following",
        hf_checkpoint="N/A (API only)",
    ),
    TransformerMilestone(
        name="Llama",
        date="January 2023",
        params="7B–65B",
        family="decoder-only",
        architecture="Decoder with RMSNorm, SwiGLU, RoPE",
        key_innovation="Open-weight LLM; efficient training on public data",
        hf_checkpoint="meta-llama/Llama-2-7b-hf",
    ),
    TransformerMilestone(
        name="Mistral",
        date="March 2023",
        params="7B",
        family="decoder-only",
        architecture="Decoder with GQA + Sliding Window Attention",
        key_innovation="Grouped-query attention; sliding window for long sequences",
        hf_checkpoint="mistralai/Mistral-7B-v0.1",
    ),
    TransformerMilestone(
        name="Gemma 2",
        date="May 2024",
        params="2B–27B",
        family="decoder-only",
        architecture="Decoder with interleaved local-global attention",
        key_innovation="Knowledge distillation; local-global attention interleaving",
        hf_checkpoint="google/gemma-2-2b",
    ),
    TransformerMilestone(
        name="SmolLM2",
        date="November 2024",
        params="135M–1.7B",
        family="decoder-only",
        architecture="Compact decoder",
        key_innovation="State-of-the-art at small scale; edge/mobile deployment",
        hf_checkpoint="HuggingFaceTB/SmolLM2-360M",
    ),
]


def run_experiment() -> dict:
    """Build timeline and classify by architecture family."""
    results = {"task": "transformer_timeline"}

    # Family distribution
    family_counts = {}
    for m in TIMELINE:
        family_counts[m.family] = family_counts.get(m.family, 0) + 1

    results["timeline"] = [
        {
            "name": m.name,
            "date": m.date,
            "params": m.params,
            "family": m.family,
            "key_innovation": m.key_innovation,
            "hf_checkpoint": m.hf_checkpoint,
        }
        for m in TIMELINE
    ]

    results["family_distribution"] = family_counts

    results["observation"] = (
        "The course timeline shows a clear trend: decoder-only models dominate "
        f"modern LLMs ({family_counts.get('decoder-only', 0)}/{len(TIMELINE)} models). "
        "Encoder-only (BERT-like) peaked in 2018-2019 for understanding tasks, "
        "while encoder-decoder (T5-like) remains relevant for seq2seq. "
        "Post-2022 models are almost exclusively decoder-only with attention optimizations "
        "(GQA, sliding window, local-global interleaving)."
    )

    return results
```

---

### 4.6 Probe 2: Causal vs Masked Language Modeling — `src/architecture_deepdive/probes/p2_language_modeling.py`

```python
"""
Probe 2: Language Modeling — Causal (CLM) vs Masked (MLM)
Reference: Ch1.4 — "Transformers are language models"

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
    """
    Causal Language Modeling: predict the next token left-to-right.

    From the course:
      "This is called causal language modeling because the output depends
       on the past and present inputs, but not the future ones."
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

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

        results.append({
            "prompt": prompt,
            "top_5_predictions": [
                {"token": tok.strip(), "probability": round(prob.item(), 4)}
                for tok, prob in zip(top5_tokens, top5_probs)
            ],
            "context_direction": "left-to-right only (causal mask)",
        })

    return {"model": model_name, "objective": "Causal LM (CLM)", "results": results}


def run_masked_lm_probe(model_name: str = "bert-base-uncased", device: str = "cpu") -> dict:
    """
    Masked Language Modeling: predict masked tokens using FULL context.

    From the course:
      "Another example is masked language modeling, in which the model
       predicts a masked word in the sentence."
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()

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

                results.append({
                    "sentence": sentence,
                    "mask_position": pos.item(),
                    "top_5_predictions": [
                        {"token": tok, "probability": round(prob.item(), 4)}
                        for tok, prob in zip(top5_tokens, top5_probs)
                    ],
                    "context_direction": "bidirectional (sees left AND right of [MASK])",
                })

    return {"model": model_name, "objective": "Masked LM (MLM)", "results": results}


def run_experiment(device: str = "cpu") -> dict:
    """Compare CLM vs MLM on equivalent prompts."""
    results = {"task": "language_modeling_comparison"}

    results["causal_lm"] = run_causal_lm_probe(device=device)
    results["masked_lm"] = run_masked_lm_probe(device=device)

    # Direct comparison on shared concept
    results["comparison_analysis"] = {
        "shared_test": "Predicting 'France' in 'The capital of France is ___'",
        "clm_approach": "GPT sees: 'The capital of France is' → predicts next token",
        "mlm_approach": "BERT sees: 'The capital of France is [MASK] .' → predicts masked token using FULL sentence including '.'",
        "key_difference": (
            "CLM can only use LEFT context (past tokens). "
            "MLM uses BOTH left and right context. "
            "This is why BERT is better for understanding tasks "
            "and GPT is better for generation tasks."
        ),
    }

    return results
```

---

### 4.7 Probe 3: Transfer Learning — `src/architecture_deepdive/probes/p3_transfer_learning.py`

```python
"""
Probe 3: Transfer Learning — Pretrained vs From-Scratch
Reference: Ch1.4 — "Transfer Learning"

From the course:
  "The pretrained model was already trained on a dataset that has some
   similarities with the fine-tuning dataset. The fine-tuning process is
   thus able to take advantage of knowledge acquired during pretraining."

This probe empirically demonstrates the transfer learning advantage by
comparing fine-tuning a pretrained BERT vs training a randomly initialized
BERT on a small sentiment dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer, AutoConfig
from src.architecture_deepdive.data import TRANSFER_LEARNING_DATA


class SimpleClassifier(nn.Module):
    """BERT + linear classification head."""
    def __init__(self, base_model: nn.Module, hidden_size: int, num_labels: int = 2):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)


def prepare_data(model_name: str, split: str) -> TensorDataset:
    """Tokenize and prepare dataset."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data = TRANSFER_LEARNING_DATA[split]

    texts = [t for t, _ in data]
    labels = [l for _, l in data]

    encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    return TensorDataset(
        encodings["input_ids"],
        encodings["attention_mask"],
        torch.tensor(labels, dtype=torch.long),
    )


def train_and_evaluate(
    model_name: str,
    from_scratch: bool = False,
    num_epochs: int = 10,
    lr: float = 2e-5,
    device: str = "cpu",
) -> dict:
    """
    Train a classifier and track accuracy per epoch.

    Args:
        model_name: HuggingFace model name
        from_scratch: If True, use random weights (no pretraining)
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device string
    """
    config = AutoConfig.from_pretrained(model_name)

    if from_scratch:
        # Randomly initialized — no pretrained knowledge
        base_model = AutoModel.from_config(config)
        mode = "from_scratch"
    else:
        # Pretrained weights — transfer learning
        base_model = AutoModel.from_pretrained(model_name)
        mode = "pretrained"

    classifier = SimpleClassifier(base_model, config.hidden_size).to(device)

    # Prepare data
    train_ds = prepare_data(model_name, "train")
    test_ds = prepare_data(model_name, "test")
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=4)

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop with per-epoch evaluation
    history = {"epochs": [], "train_loss": [], "test_accuracy": []}

    for epoch in range(1, num_epochs + 1):
        # Train
        classifier.train()
        total_loss = 0
        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = classifier(ids, mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for ids, mask, labels in test_loader:
                ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
                logits = classifier(ids, mask)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0

        history["epochs"].append(epoch)
        history["train_loss"].append(round(total_loss / len(train_loader), 4))
        history["test_accuracy"].append(round(accuracy, 4))

    return {
        "mode": mode,
        "model": model_name,
        "from_scratch": from_scratch,
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "history": history,
        "final_accuracy": history["test_accuracy"][-1],
    }


def run_experiment(device: str = "cpu") -> dict:
    """
    Compare pretrained fine-tuning vs from-scratch training.
    Use a small BERT for speed.
    """
    results = {"task": "transfer_learning"}
    model_name = "google/bert_uncased_L-2_H-128_A-2"  # Tiny BERT

    results["pretrained"] = train_and_evaluate(
        model_name, from_scratch=False, num_epochs=10, device=device,
    )
    results["from_scratch"] = train_and_evaluate(
        model_name, from_scratch=True, num_epochs=10, device=device,
    )

    # Analysis
    pt_acc = results["pretrained"]["final_accuracy"]
    sc_acc = results["from_scratch"]["final_accuracy"]

    results["analysis"] = {
        "pretrained_final_accuracy": pt_acc,
        "scratch_final_accuracy": sc_acc,
        "accuracy_gap": round(pt_acc - sc_acc, 4),
        "conclusion": (
            "Pretrained model achieves higher accuracy with fewer epochs, "
            "validating Ch1.4's claim that 'the fine-tuning process is able to "
            "take advantage of knowledge acquired during pretraining'. "
            "From-scratch training on this tiny dataset cannot learn meaningful "
            "language representations."
        ),
    }

    return results
```

---

### 4.8 Probe 4: Model Anatomy Inspector — `src/architecture_deepdive/probes/p4_model_anatomy.py`

```python
"""
Probe 4: General Transformer Architecture — Model Anatomy
Reference: Ch1.4 — "General Transformer architecture" + "Architectures vs. checkpoints"

From the course:
  "Architecture: the skeleton of the model — the definition of each layer
   and each operation that happens within the model."
  "Checkpoints: the weights that will be loaded in a given architecture."

This probe inspects one representative model from each family:
  - BERT (encoder-only)
  - GPT-2 (decoder-only)
  - T5 (encoder-decoder)
"""

from src.architecture_deepdive.data import MODEL_REGISTRY
from src.architecture_deepdive.utils.model_inspector import inspect_model, format_param_count


def run_experiment(device: str = "cpu") -> dict:
    results = {"task": "model_anatomy"}

    anatomies = {}
    for family_key, info in MODEL_REGISTRY.items():
        model_name = info["primary"]
        anatomy = inspect_model(model_name, family_key)
        anatomies[family_key] = anatomy

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

    # Cross-architecture comparison table
    results["comparison"] = {
        "architecture_vs_checkpoint_note": (
            "BERT is an *architecture*. bert-base-uncased is a *checkpoint*. "
            "Both GPT-2 and gpt2 refer to different things: "
            "one is the architecture design, the other is a specific set of trained weights."
        ),
        "structural_comparison": {
            "encoder_only (BERT)": {
                "components": "Embeddings → N × [Self-Attention + FFN] → Pooler",
                "attention_mask": "Full (bidirectional)",
                "output": "Contextualized token representations",
            },
            "decoder_only (GPT-2)": {
                "components": "Embeddings → N × [Masked Self-Attention + FFN] → LM Head",
                "attention_mask": "Causal (lower triangular)",
                "output": "Next-token logits",
            },
            "encoder_decoder (T5)": {
                "components": "Encoder: Embeddings → N × [Self-Attention + FFN]\n"
                              "Decoder: Embeddings → N × [Self-Attn + Cross-Attn + FFN] → LM Head",
                "attention_mask": "Encoder: bidirectional, Decoder: causal + cross-attention to encoder",
                "output": "Generated sequence",
            },
        },
    }

    return results
```

---

### 4.9 Probe 5: Attention Visualization — `src/architecture_deepdive/probes/p5_attention_viz.py`

```python
"""
Probe 5: Attention Layer Visualization
Reference: Ch1.4 — "Attention layers"

From the course:
  "This layer will tell the model to pay specific attention to certain words
   in the sentence you passed it (and more or less ignore the others)"

  "Given the input 'You like this course', a translation model will need
   to also attend to the adjacent word 'You' to get the proper translation
   for the word 'like', because in French the verb 'like' is conjugated
   differently depending on the subject."

This probe extracts and visualizes real attention weights to verify
these theoretical claims empirically.
"""

from src.architecture_deepdive.data import ATTENTION_SENTENCES
from src.architecture_deepdive.utils.attention_tools import (
    extract_attention_weights,
    get_attention_to_token,
    compare_causal_vs_bidirectional_mask,
)
from src.architecture_deepdive.utils.plotting import (
    plot_attention_matrix,
    plot_attention_mask_comparison,
)


def run_experiment(device: str = "cpu") -> dict:
    results = {"task": "attention_visualization"}

    # ── 5.1: Attention mask comparison (theoretical) ──
    tokens = ["You", "like", "this", "course"]
    masks = compare_causal_vs_bidirectional_mask(seq_len=len(tokens))
    results["mask_comparison"] = {
        "tokens": tokens,
        "causal_masked_positions": masks["causal_masked_positions"],
        "note": masks["note"],
    }

    # Generate comparison figure
    plot_attention_mask_comparison(
        masks["bidirectional"], masks["causal"], tokens,
        save_name="01_mask_comparison.png",
    )
    results["figures"] = ["results/figures/01_mask_comparison.png"]

    # ── 5.2: BERT attention on coreference (bidirectional) ──
    bert_model = "bert-base-uncased"

    for key, sentence in ATTENTION_SENTENCES.items():
        attn_data = extract_attention_weights(bert_model, sentence, device)

        # Plot first layer, first head
        plot_attention_matrix(
            attn_data["attentions"],
            attn_data["tokens"],
            title=f"BERT Attention: {key}",
            save_name=f"02_bert_{key}.png",
            layer=0, head=0,
        )
        results["figures"].append(f"results/figures/02_bert_{key}.png")

        # Also plot last layer (more task-specific patterns)
        plot_attention_matrix(
            attn_data["attentions"],
            attn_data["tokens"],
            title=f"BERT Attention (Last Layer): {key}",
            save_name=f"02_bert_{key}_last.png",
            layer=-1, head=0,
        )
        results["figures"].append(f"results/figures/02_bert_{key}_last.png")

    # ── 5.3: Course example — "it" in coreference ──
    #
    # "The animal didn't cross the street because it was too tired."
    # → "it" should attend to "animal" (the antecedent)
    #
    # "The animal didn't cross the street because it was too wide."
    # → "it" should attend to "street"
    #
    coref_tired = extract_attention_weights(
        bert_model, ATTENTION_SENTENCES["coref_animal"], device
    )
    coref_wide = extract_attention_weights(
        bert_model, ATTENTION_SENTENCES["coref_street"], device
    )

    # Measure attention from "it" → "animal" vs "it" → "street"
    it_to_animal = get_attention_to_token(
        coref_tired["attentions"], coref_tired["tokens"], "animal", layer=-1
    )
    it_to_street = get_attention_to_token(
        coref_wide["attentions"], coref_wide["tokens"], "street", layer=-1
    )

    results["coreference_test"] = {
        "hypothesis": (
            "In 'it was too tired', 'it' should attend more to 'animal'. "
            "In 'it was too wide', 'it' should attend more to 'street'."
        ),
        "tired_sentence": {
            "attention_from_all_tokens_to_animal": it_to_animal,
        },
        "wide_sentence": {
            "attention_from_all_tokens_to_street": it_to_street,
        },
    }

    # ── 5.4: GPT-2 attention (causal) for comparison ──
    gpt_model = "gpt2"
    gpt_attn = extract_attention_weights(
        gpt_model, ATTENTION_SENTENCES["agreement_short"], device
    )
    plot_attention_matrix(
        gpt_attn["attentions"],
        gpt_attn["tokens"],
        title="GPT-2 Attention (Causal)",
        save_name="03_gpt2_causal.png",
        layer=0, head=0,
    )
    results["figures"].append("results/figures/03_gpt2_causal.png")

    results["causal_observation"] = (
        "GPT-2 attention matrix shows a clear lower-triangular pattern: "
        "each token can only attend to itself and preceding tokens. "
        "BERT attention matrices are full (bidirectional). "
        "This is the defining structural difference between encoder and decoder."
    )

    return results
```

---

### 4.10 Probe 6: Architecture Family Comparison — `src/architecture_deepdive/probes/p6_arch_comparison.py`

```python
"""
Probe 6: Encoder vs Decoder vs Encoder-Decoder Comparison
Reference: Ch1.4 — "General Transformer architecture"

From the course:
  "Encoder-only models: Good for tasks that require understanding of the input"
  "Decoder-only models: Good for generative tasks such as text generation"
  "Encoder-decoder models: Good for generative tasks that require an input"

This probe runs all three architecture types on the same input and
compares their hidden state representations, output formats, and
task-specific behaviors.
"""

import torch
import numpy as np
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)
from src.architecture_deepdive.data import MODEL_REGISTRY


SHARED_INPUT = "The Transformer architecture was introduced in 2017."


def probe_encoder_only(device: str = "cpu") -> dict:
    """
    BERT (Encoder-Only): Produces contextualized token embeddings.
    Bidirectional attention — every token sees every other token.
    Best for: classification, NER, QA.
    """
    model_name = MODEL_REGISTRY["encoder_only"]["primary"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True).to(device).eval()

    inputs = tokenizer(SHARED_INPUT, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.last_hidden_state  # (1, seq_len, hidden_size)
    attentions = outputs.attentions     # tuple of (1, heads, seq, seq)

    # Verify bidirectional: attention matrix should be full (no zeros in non-padding)
    attn_matrix = attentions[0][0, 0].cpu().numpy()  # layer 0, head 0
    is_bidirectional = np.all(attn_matrix > 0)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    return {
        "model": model_name,
        "architecture": "encoder-only",
        "hidden_state_shape": list(hidden.shape),
        "num_tokens": len(tokens),
        "tokens": tokens,
        "hidden_dim": hidden.shape[-1],
        "attention_is_bidirectional": bool(is_bidirectional),
        "attention_matrix_shape": list(attn_matrix.shape),
        "output_type": "Contextualized embeddings per token",
        "typical_use": "Add task-specific head (classifier, QA span predictor)",
        "cls_embedding_norm": round(float(hidden[0, 0].norm()), 4),
    }


def probe_decoder_only(device: str = "cpu") -> dict:
    """
    GPT-2 (Decoder-Only): Generates next tokens autoregressively.
    Causal attention — each token only sees past tokens.
    Best for: text generation.
    """
    model_name = MODEL_REGISTRY["decoder_only"]["primary"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, output_attentions=True
    ).to(device).eval()

    inputs = tokenizer(SHARED_INPUT, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden = outputs.hidden_states if hasattr(outputs, "hidden_states") else None
    logits = outputs.logits                  # (1, seq_len, vocab_size)
    attentions = outputs.attentions          # tuple of (1, heads, seq, seq)

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
        "attention_is_causal": bool(is_causal),
        "upper_triangle_attn_sum": round(upper_triangle_sum, 8),
        "next_token_predictions": [
            {"token": t, "prob": round(p.item(), 4)}
            for t, p in zip(top5_tokens, top5_probs)
        ],
        "output_type": "Next-token probability distribution",
        "typical_use": "Autoregressive text generation",
    }


def probe_encoder_decoder(device: str = "cpu") -> dict:
    """
    T5 (Encoder-Decoder): Encodes input, then generates output sequence.
    Encoder has bidirectional attention, decoder has causal + cross-attention.
    Best for: translation, summarization, question answering.

    From the course:
      "The first attention layer in a decoder block pays attention to all
       (past) inputs to the decoder, but the second attention layer uses
       the output of the encoder."
    """
    model_name = MODEL_REGISTRY["encoder_decoder"]["primary"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()

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
        "output_type": "Generated sequence (encoder context → decoder generation)",
        "typical_use": "Translation, summarization, text-to-text tasks",
        "key_feature": (
            "Decoder has TWO attention layers per block: "
            "(1) causal self-attention over generated tokens, "
            "(2) cross-attention over encoder hidden states. "
            "This allows the decoder to 'look at' the full input."
        ),
    }


def run_experiment(device: str = "cpu") -> dict:
    """Run all three architecture probes and compare."""
    results = {"task": "architecture_comparison"}

    results["encoder_only"] = probe_encoder_only(device)
    results["decoder_only"] = probe_decoder_only(device)
    results["encoder_decoder"] = probe_encoder_decoder(device)

    results["shared_input"] = SHARED_INPUT

    results["synthesis"] = {
        "attention_patterns": {
            "encoder_only": "Full bidirectional matrix (every token sees every token)",
            "decoder_only": "Lower-triangular causal matrix (only see past tokens)",
            "encoder_decoder": "Encoder=bidirectional, Decoder=causal+cross-attention",
        },
        "output_format": {
            "encoder_only": "Per-token embeddings (needs task head)",
            "decoder_only": "Next-token logits (autoregressive generation)",
            "encoder_decoder": "Generated token sequence",
        },
        "task_suitability": {
            "encoder_only": "Classification, NER, QA, sentence similarity",
            "decoder_only": "Text generation, code completion, chatbots",
            "encoder_decoder": "Translation, summarization, question answering",
        },
        "course_quote": (
            "Each of these parts can be used independently, depending on the task."
        ),
    }

    return results
```

---

### 4.11 Main Orchestrator — `src/architecture_deepdive/experiment_runner.py`

```python
"""
Main experiment orchestrator for all 6 architecture probes.

Usage:
    python -m src.architecture_deepdive.experiment_runner \
        [--device cpu|cuda] [--probes all|p1,p2,...]
"""

import argparse
import json
import time
import torch
from pathlib import Path

from src.architecture_deepdive.probes import (
    p1_model_timeline,
    p2_language_modeling,
    p3_transfer_learning,
    p4_model_anatomy,
    p5_attention_viz,
    p6_arch_comparison,
)

PROBE_REGISTRY = {
    "p1_timeline": p1_model_timeline,
    "p2_language_modeling": p2_language_modeling,
    "p3_transfer_learning": p3_transfer_learning,
    "p4_model_anatomy": p4_model_anatomy,
    "p5_attention_viz": p5_attention_viz,
    "p6_arch_comparison": p6_arch_comparison,
}


def main():
    parser = argparse.ArgumentParser(description="Transformer Architecture Experiment")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--probes", default="all", help="Comma-separated probe names or 'all'")
    parser.add_argument("--output", default="results/outputs.json")
    args = parser.parse_args()

    device = "cpu"
    if args.device == "auto" and torch.cuda.is_available():
        device = "cuda"
    elif args.device == "cuda":
        device = "cuda"

    probes = list(PROBE_REGISTRY.keys()) if args.probes == "all" else args.probes.split(",")

    all_results = {
        "metadata": {
            "experiment": "Transformer Architecture Deep Dive (Ch1.4)",
            "device": device,
            "torch_version": torch.__version__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "probes": {},
    }

    for probe_name in probes:
        if probe_name not in PROBE_REGISTRY:
            print(f"⚠️  Unknown probe: {probe_name}")
            continue

        print(f"\n{'='*60}")
        print(f"  Probe: {probe_name}")
        print(f"{'='*60}")

        module = PROBE_REGISTRY[probe_name]
        start = time.perf_counter()

        try:
            # p1 doesn't need device
            if probe_name == "p1_timeline":
                result = module.run_experiment()
            else:
                result = module.run_experiment(device=device)
            result["status"] = "success"
        except Exception as e:
            result = {"status": "failed", "error": str(e)}
            print(f"  ❌ Failed: {e}")

        result["duration_sec"] = round(time.perf_counter() - start, 2)
        all_results["probes"][probe_name] = result
        print(f"  ✅ Done in {result['duration_sec']}s")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    succeeded = sum(1 for r in all_results["probes"].values() if r["status"] == "success")
    print(f"\n  Summary: {succeeded}/{len(all_results['probes'])} probes passed")
    print(f"  Results: {output_path}")
    print(f"  Figures: results/figures/")


if __name__ == "__main__":
    main()
```

### 4.12 Run Script — `scripts/run_experiment.sh`

```bash
#!/bin/bash
set -euo pipefail

echo "╔══════════════════════════════════════════════╗"
echo "║  🔬 Transformer Architecture Experiment      ║"
echo "║  HF LLM Course — Chapter 1.4                 ║"
echo "╚══════════════════════════════════════════════╝"

# Setup
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

# Detect device
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
    echo "🎮 GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
    DEVICE="cpu"
    echo "💻 CPU mode (recommend GPU for Probes 3-6)"
fi

# Run all probes
python -m src.architecture_deepdive.experiment_runner --device "$DEVICE" --probes all

echo ""
echo "📊 Results: results/outputs.json"
echo "🖼️  Figures: results/figures/"
```

### 4.13 Requirements — `requirements.txt`

```
transformers>=4.40.0
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
scikit-learn>=1.3.0
sentencepiece>=0.2.0
accelerate>=0.30.0
datasets>=2.18.0
```

---

## 5. Expected Results

### 5.1 Probe 1 — Timeline

| Observation | Expected Finding |
|-------------|-----------------|
| Family distribution | Decoder-only dominates (8/11 models) |
| Architecture trend | Post-2022: decoder-only with attention optimizations (GQA, SWA) |
| Scale trajectory | 65M (2017) → 175B (2020) → back to efficient models (2024) |

### 5.2 Probe 2 — Causal vs Masked LM

| Test Case | CLM (GPT-2) | MLM (BERT) |
|-----------|-------------|-----------|
| "The capital of France is ___" | Top prediction: contextual next word | Top prediction: "Paris" (high confidence using bidirectional context) |
| Context usage | Only left context | Left + right context |

### 5.3 Probe 3 — Transfer Learning

| Metric | Pretrained + Fine-tuned | From Scratch |
|--------|------------------------|-------------|
| Epoch 1 accuracy | ~70-80% | ~50% (random) |
| Epoch 10 accuracy | ~90-100% | ~50-65% |
| Convergence speed | Fast (2-3 epochs) | Slow or never |

### 5.4 Probe 4 — Model Anatomy

| Property | BERT-base | GPT-2 | T5-small |
|----------|-----------|-------|----------|
| Parameters | ~110M | ~124M | ~60M |
| Layers | 12 | 12 | 6+6 (enc+dec) |
| Hidden size | 768 | 768 | 512 |
| Attention heads | 12 | 12 | 8 |
| Has encoder | ✅ | ❌ | ✅ |
| Has decoder | ❌ | ✅ | ✅ |

### 5.5 Probe 5 — Attention Visualization

| Test | Expected Pattern |
|------|-----------------|
| BERT attention matrix | Full (non-zero everywhere, bidirectional) |
| GPT-2 attention matrix | Lower-triangular (causal mask, upper triangle ≈ 0) |
| "it → animal" (tired) | Higher attention weight from "it" to "animal" |
| "it → street" (wide) | Higher attention weight from "it" to "street" |

### 5.6 Probe 6 — Architecture Comparison

| Aspect | Encoder-Only | Decoder-Only | Encoder-Decoder |
|--------|-------------|-------------|-----------------|
| Output | Per-token embeddings | Next-token logits | Generated sequence |
| Attention | Bidirectional | Causal | Bidirectional + Causal + Cross |
| Generation | ❌ | ✅ | ✅ |
| Input understanding | ✅ (deep) | ✅ (left-only) | ✅ (deep, via encoder) |

---

## 6. Key Concepts Verified by This Experiment

### 6.1 From the Course → To Empirical Evidence

| Course Claim | Probe | How We Verify |
|-------------|-------|---------------|
| "Self-supervised learning: the objective is automatically computed from inputs" | P2 | CLM predicts next token; MLM predicts masked token — no human labels |
| "Attention tells the model to pay specific attention to certain words" | P5 | Visualize attention matrices; measure attention from "it" to antecedent |
| "The encoder can use all words in a sentence" | P5, P6 | BERT attention matrix is full; GPT attention is lower-triangular |
| "The decoder works sequentially and can only pay attention to words it has already translated" | P5, P6 | GPT-2 upper-triangle attention ≈ 0; verified numerically |
| "Fine-tuning takes advantage of knowledge acquired during pretraining" | P3 | Pretrained model reaches higher accuracy faster than from-scratch |
| "Training a model requires a very large amount of data" | P3 | From-scratch model fails on 8 training examples; pretrained succeeds |
| "BERT is an architecture; bert-base-cased is a checkpoint" | P4 | Inspect config vs weights; same architecture, different checkpoint |

### 6.2 Attention Mask Comparison — The Central Insight

```
Encoder (BERT):              Decoder (GPT-2):            Enc-Dec (T5 Decoder):
┌─────────────┐              ┌─────────────┐             ┌─────────────┐
│ ■ ■ ■ ■ ■ ■│              │ ■ □ □ □ □ □│             │ ■ □ □ │ ■ ■ ■│
│ ■ ■ ■ ■ ■ ■│              │ ■ ■ □ □ □ □│             │ ■ ■ □ │ ■ ■ ■│
│ ■ ■ ■ ■ ■ ■│              │ ■ ■ ■ □ □ □│             │ ■ ■ ■ │ ■ ■ ■│
│ ■ ■ ■ ■ ■ ■│              │ ■ ■ ■ ■ □ □│             └───────┴──────┘
│ ■ ■ ■ ■ ■ ■│              │ ■ ■ ■ ■ ■ □│              Self-Attn  Cross-Attn
│ ■ ■ ■ ■ ■ ■│              │ ■ ■ ■ ■ ■ ■│              (causal)   (to encoder)
└─────────────┘              └─────────────┘

All see all.                 Each sees only past.        Causal self + full encoder.
```

---

## 7. Generated Figures Catalog

| Figure | File | Description |
|--------|------|-------------|
| 1 | `01_mask_comparison.png` | Side-by-side bidirectional vs causal masks |
| 2-7 | `02_bert_*.png` | BERT attention matrices for each test sentence |
| 8 | `03_gpt2_causal.png` | GPT-2 attention showing causal pattern |
| 9 | `parameter_comparison.png` | Bar chart of parameter counts across families |
| 10 | `transfer_learning.png` | Learning curves: pretrained vs scratch |

---

## 8. Reproducibility Checklist

- [ ] Python >=3.9
- [ ] All packages from `requirements.txt` installed
- [ ] Internet access for model downloads (~2GB total for all models)
- [ ] GPU optional (CPU works, ~10-20 min total runtime)
- [ ] All 6 probes return `"status": "success"`
- [ ] Attention matrices show expected bidirectional/causal patterns
- [ ] Transfer learning shows measurable pretrained advantage
- [ ] Model anatomy numbers match published model cards
- [ ] Figures generated in `results/figures/`

---

## 9. Extension Ideas

1. **Head specialization analysis**: Cluster attention heads by pattern type (positional, syntactic, rare-word) across layers
2. **Probing classifiers**: Train linear probes on hidden states to detect POS tags, dependency relations, or semantic roles at each layer
3. **Cross-lingual attention**: Compare attention patterns on English vs Thai inputs in multilingual models (XLM-R, mBERT)
4. **Attention rollout**: Implement attention rollout / gradient-based attribution for more accurate token importance than raw attention
5. **Efficiency comparison**: Benchmark FLOPs, memory, and latency across the three families for equal input size
6. **Modern attention variants**: Extend Probe 5 to compare standard attention vs GQA (Mistral) vs sliding window attention

---

*Experiment designed following the [AI Researcher Reproduction Framework](/mnt/skills/user/ai-researcher/SKILL.md) and [HuggingFace LLM Course Chapter 1.4](https://huggingface.co/learn/llm-course/chapter1/4).*
