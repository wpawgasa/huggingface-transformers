"""Probe 1: Transformer History.

Reference: Ch1.4 -- "A bit of Transformer history"

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
    """A single milestone in the Transformer timeline."""

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
        key_innovation=("First pretrained Transformer; causal LM + fine-tuning"),
        hf_checkpoint="openai-gpt",
    ),
    TransformerMilestone(
        name="BERT",
        date="October 2018",
        params="110M / 340M",
        family="encoder-only",
        architecture="12/24-layer encoder",
        key_innovation=("Bidirectional context via Masked LM + Next Sentence Prediction"),
        hf_checkpoint="bert-base-uncased",
    ),
    TransformerMilestone(
        name="GPT-2",
        date="February 2019",
        params="1.5B",
        family="decoder-only",
        architecture="48-layer decoder",
        key_innovation=("Scaled-up GPT; zero-shot task transfer; " "delayed release for ethics"),
        hf_checkpoint="gpt2",
    ),
    TransformerMilestone(
        name="T5",
        date="October 2019",
        params="60M-11B",
        family="encoder-decoder",
        architecture="Encoder-decoder with text-to-text framing",
        key_innovation=("Unified all NLP tasks as text-to-text; " "span corruption pretraining"),
        hf_checkpoint="t5-small",
    ),
    TransformerMilestone(
        name="GPT-3",
        date="May 2020",
        params="175B",
        family="decoder-only",
        architecture="96-layer decoder",
        key_innovation=("In-context learning; few-shot / zero-shot without fine-tuning"),
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
        params="7B-65B",
        family="decoder-only",
        architecture="Decoder with RMSNorm, SwiGLU, RoPE",
        key_innovation=("Open-weight LLM; efficient training on public data"),
        hf_checkpoint="meta-llama/Llama-2-7b-hf",
    ),
    TransformerMilestone(
        name="Mistral",
        date="March 2023",
        params="7B",
        family="decoder-only",
        architecture="Decoder with GQA + Sliding Window Attention",
        key_innovation=("Grouped-query attention; sliding window for long sequences"),
        hf_checkpoint="mistralai/Mistral-7B-v0.1",
    ),
    TransformerMilestone(
        name="Gemma 2",
        date="May 2024",
        params="2B-27B",
        family="decoder-only",
        architecture="Decoder with interleaved local-global attention",
        key_innovation=("Knowledge distillation; local-global attention interleaving"),
        hf_checkpoint="google/gemma-2-2b",
    ),
    TransformerMilestone(
        name="SmolLM2",
        date="November 2024",
        params="135M-1.7B",
        family="decoder-only",
        architecture="Compact decoder",
        key_innovation=("State-of-the-art at small scale; edge/mobile deployment"),
        hf_checkpoint="HuggingFaceTB/SmolLM2-360M",
    ),
]


def run_experiment() -> dict:
    """Build timeline and classify by architecture family.

    Returns:
        Dict with timeline entries, family distribution, and observations.
    """
    results: dict = {"task": "transformer_timeline"}

    # Family distribution
    family_counts: dict[str, int] = {}
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
        "The course timeline shows a clear trend: decoder-only models"
        " dominate modern LLMs"
        f" ({family_counts.get('decoder-only', 0)}/{len(TIMELINE)} models)."
        " Encoder-only (BERT-like) peaked in 2018-2019 for understanding"
        " tasks, while encoder-decoder (T5-like) remains relevant for"
        " seq2seq. Post-2022 models are almost exclusively decoder-only"
        " with attention optimizations (GQA, sliding window,"
        " local-global interleaving)."
    )

    return results
