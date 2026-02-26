"""Shared test inputs for all architecture probes.

Sentences chosen to highlight attention behavior from the course:
  - "You like this course" (translation example from Ch1.4)
  - Subject-verb agreement (attention distance test)
  - Coreference resolution (long-range dependency test)
"""

# ------------------------------------------------------------------
# Core sentences from the course
# ------------------------------------------------------------------
COURSE_SENTENCES: dict = {
    "translation_example": "You like this course",
    "french_target": "Vous aimez ce cours",
    "attention_demo": "The animal didn't cross the street because it was too tired",
}

# ------------------------------------------------------------------
# Attention probing sentences
# ------------------------------------------------------------------
ATTENTION_SENTENCES: dict = {
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

# ------------------------------------------------------------------
# Language modeling test prompts
# ------------------------------------------------------------------
LM_INPUTS: dict = {
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

# ------------------------------------------------------------------
# Transfer learning dataset (simple sentiment)
# ------------------------------------------------------------------
TRANSFER_LEARNING_DATA: dict = {
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

# ------------------------------------------------------------------
# Model registry — architecture families
# ------------------------------------------------------------------
MODEL_REGISTRY: dict = {
    "encoder_only": {
        "primary": "bert-base-uncased",
        "small": "google/bert_uncased_L-2_H-128_A-2",
        "family": "BERT-like (auto-encoding)",
        "objective": "Masked Language Modeling (MLM)",
        "attention": "bidirectional",
    },
    "decoder_only": {
        "primary": "gpt2",
        "small": "sshleifer/tiny-gpt2",
        "family": "GPT-like (auto-regressive)",
        "objective": "Causal Language Modeling (CLM)",
        "attention": "causal (left-to-right)",
    },
    "encoder_decoder": {
        "primary": "t5-small",
        "small": "google/t5-efficient-tiny",
        "family": "T5-like (sequence-to-sequence)",
        "objective": "Span corruption / Denoising",
        "attention": "bidirectional encoder + causal decoder",
    },
}
