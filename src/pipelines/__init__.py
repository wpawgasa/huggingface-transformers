"""Pipeline experiment modules.

Each module implements one pipeline task from the HuggingFace LLM Course Chapter 1.3
and exports a run_experiment(device) -> dict function.
"""

from src.pipelines import (
    fill_mask,
    image_classification,
    ner,
    question_answering,
    speech_recognition,
    summarization,
    text_classification,
    text_generation,
    translation,
    zero_shot,
)

__all__ = [
    "text_classification",
    "zero_shot",
    "text_generation",
    "fill_mask",
    "ner",
    "question_answering",
    "summarization",
    "translation",
    "image_classification",
    "speech_recognition",
]
