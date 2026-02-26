"""Output schema validation for all pipeline tasks.

Each validate_* function checks a single result dict (one item from the pipeline output)
for required keys and value constraints. Raises ValueError on invalid output.
"""


def validate_output(task: str, output: dict) -> None:
    """Dispatch to the task-specific validator.

    Args:
        task: Pipeline task name (e.g., "text-classification").
        output: A single output dict returned by the pipeline.

    Raises:
        ValueError: If the task is unknown or output is invalid.
    """
    validators = {
        "text-classification": _validate_text_classification,
        "zero-shot-classification": _validate_zero_shot,
        "text-generation": _validate_text_generation,
        "fill-mask": _validate_fill_mask,
        "ner": _validate_ner,
        "question-answering": _validate_question_answering,
        "summarization": _validate_summarization,
        "translation": _validate_translation,
        "image-classification": _validate_image_classification,
        "automatic-speech-recognition": _validate_asr,
    }
    if task not in validators:
        raise ValueError(f"Unknown task: {task!r}. Valid tasks: {sorted(validators)}")
    validators[task](output)


def _check_keys(output: dict, required: set[str], task: str) -> None:
    """Assert all required keys are present."""
    missing = required - set(output.keys())
    if missing:
        raise ValueError(f"[{task}] Missing keys in output: {missing}. Got: {set(output.keys())}")


def _check_score(score: float, task: str, field: str = "score") -> None:
    """Assert score is in [0, 1]."""
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"[{task}] {field} out of range [0, 1]: {score}")


# ---------------------------------------------------------------------------
# Task-specific validators
# ---------------------------------------------------------------------------


def _validate_text_classification(output: dict) -> None:
    """Validate sentiment / text-classification output.

    Required: label (str), score (float ∈ [0, 1])
    """
    _check_keys(output, {"label", "score"}, "text-classification")
    if not isinstance(output["label"], str) or not output["label"]:
        raise ValueError(
            f"[text-classification] 'label' must be a non-empty string: {output['label']!r}"
        )
    _check_score(float(output["score"]), "text-classification")


def _validate_zero_shot(output: dict) -> None:
    """Validate zero-shot classification output.

    Required: sequence (str), labels (list[str]), scores (list[float])
    Constraint: scores must sum ≈ 1.0 (within 0.01 tolerance)
    """
    _check_keys(output, {"sequence", "labels", "scores"}, "zero-shot-classification")
    if not isinstance(output["sequence"], str):
        raise ValueError("[zero-shot-classification] 'sequence' must be a string")
    if not isinstance(output["labels"], list) or not output["labels"]:
        raise ValueError("[zero-shot-classification] 'labels' must be a non-empty list")
    if not isinstance(output["scores"], list) or not output["scores"]:
        raise ValueError("[zero-shot-classification] 'scores' must be a non-empty list")
    if len(output["labels"]) != len(output["scores"]):
        raise ValueError(
            "[zero-shot-classification] 'labels' and 'scores' must have the same length"
        )
    for s in output["scores"]:
        _check_score(float(s), "zero-shot-classification")
    total = sum(output["scores"])
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"[zero-shot-classification] scores must sum ≈ 1.0, got {total:.4f}")


def _validate_text_generation(output: dict) -> None:
    """Validate text-generation output.

    Required: generated_text (non-empty str)
    """
    _check_keys(output, {"generated_text"}, "text-generation")
    if not isinstance(output["generated_text"], str) or not output["generated_text"].strip():
        raise ValueError("[text-generation] 'generated_text' must be a non-empty string")


def _validate_fill_mask(output: dict) -> None:
    """Validate fill-mask output.

    Required: sequence (str), score (float ∈ [0, 1]), token (int), token_str (str)
    """
    _check_keys(output, {"sequence", "score", "token", "token_str"}, "fill-mask")
    if not isinstance(output["sequence"], str):
        raise ValueError("[fill-mask] 'sequence' must be a string")
    _check_score(float(output["score"]), "fill-mask")
    if not isinstance(output["token"], int):
        raise ValueError(f"[fill-mask] 'token' must be an int, got: {type(output['token'])}")
    if not isinstance(output["token_str"], str):
        raise ValueError("[fill-mask] 'token_str' must be a string")


def _validate_ner(output: dict) -> None:
    """Validate NER (named entity recognition) output.

    Supports both grouped (entity_group) and ungrouped (entity) output.
    Required (grouped): entity_group (str), word (str), score (float ∈ [0, 1]), start (int), end (int)
    Required (ungrouped): entity (str), word (str), score (float ∈ [0, 1]), start (int), end (int)
    """
    task = "ner"
    # Accept either grouped or ungrouped format
    if "entity_group" in output:
        _check_keys(output, {"entity_group", "word", "score", "start", "end"}, task)
        entity_key = "entity_group"
    elif "entity" in output:
        _check_keys(output, {"entity", "word", "score", "start", "end"}, task)
        entity_key = "entity"
    else:
        raise ValueError(
            f"[{task}] Output must contain 'entity_group' or 'entity' key. Got: {set(output.keys())}"
        )

    if not isinstance(output[entity_key], str) or not output[entity_key]:
        raise ValueError(f"[{task}] '{entity_key}' must be a non-empty string")
    if not isinstance(output["word"], str):
        raise ValueError(f"[{task}] 'word' must be a string")
    _check_score(float(output["score"]), task)
    if not isinstance(output["start"], int) or not isinstance(output["end"], int):
        raise ValueError(f"[{task}] 'start' and 'end' must be ints")
    if output["start"] < 0 or output["end"] < output["start"]:
        raise ValueError(f"[{task}] Invalid span: start={output['start']}, end={output['end']}")


def _validate_question_answering(output: dict) -> None:
    """Validate question-answering output.

    Required: answer (str), score (float ∈ [0, 1]), start (int), end (int)
    """
    _check_keys(output, {"answer", "score", "start", "end"}, "question-answering")
    if not isinstance(output["answer"], str):
        raise ValueError("[question-answering] 'answer' must be a string")
    _check_score(float(output["score"]), "question-answering")
    if not isinstance(output["start"], int) or not isinstance(output["end"], int):
        raise ValueError("[question-answering] 'start' and 'end' must be ints")


def _validate_summarization(output: dict) -> None:
    """Validate summarization output.

    Required: summary_text (non-empty str)
    """
    _check_keys(output, {"summary_text"}, "summarization")
    if not isinstance(output["summary_text"], str) or not output["summary_text"].strip():
        raise ValueError("[summarization] 'summary_text' must be a non-empty string")


def _validate_translation(output: dict) -> None:
    """Validate translation output.

    Required: translation_text (non-empty str)
    """
    _check_keys(output, {"translation_text"}, "translation")
    if not isinstance(output["translation_text"], str) or not output["translation_text"].strip():
        raise ValueError("[translation] 'translation_text' must be a non-empty string")


def _validate_image_classification(output: dict) -> None:
    """Validate image-classification output.

    Required: label (str), score (float ∈ [0, 1])
    """
    _check_keys(output, {"label", "score"}, "image-classification")
    if not isinstance(output["label"], str) or not output["label"]:
        raise ValueError("[image-classification] 'label' must be a non-empty string")
    _check_score(float(output["score"]), "image-classification")


def _validate_asr(output: dict) -> None:
    """Validate automatic-speech-recognition output.

    Required: text (non-empty str)
    """
    _check_keys(output, {"text"}, "automatic-speech-recognition")
    if not isinstance(output["text"], str) or not output["text"].strip():
        raise ValueError("[automatic-speech-recognition] 'text' must be a non-empty string")
