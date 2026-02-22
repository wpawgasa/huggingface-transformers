"""Experiment 5: Named Entity Recognition (NER).

Pipeline: ner
Default model: dbmdz/bert-large-cased-finetuned-conll03-english
Architecture: Encoder-only (BERT-large)

Ablation: grouped_entities — True (merge subword tokens) vs False (raw tokens)

Course reference: HuggingFace LLM Course Chapter 1.3
"""

from transformers import pipeline as hf_pipeline

from src.benchmarks import BenchmarkResult, benchmark_pipeline
from src.data import NER_INPUTS
from src.evaluate import validate_output

TASK = "ner"
DEFAULT_MODEL = "dbmdz/bert-large-cased-finetuned-conll03-english"


def load_pipeline(
    model: str = DEFAULT_MODEL,
    device: str = "cpu",
    grouped_entities: bool = True,
):
    """Load the NER pipeline."""
    return hf_pipeline(TASK, model=model, device=device, grouped_entities=grouped_entities)


def run_experiment(device: str = "cpu") -> dict:
    """Run the full NER experiment.

    Steps:
        1. Load pipeline with grouped_entities=True (default).
        2. Run course examples and validate outputs.
        3. Run edge cases.
        4. Ablation: grouped_entities=True vs False.
        5. Benchmark cold-start and warm inference.

    Args:
        device: Device to run on ("cpu", "cuda", or "cuda:N").

    Returns:
        dict with keys: task, model, device, course_examples, edge_cases, ablation, benchmark.
    """
    pipe = load_pipeline(device=device, grouped_entities=True)

    results: dict = {
        "task": TASK,
        "model": DEFAULT_MODEL,
        "device": device,
    }

    # --- Course examples (grouped entities) ---
    course_input = NER_INPUTS["course_examples"][0]
    course_outputs = pipe(course_input)
    for entity in course_outputs:
        validate_output(TASK, entity)
    results["course_examples"] = {
        "input": course_input,
        "grouped_entities": True,
        "outputs": course_outputs,
    }

    # --- Edge cases ---
    edge_results = {}
    for name, text in NER_INPUTS["edge_cases"].items():
        entities = pipe(text)
        for entity in entities:
            validate_output(TASK, entity)
        edge_results[name] = {"input": text, "outputs": entities}
    results["edge_cases"] = edge_results

    # --- Ablation: grouped_entities=True vs False ---
    ungrouped_pipe = load_pipeline(device=device, grouped_entities=False)
    ungrouped_outputs = ungrouped_pipe(course_input)
    for entity in ungrouped_outputs:
        validate_output(TASK, entity)

    results["ablation"] = {
        "grouped_entities": {
            "grouped_true": {
                "grouped_entities": True,
                "input": course_input,
                "outputs": course_outputs,
                "n_entities": len(course_outputs),
            },
            "grouped_false": {
                "grouped_entities": False,
                "input": course_input,
                "outputs": ungrouped_outputs,
                "n_entities": len(ungrouped_outputs),
            },
        }
    }

    # --- Benchmark (grouped entities) ---
    bm: BenchmarkResult = benchmark_pipeline(
        pipe_factory=lambda: load_pipeline(device=device, grouped_entities=True),
        inputs=course_input,
        task_name=TASK,
        model_name=DEFAULT_MODEL,
    )
    results["benchmark"] = bm.to_dict()

    return results
