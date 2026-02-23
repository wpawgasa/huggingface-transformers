"""Experiment 1: Text Classification (Sentiment Analysis).

Pipeline: text-classification
Default model: distilbert-base-uncased-finetuned-sst-2-english
Architecture: Encoder-only (DistilBERT)

Course reference: HuggingFace LLM Course Chapter 1.3
"""

from transformers import pipeline as hf_pipeline

from src.benchmarks import BenchmarkResult, benchmark_pipeline
from src.data import TEXT_CLASSIFICATION_INPUTS
from src.evaluate import validate_output

TASK = "text-classification"
DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def load_pipeline(model: str = DEFAULT_MODEL, device: str = "cpu"):
    """Load the text-classification pipeline."""
    return hf_pipeline(TASK, model=model, device=device)


def run_experiment(device: str = "cpu") -> dict:
    """Run the full text-classification experiment.

    Steps:
        1. Load pipeline with default model.
        2. Run course examples and validate outputs.
        3. Run edge cases.
        4. Benchmark cold-start and warm inference.

    Args:
        device: Device to run on ("cpu", "cuda", or "cuda:N").

    Returns:
        dict with keys: task, model, device, course_examples, edge_cases, benchmark.
    """
    pipe = load_pipeline(device=device)

    results: dict = {
        "task": TASK,
        "model": DEFAULT_MODEL,
        "device": device,
    }

    # --- Course examples ---
    course_inputs = TEXT_CLASSIFICATION_INPUTS["course_examples"]
    course_outputs = pipe(course_inputs)
    for output in course_outputs:
        validate_output(TASK, output)
    results["course_examples"] = {
        "inputs": course_inputs,
        "outputs": course_outputs,
    }

    # --- Edge cases ---
    edge_results = {}
    for name, text in TEXT_CLASSIFICATION_INPUTS["edge_cases"].items():
        if not text:  # skip empty string (model may error on empty input)
            edge_results[name] = {"input": text, "output": None, "skipped": True}
            continue
        output = pipe(text)[0]
        validate_output(TASK, output)
        edge_results[name] = {"input": text, "output": output}
    results["edge_cases"] = edge_results

    # --- Benchmark ---
    bm: BenchmarkResult = benchmark_pipeline(
        pipe_factory=lambda: load_pipeline(device=device),
        inputs=course_inputs[0],
        task_name=TASK,
        model_name=DEFAULT_MODEL,
    )
    results["benchmark"] = bm.to_dict()

    return results
