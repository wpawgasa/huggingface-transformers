"""Experiment 6: Question Answering (Extractive QA).

Pipeline: question-answering
Default model: distilbert-base-cased-distilled-squad
Architecture: Encoder-only (DistilBERT)

Course reference: HuggingFace LLM Course Chapter 1.3
"""

from transformers import pipeline as hf_pipeline

from src.pipeline_exploration.benchmarks import BenchmarkResult, benchmark_pipeline
from src.pipeline_exploration.data import QA_INPUTS
from src.pipeline_exploration.evaluate import validate_output

TASK = "question-answering"
DEFAULT_MODEL = "distilbert-base-cased-distilled-squad"


def load_pipeline(model: str = DEFAULT_MODEL, device: str = "cpu"):
    """Load the question-answering pipeline."""
    return hf_pipeline(TASK, model=model, device=device)


def run_experiment(device: str = "cpu") -> dict:
    """Run the full question-answering experiment.

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
    ce = QA_INPUTS["course_examples"]
    course_output = pipe(question=ce["question"], context=ce["context"])
    validate_output(TASK, course_output)
    results["course_examples"] = {
        "input": ce,
        "output": course_output,
    }

    # --- Edge cases ---
    edge_results = {}
    for name, item in QA_INPUTS["edge_cases"].items():
        output = pipe(question=item["question"], context=item["context"])
        validate_output(TASK, output)
        edge_results[name] = {"input": item, "output": output}
    results["edge_cases"] = edge_results

    # --- Benchmark (QA pipeline accepts dict input) ---
    bench_input = {"question": ce["question"], "context": ce["context"]}
    bm: BenchmarkResult = benchmark_pipeline(
        pipe_factory=lambda: load_pipeline(device=device),
        inputs=bench_input,
        task_name=TASK,
        model_name=DEFAULT_MODEL,
    )
    results["benchmark"] = bm.to_dict()

    return results
