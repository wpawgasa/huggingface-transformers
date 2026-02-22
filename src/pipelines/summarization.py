"""Experiment 7: Summarization.

Pipeline: summarization
Default model: sshleifer/distilbart-cnn-12-6 (DistilBART-CNN)
Architecture: Encoder-decoder (BART)

Ablation: summary length — max_length ∈ {50, 100, 200}

Course reference: HuggingFace LLM Course Chapter 1.3
"""

from transformers import pipeline as hf_pipeline

from src.benchmarks import BenchmarkResult, benchmark_pipeline
from src.data import SUMMARIZATION_INPUTS
from src.evaluate import validate_output

TASK = "summarization"
DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"


def load_pipeline(model: str = DEFAULT_MODEL, device: str = "cpu"):
    """Load the summarization pipeline."""
    return hf_pipeline(TASK, model=model, device=device)


def run_experiment(device: str = "cpu") -> dict:
    """Run the full summarization experiment.

    Steps:
        1. Load pipeline with default model.
        2. Run course examples and validate outputs.
        3. Run edge cases.
        4. Ablation: max_length sweep (50, 100, 200 tokens).
        5. Benchmark cold-start and warm inference.

    Args:
        device: Device to run on ("cpu", "cuda", or "cuda:N").

    Returns:
        dict with keys: task, model, device, course_examples, edge_cases, ablation, benchmark.
    """
    pipe = load_pipeline(device=device)

    results: dict = {
        "task": TASK,
        "model": DEFAULT_MODEL,
        "device": device,
    }

    # --- Course examples ---
    course_inputs = SUMMARIZATION_INPUTS["course_examples"]
    course_outputs = pipe(course_inputs, max_length=130, min_length=30)
    for output in course_outputs:
        validate_output(TASK, output)
    results["course_examples"] = {
        "inputs": course_inputs,
        "outputs": course_outputs,
    }

    # --- Edge cases ---
    edge_results = {}
    for name, text in SUMMARIZATION_INPUTS["edge_cases"].items():
        outputs = pipe(text, max_length=100, min_length=10)
        for output in outputs if isinstance(outputs, list) else [outputs]:
            validate_output(TASK, output)
        edge_results[name] = {"input": text, "outputs": outputs}
    results["edge_cases"] = edge_results

    # --- Ablation: max_length sweep ---
    ablation_text = course_inputs[0]
    length_results = {}
    for max_len in SUMMARIZATION_INPUTS["ablation"]["max_lengths"]:
        outputs = pipe(ablation_text, max_length=max_len, min_length=min(20, max_len - 10))
        for output in outputs if isinstance(outputs, list) else [outputs]:
            validate_output(TASK, output)
        length_results[f"max_length_{max_len}"] = {
            "max_length": max_len,
            "outputs": outputs,
        }
    results["ablation"] = {"summary_length": length_results}

    # --- Benchmark ---
    bm: BenchmarkResult = benchmark_pipeline(
        pipe_factory=lambda: load_pipeline(device=device),
        inputs=course_inputs[0],
        task_name=TASK,
        model_name=DEFAULT_MODEL,
    )
    results["benchmark"] = bm.to_dict()

    return results
