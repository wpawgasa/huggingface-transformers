"""Experiment 8: Translation (French → English).

Pipeline: translation
Default model: Helsinki-NLP/opus-mt-fr-en
Architecture: Encoder-decoder (MarianMT)

Course reference: HuggingFace LLM Course Chapter 1.3
"""

from transformers import pipeline as hf_pipeline

from src.pipeline_exploration.benchmarks import BenchmarkResult, benchmark_pipeline
from src.pipeline_exploration.data import TRANSLATION_INPUTS
from src.pipeline_exploration.evaluate import validate_output

TASK = "translation"
DEFAULT_MODEL = "Helsinki-NLP/opus-mt-fr-en"


def load_pipeline(model: str = DEFAULT_MODEL, device: str = "cpu"):
    """Load the translation pipeline."""
    return hf_pipeline(TASK, model=model, device=device)


def run_experiment(device: str = "cpu") -> dict:
    """Run the full translation experiment.

    Steps:
        1. Load pipeline with Helsinki-NLP/opus-mt-fr-en (French → English).
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
    course_inputs = TRANSLATION_INPUTS["course_examples"]
    course_outputs = pipe(course_inputs)
    for output in course_outputs:
        validate_output(TASK, output)
    results["course_examples"] = {
        "inputs": course_inputs,
        "outputs": course_outputs,
    }

    # --- Edge cases ---
    edge_results = {}
    for name, text in TRANSLATION_INPUTS["edge_cases"].items():
        outputs = pipe(text)
        output = outputs[0] if isinstance(outputs, list) else outputs
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
