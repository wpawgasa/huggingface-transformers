"""Experiment 2: Zero-Shot Classification.

Pipeline: zero-shot-classification
Default model: facebook/bart-large-mnli
Architecture: Encoder-decoder (BART)

Ablation: label count scaling — n_labels ∈ {2, 3, 5, 10}

Course reference: HuggingFace LLM Course Chapter 1.3
"""

from transformers import pipeline as hf_pipeline

from src.benchmarks import BenchmarkResult, benchmark_pipeline
from src.data import ZERO_SHOT_INPUTS
from src.evaluate import validate_output

TASK = "zero-shot-classification"
DEFAULT_MODEL = "facebook/bart-large-mnli"

# Label pool used for the ablation study
_LABEL_POOL = [
    "education",
    "politics",
    "business",
    "technology",
    "science",
    "sports",
    "entertainment",
    "health",
    "environment",
    "culture",
]
_ABLATION_N_LABELS = [2, 3, 5, 10]


def load_pipeline(model: str = DEFAULT_MODEL, device: str = "cpu"):
    """Load the zero-shot-classification pipeline."""
    return hf_pipeline(TASK, model=model, device=device)


def run_experiment(device: str = "cpu") -> dict:
    """Run the full zero-shot-classification experiment.

    Steps:
        1. Load pipeline with default model.
        2. Run course examples and validate outputs.
        3. Run edge cases (varying label counts).
        4. Ablation: label count scaling (2, 3, 5, 10 labels).
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
    ce = ZERO_SHOT_INPUTS["course_examples"]
    course_output = pipe(ce["sequence"], candidate_labels=ce["candidate_labels"])
    validate_output(TASK, course_output)
    results["course_examples"] = {
        "input": ce,
        "output": course_output,
    }

    # --- Edge cases ---
    edge_results = {}
    for name, item in ZERO_SHOT_INPUTS["edge_cases"].items():
        output = pipe(item["sequence"], candidate_labels=item["candidate_labels"])
        validate_output(TASK, output)
        edge_results[name] = {
            "n_labels": len(item["candidate_labels"]),
            "input": item,
            "output": output,
        }
    results["edge_cases"] = edge_results

    # --- Ablation: label count scaling ---
    ablation_results = {}
    for n in _ABLATION_N_LABELS:
        labels = _LABEL_POOL[:n]
        output = pipe(ce["sequence"], candidate_labels=labels)
        validate_output(TASK, output)
        ablation_results[f"n_labels_{n}"] = {
            "n_labels": n,
            "candidate_labels": labels,
            "output": output,
        }
    results["ablation"] = {"label_count_scaling": ablation_results}

    # --- Benchmark (use a wrapper so benchmark_pipeline can call pipe with one arg) ---
    _labels = ce["candidate_labels"]

    def _bench_factory():
        p = load_pipeline(device=device)

        def _wrapped(seq: str):
            return p(seq, candidate_labels=_labels)

        return _wrapped

    bm: BenchmarkResult = benchmark_pipeline(
        pipe_factory=_bench_factory,
        inputs=ce["sequence"],
        task_name=TASK,
        model_name=DEFAULT_MODEL,
    )
    results["benchmark"] = bm.to_dict()

    return results
