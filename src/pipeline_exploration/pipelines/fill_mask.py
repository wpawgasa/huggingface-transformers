"""Experiment 4: Mask Filling.

Pipeline: fill-mask
Default model: distilroberta-base (DistilRoBERTa, uses <mask> token)
Architecture: Encoder-only (DistilRoBERTa)

Ablation: mask token / model — DistilRoBERTa (<mask>) vs bert-base-cased ([MASK])

Course reference: HuggingFace LLM Course Chapter 1.3
"""

from transformers import pipeline as hf_pipeline

from src.pipeline_exploration.benchmarks import BenchmarkResult, benchmark_pipeline
from src.pipeline_exploration.data import FILL_MASK_INPUTS
from src.pipeline_exploration.evaluate import validate_output

TASK = "fill-mask"
DEFAULT_MODEL = "distilroberta-base"  # uses <mask>
BERT_MODEL = "bert-base-cased"  # uses [MASK]


def load_pipeline(model: str = DEFAULT_MODEL, device: str = "cpu"):
    """Load the fill-mask pipeline."""
    return hf_pipeline(TASK, model=model, device=device)


def run_experiment(device: str = "cpu") -> dict:
    """Run the full fill-mask experiment.

    Steps:
        1. Load pipeline with DistilRoBERTa (default).
        2. Run course examples and validate outputs.
        3. Run edge cases.
        4. Ablation: DistilRoBERTa (<mask>) vs BERT-base-cased ([MASK]).
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

    # --- Course examples (DistilRoBERTa with <mask>) ---
    course_input = FILL_MASK_INPUTS["course_examples"]["distilroberta"]
    course_outputs = pipe(course_input)
    # pipeline returns a list of top-k predictions
    for output in course_outputs:
        validate_output(TASK, output)
    results["course_examples"] = {
        "model": DEFAULT_MODEL,
        "input": course_input,
        "outputs": course_outputs,
    }

    # --- Edge cases ---
    edge_results = {}
    for name, text in FILL_MASK_INPUTS["edge_cases"].items():
        outputs = pipe(text)
        for output in outputs:
            validate_output(TASK, output)
        edge_results[name] = {"input": text, "outputs": outputs}
    results["edge_cases"] = edge_results

    # --- Ablation: BERT-base-cased with [MASK] token ---
    bert_pipe = load_pipeline(model=BERT_MODEL, device=device)
    bert_input = FILL_MASK_INPUTS["course_examples"]["bert"]
    bert_outputs = bert_pipe(bert_input)
    for output in bert_outputs:
        validate_output(TASK, output)
    results["ablation"] = {
        "mask_token_model_swap": {
            "distilroberta": {
                "model": DEFAULT_MODEL,
                "mask_token": "<mask>",
                "input": course_input,
                "outputs": course_outputs,
            },
            "bert": {
                "model": BERT_MODEL,
                "mask_token": "[MASK]",
                "input": bert_input,
                "outputs": bert_outputs,
            },
        }
    }

    # --- Benchmark ---
    bm: BenchmarkResult = benchmark_pipeline(
        pipe_factory=lambda: load_pipeline(device=device),
        inputs=course_input,
        task_name=TASK,
        model_name=DEFAULT_MODEL,
    )
    results["benchmark"] = bm.to_dict()

    return results
