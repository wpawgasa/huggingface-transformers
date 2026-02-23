"""Experiment 9: Image Classification.

Pipeline: image-classification
Default model: google/vit-base-patch16-224 (ViT-base)
Architecture: Vision Transformer (ViT)

Course reference: HuggingFace LLM Course Chapter 1.3
"""

from transformers import pipeline as hf_pipeline

from src.benchmarks import BenchmarkResult, benchmark_pipeline
from src.data import IMAGE_CLASSIFICATION_INPUTS
from src.evaluate import validate_output

TASK = "image-classification"
DEFAULT_MODEL = "google/vit-base-patch16-224"


def load_pipeline(model: str = DEFAULT_MODEL, device: str = "cpu"):
    """Load the image-classification pipeline."""
    return hf_pipeline(TASK, model=model, device=device)


def run_experiment(device: str = "cpu") -> dict:
    """Run the full image-classification experiment.

    Steps:
        1. Load pipeline with ViT-base-patch16-224.
        2. Run course examples (cat image URL) and validate outputs.
        3. Run edge cases.
        4. Benchmark cold-start and warm inference.

    Args:
        device: Device to run on ("cpu", "cuda", or "cuda:N").

    Returns:
        dict with keys: task, model, device, course_examples, edge_cases, benchmark.

    Note:
        Requires Pillow and an internet connection to download the images.
    """
    pipe = load_pipeline(device=device)

    results: dict = {
        "task": TASK,
        "model": DEFAULT_MODEL,
        "device": device,
    }

    # --- Course examples ---
    course_inputs = IMAGE_CLASSIFICATION_INPUTS["course_examples"]
    course_outputs = pipe(course_inputs[0])
    # pipeline returns a list of top-k predictions per image
    for output in course_outputs:
        validate_output(TASK, output)
    results["course_examples"] = {
        "input": course_inputs[0],
        "outputs": course_outputs,
        "top_prediction": course_outputs[0],
    }

    # --- Edge cases ---
    edge_results = {}
    for name, image_url in IMAGE_CLASSIFICATION_INPUTS["edge_cases"].items():
        outputs = pipe(image_url)
        for output in outputs:
            validate_output(TASK, output)
        edge_results[name] = {"input": image_url, "outputs": outputs}
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
