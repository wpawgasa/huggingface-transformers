"""Experiment 3: Text Generation.

Pipeline: text-generation
Default model: openai-community/gpt2 (GPT-2)
Architecture: Decoder-only (GPT-2)

Ablations:
- Temperature sweep: temperature ∈ {0.7, 1.0, 1.5}
- Model comparison: GPT-2 vs HuggingFaceTB/SmolLM2-360M

Course reference: HuggingFace LLM Course Chapter 1.3
"""

from transformers import pipeline as hf_pipeline

from src.pipeline_exploration.benchmarks import BenchmarkResult, benchmark_pipeline
from src.pipeline_exploration.data import TEXT_GENERATION_INPUTS
from src.pipeline_exploration.evaluate import validate_output

TASK = "text-generation"
DEFAULT_MODEL = "openai-community/gpt2"
ALT_MODEL = "HuggingFaceTB/SmolLM2-360M"
MAX_NEW_TOKENS = 50


def load_pipeline(model: str = DEFAULT_MODEL, device: str = "cpu"):
    """Load the text-generation pipeline."""
    return hf_pipeline(TASK, model=model, device=device)


def run_experiment(device: str = "cpu") -> dict:
    """Run the full text-generation experiment.

    Steps:
        1. Load pipeline with default model (GPT-2).
        2. Run course examples and validate outputs.
        3. Run edge cases.
        4. Ablation: temperature sweep (0.7, 1.0, 1.5).
        5. Ablation: model comparison (GPT-2 vs SmolLM2-360M).
        6. Benchmark cold-start and warm inference.

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
    course_inputs = TEXT_GENERATION_INPUTS["course_examples"]
    # Call pipeline on each prompt individually; returns [{"generated_text": "..."}] per call
    course_outputs = []
    for prompt_text in course_inputs:
        outputs = pipe(prompt_text, max_new_tokens=MAX_NEW_TOKENS, num_return_sequences=1)
        for output in outputs:
            validate_output(TASK, output)
        course_outputs.append(outputs)
    results["course_examples"] = {
        "inputs": course_inputs,
        "outputs": course_outputs,
    }

    # --- Edge cases ---
    edge_results = {}
    for name, prompt in TEXT_GENERATION_INPUTS["edge_cases"].items():
        outputs = pipe(prompt, max_new_tokens=MAX_NEW_TOKENS, num_return_sequences=1)
        for output in outputs:
            validate_output(TASK, output)
        edge_results[name] = {"input": prompt, "outputs": outputs}
    results["edge_cases"] = edge_results

    # --- Ablation: temperature sweep ---
    temp_results = {}
    prompt = course_inputs[0]
    for temp in TEXT_GENERATION_INPUTS["ablation"]["temperatures"]:
        outputs = pipe(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temp,
            do_sample=True,
            num_return_sequences=1,
        )
        for output in outputs:
            validate_output(TASK, output)
        temp_results[f"temperature_{temp}"] = {
            "temperature": temp,
            "input": prompt,
            "outputs": outputs,
        }
    results["ablation"] = {"temperature_sweep": temp_results}

    # --- Ablation: model comparison ---
    alt_pipe = load_pipeline(model=ALT_MODEL, device=device)
    alt_outputs = alt_pipe(
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        num_return_sequences=1,
    )
    for output in alt_outputs:
        validate_output(TASK, output)
    results["ablation"]["model_comparison"] = {
        "default_model": DEFAULT_MODEL,
        "alt_model": ALT_MODEL,
        "input": prompt,
        "default_outputs": course_outputs[0],
        "alt_outputs": alt_outputs,
    }

    # --- Benchmark (default model, greedy decoding for reproducibility) ---
    bm: BenchmarkResult = benchmark_pipeline(
        pipe_factory=lambda: load_pipeline(device=device),
        inputs=prompt,
        task_name=TASK,
        model_name=DEFAULT_MODEL,
    )
    results["benchmark"] = bm.to_dict()

    return results
