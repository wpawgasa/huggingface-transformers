"""Experiment 10: Automatic Speech Recognition (ASR).

Pipeline: automatic-speech-recognition
Default model (GPU): openai/whisper-large-v3
Default model (CPU): openai/whisper-tiny

Ablation: CPU model substitution — whisper-tiny vs whisper-large-v3

Architecture: Encoder-decoder (Whisper)

Course reference: HuggingFace LLM Course Chapter 1.3
"""

from transformers import pipeline as hf_pipeline

from src.benchmarks import BenchmarkResult, benchmark_pipeline
from src.data import SPEECH_RECOGNITION_INPUTS
from src.evaluate import validate_output

TASK = "automatic-speech-recognition"
GPU_MODEL = "openai/whisper-large-v3"
CPU_MODEL = "openai/whisper-tiny"


def _select_model(device: str) -> str:
    """Select the appropriate model based on device."""
    return GPU_MODEL if device.startswith("cuda") else CPU_MODEL


def load_pipeline(model: str | None = None, device: str = "cpu"):
    """Load the ASR pipeline, selecting model based on device if not specified."""
    if model is None:
        model = _select_model(device)
    return hf_pipeline(TASK, model=model, device=device)


def run_experiment(device: str = "cpu") -> dict:
    """Run the full ASR experiment.

    Steps:
        1. Load pipeline with appropriate model (whisper-tiny on CPU, large-v3 on GPU).
        2. Run course examples (MLK speech excerpt) and validate outputs.
        3. Run edge cases.
        4. Ablation: note model substitution (CPU tiny vs GPU large-v3).
        5. Benchmark cold-start and warm inference.

    Args:
        device: Device to run on ("cpu", "cuda", or "cuda:N").

    Returns:
        dict with keys: task, model, device, course_examples, edge_cases, ablation, benchmark.

    Note:
        Requires soundfile and librosa (both declared in project dependencies) plus
        an internet connection to download audio files.
    """
    model_name = _select_model(device)
    pipe = load_pipeline(model=model_name, device=device)

    results: dict = {
        "task": TASK,
        "model": model_name,
        "device": device,
    }

    # --- Course examples ---
    course_input = SPEECH_RECOGNITION_INPUTS["course_examples"][0]
    course_output = pipe(course_input)
    validate_output(TASK, course_output)
    results["course_examples"] = {
        "input": course_input,
        "output": course_output,
    }

    # --- Edge cases ---
    edge_results = {}
    for name, audio_url in SPEECH_RECOGNITION_INPUTS["edge_cases"].items():
        output = pipe(audio_url)
        validate_output(TASK, output)
        edge_results[name] = {"input": audio_url, "output": output}
    results["edge_cases"] = edge_results

    # --- Ablation: model substitution note ---
    results["ablation"] = {
        "model_substitution": {
            "description": "whisper-tiny on CPU vs whisper-large-v3 on GPU",
            "cpu_model": CPU_MODEL,
            "gpu_model": GPU_MODEL,
            "active_model": model_name,
            "device": device,
            "note": (
                f"Running {model_name!r} on {device!r}. "
                "For full ablation, run once on CPU (whisper-tiny) and once on GPU (whisper-large-v3)."
            ),
        }
    }

    # --- Benchmark ---
    bm: BenchmarkResult = benchmark_pipeline(
        pipe_factory=lambda: load_pipeline(model=model_name, device=device),
        inputs=course_input,
        task_name=TASK,
        model_name=model_name,
    )
    results["benchmark"] = bm.to_dict()

    return results
