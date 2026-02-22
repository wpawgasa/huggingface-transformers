"""CLI orchestrator for Transformers Pipeline Experiments.

Usage:
    # Run all 10 tasks (auto-detect GPU)
    python -m src.experiment_runner --device auto --tasks all --output results/outputs.json

    # Run a subset
    python -m src.experiment_runner --device cpu \\
        --tasks text-classification,zero-shot,fill-mask \\
        --output results/text_only.json
"""

import argparse
import csv
import json
import os
import sys
from collections.abc import Callable

import torch

from src.pipelines import (
    fill_mask,
    image_classification,
    ner,
    question_answering,
    speech_recognition,
    summarization,
    text_classification,
    text_generation,
    translation,
    zero_shot,
)

# Registry mapping task name → run_experiment function
TASK_REGISTRY: dict[str, Callable[..., dict]] = {
    "text-classification": text_classification.run_experiment,
    "zero-shot": zero_shot.run_experiment,
    "text-generation": text_generation.run_experiment,
    "fill-mask": fill_mask.run_experiment,
    "ner": ner.run_experiment,
    "question-answering": question_answering.run_experiment,
    "summarization": summarization.run_experiment,
    "translation": translation.run_experiment,
    "image-classification": image_classification.run_experiment,
    "speech-recognition": speech_recognition.run_experiment,
}


def resolve_device(device: str) -> str:
    """Resolve 'auto' to 'cuda' or 'cpu' based on hardware availability.

    Args:
        device: One of "auto", "cpu", or "cuda" (or "cuda:N").

    Returns:
        Resolved device string ("cpu" or "cuda").
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def parse_tasks(tasks_str: str) -> list[str]:
    """Parse a comma-separated task list or the special value 'all'.

    Args:
        tasks_str: "all" or comma-separated task names.

    Returns:
        List of task name strings.
    """
    if tasks_str.strip().lower() == "all":
        return list(TASK_REGISTRY.keys())
    return [t.strip() for t in tasks_str.split(",") if t.strip()]


def run_tasks(
    tasks: list[str],
    device: str,
    output_path: str,
    benchmark_path: str,
) -> dict:
    """Run the requested pipeline experiments and persist results.

    Args:
        tasks: List of task names to run.
        device: Resolved device string ("cpu" or "cuda").
        output_path: Path to write the JSON results file.
        benchmark_path: Path to write the CSV benchmark file.

    Returns:
        Dict mapping task name → result dict.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Running {len(tasks)} task(s) on device: {device}")
    print(f"Tasks: {tasks}\n")

    all_results: dict = {}
    benchmark_rows: list[dict] = []

    for task_name in tasks:
        print(f"{'=' * 60}")
        print(f"Running: {task_name}")
        print(f"{'=' * 60}")

        try:
            result = TASK_REGISTRY[task_name](device=device)
            all_results[task_name] = result

            if "benchmark" in result:
                bm = result["benchmark"]
                benchmark_rows.append(
                    {
                        "task": task_name,
                        "model": result.get("model", "unknown"),
                        "cold_start_ms": bm.get("cold_start_ms", 0),
                        "warm_latency_ms": bm.get("warm_latency_ms", 0),
                        "throughput_samples_per_sec": bm.get("throughput_samples_per_sec", 0),
                        "num_warm_runs": bm.get("num_warm_runs", 0),
                    }
                )

            print(f"  OK {task_name} completed\n")

        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {task_name} failed: {exc}\n", file=sys.stderr)
            all_results[task_name] = {"error": str(exc)}

    # Write JSON results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # Write benchmark CSV
    if benchmark_rows:
        os.makedirs(os.path.dirname(benchmark_path) or ".", exist_ok=True)
        fieldnames = [
            "task",
            "model",
            "cold_start_ms",
            "warm_latency_ms",
            "throughput_samples_per_sec",
            "num_warm_runs",
        ]
        with open(benchmark_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(benchmark_rows)
        print(f"Benchmarks saved to: {benchmark_path}")

    return all_results


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run HuggingFace Transformers Pipeline Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on: 'cpu', 'cuda', 'cuda:N', or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--tasks",
        default="all",
        help=f"Comma-separated task names or 'all'. Valid tasks: {', '.join(TASK_REGISTRY)}",
    )
    parser.add_argument(
        "--output",
        default="results/outputs.json",
        help="Path to write JSON results (default: results/outputs.json)",
    )
    parser.add_argument(
        "--benchmark-output",
        default="results/benchmarks.csv",
        help="Path to write benchmark CSV (default: results/benchmarks.csv)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the experiment runner CLI.

    Args:
        argv: Command-line arguments (defaults to sys.argv).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    device = resolve_device(args.device)
    tasks = parse_tasks(args.tasks)

    # Validate task names
    invalid = [t for t in tasks if t not in TASK_REGISTRY]
    if invalid:
        print(f"Error: Unknown task(s): {invalid}", file=sys.stderr)
        print(f"Valid tasks: {sorted(TASK_REGISTRY)}", file=sys.stderr)
        return 1

    if not tasks:
        print("Error: No tasks specified.", file=sys.stderr)
        return 1

    run_tasks(
        tasks=tasks,
        device=device,
        output_path=args.output,
        benchmark_path=args.benchmark_output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
