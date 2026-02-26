"""CLI orchestrator for Transformer Architecture Deep Dive Experiments.

Usage:
    # Run all 6 probes (auto-detect GPU)
    python -m src.architecture_deepdive.experiment_runner --device auto --probes all

    # Run a subset
    python -m src.architecture_deepdive.experiment_runner --device cpu \\
        --probes p1_timeline,p2_language_modeling
"""

import argparse
import json
import os
import sys
import time
from collections.abc import Callable

import torch

from src.architecture_deepdive.probes import (
    p1_model_timeline,
    p2_language_modeling,
    p3_transfer_learning,
    p4_model_anatomy,
    p5_attention_viz,
    p6_arch_comparison,
)

# Registry mapping probe name -> module with run_experiment()
PROBE_REGISTRY: dict[str, Callable[..., dict]] = {
    "p1_timeline": p1_model_timeline.run_experiment,
    "p2_language_modeling": p2_language_modeling.run_experiment,
    "p3_transfer_learning": p3_transfer_learning.run_experiment,
    "p4_model_anatomy": p4_model_anatomy.run_experiment,
    "p5_attention_viz": p5_attention_viz.run_experiment,
    "p6_arch_comparison": p6_arch_comparison.run_experiment,
}

# p1_timeline does not accept a device argument
_NO_DEVICE_PROBES = {"p1_timeline"}


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


def parse_probes(probes_str: str) -> list[str]:
    """Parse a comma-separated probe list or the special value 'all'.

    Args:
        probes_str: "all" or comma-separated probe names.

    Returns:
        List of probe name strings.
    """
    if probes_str.strip().lower() == "all":
        return list(PROBE_REGISTRY.keys())
    return [p.strip() for p in probes_str.split(",") if p.strip()]


def run_probes(
    probes: list[str],
    device: str,
    output_path: str,
) -> tuple[dict, int]:
    """Run the requested architecture probes and persist results.

    Args:
        probes: List of probe names to run.
        device: Resolved device string ("cpu" or "cuda").
        output_path: Path to write the JSON results file.

    Returns:
        Tuple of (results dict, failure count).
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Running {len(probes)} probe(s) on device: {device}")
    print(f"Probes: {probes}\n")

    all_results: dict = {
        "metadata": {
            "experiment": "Transformer Architecture Deep Dive (Ch1.4)",
            "device": device,
            "torch_version": torch.__version__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "probes": {},
    }
    failures = 0

    for probe_name in probes:
        print(f"{'=' * 60}")
        print(f"Running: {probe_name}")
        print(f"{'=' * 60}")

        start = time.perf_counter()

        try:
            if probe_name in _NO_DEVICE_PROBES:
                result = PROBE_REGISTRY[probe_name]()
            else:
                result = PROBE_REGISTRY[probe_name](device=device)
            result["status"] = "success"
            print(f"  OK {probe_name} completed\n")
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL {probe_name} failed: {exc}\n", file=sys.stderr)
            result = {"status": "failed", "error": str(exc)}
            failures += 1

        result["duration_sec"] = round(time.perf_counter() - start, 2)
        all_results["probes"][probe_name] = result

    # Write JSON results
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    succeeded = sum(1 for r in all_results["probes"].values() if r["status"] == "success")
    print(f"Summary: {succeeded}/{len(all_results['probes'])} probes passed")
    print("Figures: results/figures/")

    return all_results, failures


def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run Transformer Architecture Deep Dive Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--device",
        default="auto",
        help=("Device to run on: 'cpu', 'cuda', 'cuda:N', or 'auto'" " (default: auto)"),
    )
    parser.add_argument(
        "--probes",
        default="all",
        help=(
            "Comma-separated probe names or 'all'." f" Valid probes: {', '.join(PROBE_REGISTRY)}"
        ),
    )
    parser.add_argument(
        "--output",
        default="results/outputs.json",
        help="Path to write JSON results (default: results/outputs.json)",
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
    probes = parse_probes(args.probes)

    # Validate probe names
    invalid = [p for p in probes if p not in PROBE_REGISTRY]
    if invalid:
        print(f"Error: Unknown probe(s): {invalid}", file=sys.stderr)
        print(f"Valid probes: {sorted(PROBE_REGISTRY)}", file=sys.stderr)
        return 1

    if not probes:
        print("Error: No probes specified.", file=sys.stderr)
        return 1

    _, failures = run_probes(
        probes=probes,
        device=device,
        output_path=args.output,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
