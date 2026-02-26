"""Latency and throughput measurement utilities for pipeline benchmarking.

Measures three metrics:
- cold_start_ms: time to instantiate the pipeline (load model into memory)
- warm_latency_ms: mean latency per call after warm-up
- throughput_samples_per_sec: samples processed per second at warm latency
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class BenchmarkResult:
    """Result of a pipeline benchmark run."""

    task: str
    model: str
    cold_start_ms: float
    warm_latency_ms: float
    throughput_samples_per_sec: float
    num_warm_runs: int

    def __str__(self) -> str:
        return (
            f"BenchmarkResult(task={self.task!r}, model={self.model!r}, "
            f"cold_start={self.cold_start_ms:.1f}ms, "
            f"warm_latency={self.warm_latency_ms:.1f}ms, "
            f"throughput={self.throughput_samples_per_sec:.2f} samples/s)"
        )

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "model": self.model,
            "cold_start_ms": round(self.cold_start_ms, 2),
            "warm_latency_ms": round(self.warm_latency_ms, 2),
            "throughput_samples_per_sec": round(self.throughput_samples_per_sec, 4),
            "num_warm_runs": self.num_warm_runs,
        }


def benchmark_pipeline(
    pipe_factory: Callable[[], Any],
    inputs: Any,
    task_name: str,
    model_name: str,
    n_warm_runs: int = 5,
) -> BenchmarkResult:
    """Benchmark a HuggingFace pipeline.

    Args:
        pipe_factory: Callable that returns a new pipeline instance.
                      Used to measure cold-start time (model loading from disk).
        inputs: Input(s) to pass to the pipeline. Can be a string, dict, or list.
        task_name: Name of the pipeline task (e.g., "text-classification").
        model_name: Model identifier (e.g., "distilbert-base-uncased-finetuned-sst-2-english").
        n_warm_runs: Number of warm inference runs to average over.

    Returns:
        BenchmarkResult with cold-start, warm-latency, and throughput measurements.
    """
    # --- Cold start: measure time to load the pipeline from disk ---
    t0 = time.perf_counter()
    pipe = pipe_factory()
    t1 = time.perf_counter()
    cold_start_ms = (t1 - t0) * 1000.0

    # --- Warm-up: one unpublished run to fill caches ---
    pipe(inputs)

    # --- Timed warm runs ---
    latencies_ms: list[float] = []
    for _ in range(n_warm_runs):
        t0 = time.perf_counter()
        pipe(inputs)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    warm_latency_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0

    # Throughput: how many samples processed per second
    n_inputs = len(inputs) if isinstance(inputs, (list, tuple)) else 1
    throughput = n_inputs * 1000.0 / warm_latency_ms if warm_latency_ms > 0 else 0.0

    return BenchmarkResult(
        task=task_name,
        model=model_name,
        cold_start_ms=cold_start_ms,
        warm_latency_ms=warm_latency_ms,
        throughput_samples_per_sec=throughput,
        num_warm_runs=n_warm_runs,
    )
