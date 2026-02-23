"""Tests for src/experiment_runner.py."""

import json
import os
from unittest.mock import MagicMock, patch

from src.experiment_runner import (
    TASK_REGISTRY,
    build_parser,
    main,
    parse_tasks,
    resolve_device,
    run_tasks,
)


class TestResolveDevice:
    def test_cpu_passthrough(self):
        assert resolve_device("cpu") == "cpu"

    def test_cuda_passthrough(self):
        assert resolve_device("cuda") == "cuda"

    def test_cuda_n_passthrough(self):
        assert resolve_device("cuda:1") == "cuda:1"

    @patch("src.experiment_runner.torch")
    def test_auto_resolves_to_cuda_when_available(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        assert resolve_device("auto") == "cuda"

    @patch("src.experiment_runner.torch")
    def test_auto_resolves_to_cpu_when_no_gpu(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        assert resolve_device("auto") == "cpu"


class TestParseTasks:
    def test_all_returns_all_tasks(self):
        tasks = parse_tasks("all")
        assert set(tasks) == set(TASK_REGISTRY.keys())

    def test_all_case_insensitive(self):
        tasks = parse_tasks("ALL")
        assert set(tasks) == set(TASK_REGISTRY.keys())

    def test_single_task(self):
        tasks = parse_tasks("text-classification")
        assert tasks == ["text-classification"]

    def test_multiple_tasks(self):
        tasks = parse_tasks("text-classification,fill-mask,ner")
        assert tasks == ["text-classification", "fill-mask", "ner"]

    def test_strips_whitespace(self):
        tasks = parse_tasks(" text-classification , fill-mask ")
        assert tasks == ["text-classification", "fill-mask"]

    def test_empty_string_returns_empty(self):
        tasks = parse_tasks("")
        assert tasks == []

    def test_all_task_names_are_valid(self):
        tasks = parse_tasks("all")
        for t in tasks:
            assert t in TASK_REGISTRY


class TestBuildParser:
    def test_default_device(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.device == "auto"

    def test_default_tasks(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.tasks == "all"

    def test_default_output(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.output == "results/outputs.json"

    def test_custom_device(self):
        parser = build_parser()
        args = parser.parse_args(["--device", "cpu"])
        assert args.device == "cpu"

    def test_custom_tasks(self):
        parser = build_parser()
        args = parser.parse_args(["--tasks", "text-classification,fill-mask"])
        assert args.tasks == "text-classification,fill-mask"

    def test_custom_output(self):
        parser = build_parser()
        args = parser.parse_args(["--output", "/tmp/results.json"])
        assert args.output == "/tmp/results.json"


class TestRunTasks:
    def _mock_run_experiment(self, return_value=None):
        """Return a mock run_experiment that returns a predetermined dict."""
        result = return_value or {
            "task": "text-classification",
            "model": "distilbert",
            "device": "cpu",
            "benchmark": {
                "task": "text-classification",
                "model": "distilbert",
                "cold_start_ms": 100.0,
                "warm_latency_ms": 50.0,
                "throughput_samples_per_sec": 20.0,
                "num_warm_runs": 5,
            },
        }
        return MagicMock(return_value=result)

    def test_writes_json_output(self, tmp_path):
        output_path = str(tmp_path / "results.json")
        benchmark_path = str(tmp_path / "benchmarks.csv")

        mock_fn = self._mock_run_experiment()

        with patch.dict(TASK_REGISTRY, {"text-classification": mock_fn}):
            run_tasks(
                tasks=["text-classification"],
                device="cpu",
                output_path=output_path,
                benchmark_path=benchmark_path,
            )

        assert os.path.exists(output_path)
        with open(output_path) as f:
            data = json.load(f)
        assert "text-classification" in data

    def test_writes_benchmark_csv(self, tmp_path):
        output_path = str(tmp_path / "results.json")
        benchmark_path = str(tmp_path / "benchmarks.csv")

        mock_fn = self._mock_run_experiment()

        with patch.dict(TASK_REGISTRY, {"text-classification": mock_fn}):
            run_tasks(
                tasks=["text-classification"],
                device="cpu",
                output_path=output_path,
                benchmark_path=benchmark_path,
            )

        assert os.path.exists(benchmark_path)

    def test_failed_task_recorded_as_error(self, tmp_path):
        output_path = str(tmp_path / "results.json")
        benchmark_path = str(tmp_path / "benchmarks.csv")

        def failing_fn(**kwargs):
            raise RuntimeError("Model download failed")

        with patch.dict(TASK_REGISTRY, {"text-classification": failing_fn}):
            result, failures = run_tasks(
                tasks=["text-classification"],
                device="cpu",
                output_path=output_path,
                benchmark_path=benchmark_path,
            )

        assert "error" in result["text-classification"]
        assert failures == 1

    def test_creates_output_directory(self, tmp_path):
        output_path = str(tmp_path / "subdir" / "results.json")
        benchmark_path = str(tmp_path / "benchmarks.csv")

        mock_fn = self._mock_run_experiment()

        with patch.dict(TASK_REGISTRY, {"text-classification": mock_fn}):
            run_tasks(
                tasks=["text-classification"],
                device="cpu",
                output_path=output_path,
                benchmark_path=benchmark_path,
            )

        assert os.path.exists(output_path)

    def test_task_called_with_device(self, tmp_path):
        output_path = str(tmp_path / "results.json")
        benchmark_path = str(tmp_path / "benchmarks.csv")

        mock_fn = self._mock_run_experiment()

        with patch.dict(TASK_REGISTRY, {"text-classification": mock_fn}):
            run_tasks(
                tasks=["text-classification"],
                device="cuda",
                output_path=output_path,
                benchmark_path=benchmark_path,
            )

        mock_fn.assert_called_once_with(device="cuda")

    def test_no_benchmark_csv_when_no_benchmark(self, tmp_path):
        output_path = str(tmp_path / "results.json")
        benchmark_path = str(tmp_path / "benchmarks.csv")

        # Return without benchmark key
        mock_fn = MagicMock(return_value={"task": "t", "model": "m", "device": "cpu"})

        with patch.dict(TASK_REGISTRY, {"text-classification": mock_fn}):
            run_tasks(
                tasks=["text-classification"],
                device="cpu",
                output_path=output_path,
                benchmark_path=benchmark_path,
            )

        # No benchmark data, so CSV should not be written
        assert not os.path.exists(benchmark_path)


class TestMain:
    def test_unknown_task_returns_exit_code_1(self):
        exit_code = main(["--tasks", "nonexistent-task", "--output", "/tmp/out.json"])
        assert exit_code == 1

    def test_empty_tasks_returns_exit_code_1(self):
        exit_code = main(["--tasks", "", "--output", "/tmp/out.json"])
        assert exit_code == 1

    def test_valid_run_returns_zero(self, tmp_path):
        output_path = str(tmp_path / "results.json")
        benchmark_path = str(tmp_path / "benchmarks.csv")

        mock_fn = MagicMock(
            return_value={
                "task": "text-classification",
                "model": "distilbert",
                "device": "cpu",
                "benchmark": {
                    "cold_start_ms": 100.0,
                    "warm_latency_ms": 50.0,
                    "throughput_samples_per_sec": 20.0,
                    "num_warm_runs": 5,
                },
            }
        )

        with (
            patch.dict(TASK_REGISTRY, {"text-classification": mock_fn}),
            patch("src.experiment_runner.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            exit_code = main(
                [
                    "--tasks",
                    "text-classification",
                    "--output",
                    output_path,
                    "--benchmark-output",
                    benchmark_path,
                ]
            )

        assert exit_code == 0

    def test_failed_task_returns_exit_code_1(self, tmp_path):
        output_path = str(tmp_path / "results.json")
        benchmark_path = str(tmp_path / "benchmarks.csv")

        def failing_fn(**kwargs):
            raise RuntimeError("boom")

        with (
            patch.dict(TASK_REGISTRY, {"text-classification": failing_fn}),
            patch("src.experiment_runner.torch") as mock_torch,
        ):
            mock_torch.cuda.is_available.return_value = False
            exit_code = main(
                [
                    "--tasks",
                    "text-classification",
                    "--output",
                    output_path,
                    "--benchmark-output",
                    benchmark_path,
                ]
            )

        assert exit_code == 1

    def test_task_registry_has_expected_tasks(self):
        expected = {
            "text-classification",
            "zero-shot",
            "text-generation",
            "fill-mask",
            "ner",
            "question-answering",
            "summarization",
            "translation",
            "image-classification",
            "speech-recognition",
        }
        assert set(TASK_REGISTRY.keys()) == expected
