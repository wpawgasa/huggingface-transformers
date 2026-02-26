"""Tests for src/architecture_deepdive/experiment_runner.py."""

import json
import os
from unittest.mock import MagicMock, patch

from src.architecture_deepdive.experiment_runner import (
    PROBE_REGISTRY,
    build_parser,
    main,
    parse_probes,
    resolve_device,
    run_probes,
)


class TestResolveDevice:
    def test_cpu_passthrough(self):
        assert resolve_device("cpu") == "cpu"

    def test_cuda_passthrough(self):
        assert resolve_device("cuda") == "cuda"

    def test_cuda_n_passthrough(self):
        assert resolve_device("cuda:1") == "cuda:1"

    @patch("src.architecture_deepdive.experiment_runner.torch")
    def test_auto_resolves_to_cuda_when_available(self, mock_torch):
        mock_torch.cuda.is_available.return_value = True
        assert resolve_device("auto") == "cuda"

    @patch("src.architecture_deepdive.experiment_runner.torch")
    def test_auto_resolves_to_cpu_when_no_gpu(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        assert resolve_device("auto") == "cpu"


class TestParseProbes:
    def test_all_returns_all_probes(self):
        probes = parse_probes("all")
        assert set(probes) == set(PROBE_REGISTRY.keys())

    def test_all_case_insensitive(self):
        probes = parse_probes("ALL")
        assert set(probes) == set(PROBE_REGISTRY.keys())

    def test_single_probe(self):
        probes = parse_probes("p1_timeline")
        assert probes == ["p1_timeline"]

    def test_multiple_probes(self):
        probes = parse_probes("p1_timeline,p2_language_modeling")
        assert probes == ["p1_timeline", "p2_language_modeling"]

    def test_strips_whitespace(self):
        probes = parse_probes(" p1_timeline , p2_language_modeling ")
        assert probes == ["p1_timeline", "p2_language_modeling"]

    def test_empty_string_returns_empty(self):
        probes = parse_probes("")
        assert probes == []

    def test_all_probe_names_are_valid(self):
        probes = parse_probes("all")
        for p in probes:
            assert p in PROBE_REGISTRY


class TestBuildParser:
    def test_default_device(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.device == "auto"

    def test_default_probes(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.probes == "all"

    def test_default_output(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.output == "results/outputs.json"

    def test_custom_device(self):
        parser = build_parser()
        args = parser.parse_args(["--device", "cuda"])
        assert args.device == "cuda"

    def test_custom_probes(self):
        parser = build_parser()
        args = parser.parse_args(["--probes", "p1_timeline,p4_model_anatomy"])
        assert args.probes == "p1_timeline,p4_model_anatomy"

    def test_custom_output(self):
        parser = build_parser()
        args = parser.parse_args(["--output", "/tmp/out.json"])
        assert args.output == "/tmp/out.json"


class TestRunProbes:
    def _mock_run_experiment(self, task_name="test_probe"):
        return MagicMock(return_value={"task": task_name, "result": "ok"})

    def test_writes_json_output(self, tmp_path):
        output_path = str(tmp_path / "results.json")
        mock_fn = self._mock_run_experiment()

        with patch.dict(PROBE_REGISTRY, {"p1_timeline": mock_fn}, clear=True):
            run_probes(
                probes=["p1_timeline"],
                device="cpu",
                output_path=output_path,
            )

        assert os.path.exists(output_path)
        with open(output_path) as f:
            data = json.load(f)
        assert "probes" in data
        assert "p1_timeline" in data["probes"]

    def test_returns_results_and_failures(self, tmp_path):
        output_path = str(tmp_path / "results.json")
        mock_fn = self._mock_run_experiment()

        with patch.dict(PROBE_REGISTRY, {"p1_timeline": mock_fn}, clear=True):
            results, failures = run_probes(
                probes=["p1_timeline"],
                device="cpu",
                output_path=output_path,
            )

        assert isinstance(results, dict)
        assert failures == 0

    def test_catches_exceptions(self, tmp_path):
        output_path = str(tmp_path / "results.json")
        mock_fn = MagicMock(side_effect=RuntimeError("boom"))

        with patch.dict(PROBE_REGISTRY, {"p1_timeline": mock_fn}, clear=True):
            results, failures = run_probes(
                probes=["p1_timeline"],
                device="cpu",
                output_path=output_path,
            )

        assert failures == 1
        assert results["probes"]["p1_timeline"]["status"] == "failed"

    def test_adds_status_and_duration(self, tmp_path):
        output_path = str(tmp_path / "results.json")
        mock_fn = self._mock_run_experiment()

        with patch.dict(PROBE_REGISTRY, {"p1_timeline": mock_fn}, clear=True):
            results, _ = run_probes(
                probes=["p1_timeline"],
                device="cpu",
                output_path=output_path,
            )

        probe_result = results["probes"]["p1_timeline"]
        assert probe_result["status"] == "success"
        assert "duration_sec" in probe_result

    def test_has_metadata(self, tmp_path):
        output_path = str(tmp_path / "results.json")
        mock_fn = self._mock_run_experiment()

        with patch.dict(PROBE_REGISTRY, {"p1_timeline": mock_fn}, clear=True):
            results, _ = run_probes(
                probes=["p1_timeline"],
                device="cpu",
                output_path=output_path,
            )

        assert "metadata" in results
        assert results["metadata"]["device"] == "cpu"
        assert "torch_version" in results["metadata"]
        assert "timestamp" in results["metadata"]

    def test_multiple_probes(self, tmp_path):
        output_path = str(tmp_path / "results.json")
        mock_fn1 = self._mock_run_experiment("probe1")
        mock_fn2 = self._mock_run_experiment("probe2")

        with patch.dict(
            PROBE_REGISTRY,
            {"p1_timeline": mock_fn1, "p2_language_modeling": mock_fn2},
            clear=True,
        ):
            results, failures = run_probes(
                probes=["p1_timeline", "p2_language_modeling"],
                device="cpu",
                output_path=output_path,
            )

        assert failures == 0
        assert "p1_timeline" in results["probes"]
        assert "p2_language_modeling" in results["probes"]


class TestMain:
    @patch("src.architecture_deepdive.experiment_runner.run_probes")
    @patch("src.architecture_deepdive.experiment_runner.torch")
    def test_returns_zero_on_success(self, mock_torch, mock_run):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.__version__ = "2.0.0"
        mock_run.return_value = ({}, 0)
        exit_code = main(["--probes", "p1_timeline", "--device", "cpu"])
        assert exit_code == 0

    @patch("src.architecture_deepdive.experiment_runner.run_probes")
    @patch("src.architecture_deepdive.experiment_runner.torch")
    def test_returns_one_on_failure(self, mock_torch, mock_run):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.__version__ = "2.0.0"
        mock_run.return_value = ({}, 1)
        exit_code = main(["--probes", "p1_timeline", "--device", "cpu"])
        assert exit_code == 1

    @patch("src.architecture_deepdive.experiment_runner.torch")
    def test_invalid_probe_returns_one(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        exit_code = main(["--probes", "nonexistent_probe", "--device", "cpu"])
        assert exit_code == 1

    @patch("src.architecture_deepdive.experiment_runner.torch")
    def test_empty_probes_returns_one(self, mock_torch):
        mock_torch.cuda.is_available.return_value = False
        exit_code = main(["--probes", "", "--device", "cpu"])
        assert exit_code == 1
