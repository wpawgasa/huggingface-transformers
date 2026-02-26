"""Tests for src/pipelines/text_generation.py."""

from unittest.mock import MagicMock, patch

from src.pipeline_exploration.pipelines.text_generation import DEFAULT_MODEL, TASK, run_experiment


def _gen_output(text="In this course, we will teach you how to build a model."):
    return [{"generated_text": text}]


@patch("src.pipeline_exploration.pipelines.text_generation.benchmark_pipeline")
@patch("src.pipeline_exploration.pipelines.text_generation.hf_pipeline")
class TestRunExperiment:
    def _setup(self, mock_hf, mock_bench):
        mock_pipe = MagicMock()
        mock_pipe.return_value = _gen_output()
        mock_hf.return_value = mock_pipe
        mock_bench.return_value = MagicMock(to_dict=lambda: {"warm_latency_ms": 150.0})
        return mock_pipe

    def test_returns_dict(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert isinstance(result, dict)

    def test_task_and_model(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert result["task"] == TASK
        assert result["model"] == DEFAULT_MODEL

    def test_has_course_examples(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "course_examples" in result

    def test_has_edge_cases(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "edge_cases" in result

    def test_has_ablation_temperature_sweep(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "ablation" in result
        assert "temperature_sweep" in result["ablation"]

    def test_has_ablation_model_comparison(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "model_comparison" in result["ablation"]

    def test_has_benchmark(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "benchmark" in result

    def test_temperature_sweep_has_three_entries(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        sweep = result["ablation"]["temperature_sweep"]
        assert "temperature_0.7" in sweep
        assert "temperature_1.0" in sweep
        assert "temperature_1.5" in sweep
