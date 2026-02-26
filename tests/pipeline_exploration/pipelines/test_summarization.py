"""Tests for src/pipelines/summarization.py."""

from unittest.mock import MagicMock, patch

from src.pipeline_exploration.pipelines.summarization import DEFAULT_MODEL, TASK, run_experiment


def _sum_output():
    return [{"summary_text": "America has changed dramatically in recent years."}]


@patch("src.pipeline_exploration.pipelines.summarization.benchmark_pipeline")
@patch("src.pipeline_exploration.pipelines.summarization.hf_pipeline")
class TestRunExperiment:
    def _setup(self, mock_hf, mock_bench):
        mock_pipe = MagicMock()
        mock_pipe.return_value = _sum_output()
        mock_hf.return_value = mock_pipe
        mock_bench.return_value = MagicMock(to_dict=lambda: {"warm_latency_ms": 800.0})
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
        assert "inputs" in result["course_examples"]
        assert "outputs" in result["course_examples"]

    def test_has_edge_cases(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "edge_cases" in result

    def test_has_ablation_summary_length(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "ablation" in result
        assert "summary_length" in result["ablation"]

    def test_ablation_has_three_lengths(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        lengths = result["ablation"]["summary_length"]
        assert "max_length_50" in lengths
        assert "max_length_100" in lengths
        assert "max_length_200" in lengths

    def test_has_benchmark(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "benchmark" in result
