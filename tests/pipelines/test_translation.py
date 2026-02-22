"""Tests for src/pipelines/translation.py."""

from unittest.mock import MagicMock, patch

from src.pipelines.translation import DEFAULT_MODEL, TASK, run_experiment


def _trans_output():
    return [{"translation_text": "This course is produced by Hugging Face."}]


@patch("src.pipelines.translation.benchmark_pipeline")
@patch("src.pipelines.translation.hf_pipeline")
class TestRunExperiment:
    def _setup(self, mock_hf, mock_bench):
        mock_pipe = MagicMock()
        mock_pipe.return_value = _trans_output()
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
        assert "inputs" in result["course_examples"]
        assert "outputs" in result["course_examples"]

    def test_has_edge_cases(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "edge_cases" in result

    def test_has_benchmark(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "benchmark" in result

    def test_model_is_french_to_english(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        assert "fr-en" in DEFAULT_MODEL or "fr_en" in DEFAULT_MODEL or "fr" in DEFAULT_MODEL
