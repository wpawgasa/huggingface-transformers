"""Tests for src/pipelines/image_classification.py."""

from unittest.mock import MagicMock, patch

from src.pipeline_exploration.pipelines.image_classification import DEFAULT_MODEL, TASK, run_experiment


def _img_cls_output():
    return [
        {"label": "Egyptian cat", "score": 0.9879},
        {"label": "tabby, tabby cat", "score": 0.0049},
    ]


@patch("src.pipeline_exploration.pipelines.image_classification.benchmark_pipeline")
@patch("src.pipeline_exploration.pipelines.image_classification.hf_pipeline")
class TestRunExperiment:
    def _setup(self, mock_hf, mock_bench):
        mock_pipe = MagicMock()
        mock_pipe.return_value = _img_cls_output()
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
        ce = result["course_examples"]
        assert "input" in ce
        assert "outputs" in ce
        assert "top_prediction" in ce

    def test_top_prediction_is_dict(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert isinstance(result["course_examples"]["top_prediction"], dict)

    def test_has_edge_cases(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "edge_cases" in result

    def test_has_benchmark(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "benchmark" in result
