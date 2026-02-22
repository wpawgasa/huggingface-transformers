"""Tests for src/pipelines/zero_shot.py."""

from unittest.mock import MagicMock, patch

from src.pipelines.zero_shot import DEFAULT_MODEL, TASK, run_experiment


def _make_zero_shot_output(seq="test", labels=None, scores=None):
    labels = labels or ["education", "business", "politics"]
    scores = scores or [0.6, 0.3, 0.1]
    return {"sequence": seq, "labels": labels, "scores": scores}


@patch("src.pipelines.zero_shot.benchmark_pipeline")
@patch("src.pipelines.zero_shot.hf_pipeline")
class TestRunExperiment:
    def _setup_mocks(self, mock_hf, mock_bench, n_labels=3):
        labels = [
            "education",
            "business",
            "politics",
            "technology",
            "science",
            "sports",
            "entertainment",
            "health",
            "environment",
            "culture",
        ][:n_labels]
        scores = [1.0 / n_labels] * n_labels

        mock_pipe = MagicMock()
        mock_pipe.return_value = _make_zero_shot_output(labels=labels, scores=scores)
        mock_hf.return_value = mock_pipe
        mock_bench.return_value = MagicMock(to_dict=lambda: {"warm_latency_ms": 200.0})
        return mock_pipe

    def test_returns_dict(self, mock_hf, mock_bench):
        self._setup_mocks(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert isinstance(result, dict)

    def test_task_and_model_set(self, mock_hf, mock_bench):
        self._setup_mocks(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert result["task"] == TASK
        assert result["model"] == DEFAULT_MODEL

    def test_has_course_examples(self, mock_hf, mock_bench):
        self._setup_mocks(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "course_examples" in result

    def test_has_edge_cases(self, mock_hf, mock_bench):
        self._setup_mocks(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "edge_cases" in result

    def test_has_ablation(self, mock_hf, mock_bench):
        self._setup_mocks(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "ablation" in result
        assert "label_count_scaling" in result["ablation"]

    def test_has_benchmark(self, mock_hf, mock_bench):
        self._setup_mocks(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "benchmark" in result

    def test_ablation_has_n_labels_entries(self, mock_hf, mock_bench):
        self._setup_mocks(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        scaling = result["ablation"]["label_count_scaling"]
        # Should have entries for n_labels ∈ {2, 3, 5, 10}
        assert "n_labels_2" in scaling
        assert "n_labels_3" in scaling
        assert "n_labels_5" in scaling
        assert "n_labels_10" in scaling
