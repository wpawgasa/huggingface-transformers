"""Tests for src/pipelines/fill_mask.py."""

from unittest.mock import MagicMock, patch

from src.pipeline_exploration.pipelines.fill_mask import BERT_MODEL, DEFAULT_MODEL, TASK, run_experiment


def _fill_output():
    return [
        {
            "sequence": "This course will teach you all about language models.",
            "score": 0.85,
            "token": 2235,
            "token_str": " language",
        },
        {
            "sequence": "This course will teach you all about deep models.",
            "score": 0.10,
            "token": 2784,
            "token_str": " deep",
        },
    ]


@patch("src.pipeline_exploration.pipelines.fill_mask.benchmark_pipeline")
@patch("src.pipeline_exploration.pipelines.fill_mask.hf_pipeline")
class TestRunExperiment:
    def _setup(self, mock_hf, mock_bench):
        mock_pipe = MagicMock()
        mock_pipe.return_value = _fill_output()
        mock_hf.return_value = mock_pipe
        mock_bench.return_value = MagicMock(to_dict=lambda: {"warm_latency_ms": 60.0})
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
        assert "model" in ce
        assert "input" in ce
        assert "outputs" in ce

    def test_has_edge_cases(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "edge_cases" in result

    def test_has_ablation_mask_model_swap(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "ablation" in result
        assert "mask_token_model_swap" in result["ablation"]

    def test_ablation_has_both_models(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        swap = result["ablation"]["mask_token_model_swap"]
        assert "distilroberta" in swap
        assert "bert" in swap
        assert swap["distilroberta"]["model"] == DEFAULT_MODEL
        assert swap["bert"]["model"] == BERT_MODEL

    def test_has_benchmark(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "benchmark" in result
