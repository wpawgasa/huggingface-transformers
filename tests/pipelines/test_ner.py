"""Tests for src/pipelines/ner.py."""

from unittest.mock import MagicMock, patch

from src.pipelines.ner import DEFAULT_MODEL, TASK, run_experiment


def _ner_grouped_output():
    return [
        {"entity_group": "PER", "word": "Sylvain", "score": 0.9985, "start": 11, "end": 18},
        {"entity_group": "ORG", "word": "Hugging Face", "score": 0.9811, "start": 33, "end": 45},
        {"entity_group": "LOC", "word": "Brooklyn", "score": 0.9975, "start": 49, "end": 57},
    ]


def _ner_ungrouped_output():
    return [
        {"entity": "B-PER", "word": "Syl", "score": 0.9985, "start": 11, "end": 14},
        {"entity": "I-PER", "word": "##vain", "score": 0.9975, "start": 14, "end": 18},
    ]


@patch("src.pipelines.ner.benchmark_pipeline")
@patch("src.pipelines.ner.hf_pipeline")
class TestRunExperiment:
    def _setup(self, mock_hf, mock_bench):
        mock_pipe = MagicMock()
        # Return grouped for first call, ungrouped for second (ablation)
        mock_pipe.side_effect = [
            _ner_grouped_output(),  # grouped main pipe
            _ner_grouped_output(),  # edge case 1
            _ner_grouped_output(),  # edge case 2
            _ner_grouped_output(),  # edge case 3
            _ner_ungrouped_output(),  # ablation ungrouped
        ] + [
            _ner_grouped_output()
        ] * 100  # catch-all
        mock_hf.return_value = mock_pipe
        mock_bench.return_value = MagicMock(to_dict=lambda: {"warm_latency_ms": 120.0})
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
        assert "grouped_entities" in ce
        assert ce["grouped_entities"] is True

    def test_has_edge_cases(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "edge_cases" in result

    def test_has_ablation_grouped_entities(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "ablation" in result
        assert "grouped_entities" in result["ablation"]

    def test_ablation_has_both_modes(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        ge = result["ablation"]["grouped_entities"]
        assert "grouped_true" in ge
        assert "grouped_false" in ge

    def test_has_benchmark(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "benchmark" in result
