"""Tests for src/pipelines/question_answering.py."""

from unittest.mock import MagicMock, patch

from src.pipeline_exploration.pipelines.question_answering import DEFAULT_MODEL, TASK, run_experiment


def _qa_output():
    return {"answer": "Hugging Face", "score": 0.9714, "start": 35, "end": 47}


@patch("src.pipeline_exploration.pipelines.question_answering.benchmark_pipeline")
@patch("src.pipeline_exploration.pipelines.question_answering.hf_pipeline")
class TestRunExperiment:
    def _setup(self, mock_hf, mock_bench):
        mock_pipe = MagicMock()
        mock_pipe.return_value = _qa_output()
        mock_hf.return_value = mock_pipe
        mock_bench.return_value = MagicMock(to_dict=lambda: {"warm_latency_ms": 75.0})
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
        assert "output" in ce

    def test_has_edge_cases(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "edge_cases" in result

    def test_has_benchmark(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "benchmark" in result

    def test_pipe_called_with_question_and_context(self, mock_hf, mock_bench):
        mock_pipe = self._setup(mock_hf, mock_bench)
        run_experiment(device="cpu")
        # The pipeline should be called with question= and context= kwargs
        first_call = mock_pipe.call_args_list[0]
        kwargs = first_call.kwargs
        assert "question" in kwargs
        assert "context" in kwargs
