"""Tests for src/pipelines/text_classification.py."""

from unittest.mock import MagicMock, patch

from src.pipeline_exploration.pipelines.text_classification import DEFAULT_MODEL, TASK, load_pipeline, run_experiment


def _make_mock_pipe(outputs):
    """Return a mock pipeline callable that yields ``outputs``."""
    mock = MagicMock()
    mock.return_value = outputs
    return mock


@patch("src.pipeline_exploration.pipelines.text_classification.benchmark_pipeline")
@patch("src.pipeline_exploration.pipelines.text_classification.hf_pipeline")
class TestRunExperiment:
    def _mock_outputs(self):
        return [{"label": "POSITIVE", "score": 0.9598}]

    def test_returns_dict(self, mock_hf, mock_bench):
        mock_pipe = _make_mock_pipe(self._mock_outputs())
        mock_hf.return_value = mock_pipe
        mock_bench.return_value = MagicMock(to_dict=lambda: {"cold_start_ms": 100.0})

        result = run_experiment(device="cpu")
        assert isinstance(result, dict)

    def test_has_required_top_level_keys(self, mock_hf, mock_bench):
        mock_pipe = _make_mock_pipe(self._mock_outputs())
        mock_hf.return_value = mock_pipe
        mock_bench.return_value = MagicMock(to_dict=lambda: {})

        result = run_experiment(device="cpu")
        assert result["task"] == TASK
        assert result["model"] == DEFAULT_MODEL
        assert result["device"] == "cpu"

    def test_course_examples_key_present(self, mock_hf, mock_bench):
        mock_pipe = _make_mock_pipe(self._mock_outputs())
        mock_hf.return_value = mock_pipe
        mock_bench.return_value = MagicMock(to_dict=lambda: {})

        result = run_experiment(device="cpu")
        assert "course_examples" in result
        assert "inputs" in result["course_examples"]
        assert "outputs" in result["course_examples"]

    def test_edge_cases_key_present(self, mock_hf, mock_bench):
        mock_pipe = _make_mock_pipe(self._mock_outputs())
        mock_hf.return_value = mock_pipe
        mock_bench.return_value = MagicMock(to_dict=lambda: {})

        result = run_experiment(device="cpu")
        assert "edge_cases" in result

    def test_benchmark_key_present(self, mock_hf, mock_bench):
        mock_pipe = _make_mock_pipe(self._mock_outputs())
        mock_hf.return_value = mock_pipe
        mock_bench.return_value = MagicMock(to_dict=lambda: {"warm_latency_ms": 55.0})

        result = run_experiment(device="cpu")
        assert "benchmark" in result

    def test_hf_pipeline_called_with_correct_task(self, mock_hf, mock_bench):
        mock_pipe = _make_mock_pipe(self._mock_outputs())
        mock_hf.return_value = mock_pipe
        mock_bench.return_value = MagicMock(to_dict=lambda: {})

        run_experiment(device="cpu")
        call_args = mock_hf.call_args
        assert (
            call_args[0][0] == TASK
            or call_args.kwargs.get("task") == TASK
            or TASK in str(call_args)
        )

    def test_device_passed_to_pipeline(self, mock_hf, mock_bench):
        mock_pipe = _make_mock_pipe(self._mock_outputs())
        mock_hf.return_value = mock_pipe
        mock_bench.return_value = MagicMock(to_dict=lambda: {})

        run_experiment(device="cuda")
        # hf_pipeline should have been called with device="cuda"
        _, kwargs = mock_hf.call_args
        assert kwargs.get("device") == "cuda"


@patch("src.pipeline_exploration.pipelines.text_classification.hf_pipeline")
class TestLoadPipeline:
    def test_returns_pipeline(self, mock_hf):
        mock_pipe = MagicMock()
        mock_hf.return_value = mock_pipe
        result = load_pipeline()
        assert result is mock_pipe

    def test_default_model(self, mock_hf):
        mock_hf.return_value = MagicMock()
        load_pipeline()
        _, kwargs = mock_hf.call_args
        assert kwargs.get("model") == DEFAULT_MODEL

    def test_custom_model(self, mock_hf):
        mock_hf.return_value = MagicMock()
        load_pipeline(model="custom/model")
        _, kwargs = mock_hf.call_args
        assert kwargs.get("model") == "custom/model"
