"""Tests for src/pipelines/speech_recognition.py."""

from unittest.mock import MagicMock, patch

from src.pipeline_exploration.pipelines.speech_recognition import (
    CPU_MODEL,
    GPU_MODEL,
    TASK,
    _select_model,
    load_pipeline,
    run_experiment,
)


def _asr_output():
    return {"text": "My fellow Americans, I have a dream."}


class TestSelectModel:
    def test_cpu_returns_tiny(self):
        assert _select_model("cpu") == CPU_MODEL

    def test_cuda_returns_large(self):
        assert _select_model("cuda") == GPU_MODEL

    def test_cuda_with_index_returns_large(self):
        assert _select_model("cuda:0") == GPU_MODEL

    def test_cuda_1_returns_large(self):
        assert _select_model("cuda:1") == GPU_MODEL


@patch("src.pipeline_exploration.pipelines.speech_recognition.hf_pipeline")
class TestLoadPipeline:
    def test_default_uses_cpu_model(self, mock_hf):
        mock_hf.return_value = MagicMock()
        load_pipeline(device="cpu")
        _, kwargs = mock_hf.call_args
        assert kwargs.get("model") == CPU_MODEL

    def test_none_model_selects_based_on_device(self, mock_hf):
        """load_pipeline(model=None, device='cpu') should call _select_model."""
        mock_hf.return_value = MagicMock()
        load_pipeline(model=None, device="cpu")
        _, kwargs = mock_hf.call_args
        assert kwargs.get("model") == CPU_MODEL


@patch("src.pipeline_exploration.pipelines.speech_recognition.benchmark_pipeline")
@patch("src.pipeline_exploration.pipelines.speech_recognition.hf_pipeline")
class TestRunExperiment:
    def _setup(self, mock_hf, mock_bench, device="cpu"):
        mock_pipe = MagicMock()
        mock_pipe.return_value = _asr_output()
        mock_hf.return_value = mock_pipe
        mock_bench.return_value = MagicMock(to_dict=lambda: {"warm_latency_ms": 2000.0})
        return mock_pipe

    def test_returns_dict(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert isinstance(result, dict)

    def test_task_set(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert result["task"] == TASK

    def test_cpu_uses_tiny_model(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert result["model"] == CPU_MODEL

    def test_cuda_uses_large_model(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench, device="cuda")
        result = run_experiment(device="cuda")
        assert result["model"] == GPU_MODEL

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

    def test_has_ablation(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "ablation" in result
        assert "model_substitution" in result["ablation"]

    def test_ablation_has_cpu_and_gpu_models(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        sub = result["ablation"]["model_substitution"]
        assert sub["cpu_model"] == CPU_MODEL
        assert sub["gpu_model"] == GPU_MODEL

    def test_has_benchmark(self, mock_hf, mock_bench):
        self._setup(mock_hf, mock_bench)
        result = run_experiment(device="cpu")
        assert "benchmark" in result
