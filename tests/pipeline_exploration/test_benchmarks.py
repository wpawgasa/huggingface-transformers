"""Tests for src/benchmarks.py."""

from unittest.mock import MagicMock, call

from src.pipeline_exploration.benchmarks import BenchmarkResult, benchmark_pipeline


class TestBenchmarkResult:
    def test_basic_creation(self):
        result = BenchmarkResult(
            task="text-classification",
            model="distilbert",
            cold_start_ms=150.0,
            warm_latency_ms=55.0,
            throughput_samples_per_sec=18.18,
            num_warm_runs=5,
        )
        assert result.task == "text-classification"
        assert result.model == "distilbert"
        assert result.cold_start_ms == 150.0
        assert result.warm_latency_ms == 55.0
        assert result.throughput_samples_per_sec == 18.18
        assert result.num_warm_runs == 5

    def test_str_representation(self):
        result = BenchmarkResult(
            task="fill-mask",
            model="distilroberta",
            cold_start_ms=200.0,
            warm_latency_ms=80.0,
            throughput_samples_per_sec=12.5,
            num_warm_runs=3,
        )
        s = str(result)
        assert "fill-mask" in s
        assert "distilroberta" in s
        assert "80.0" in s

    def test_to_dict(self):
        result = BenchmarkResult(
            task="ner",
            model="bert-large",
            cold_start_ms=300.123456,
            warm_latency_ms=100.987654,
            throughput_samples_per_sec=9.90,
            num_warm_runs=5,
        )
        d = result.to_dict()
        assert d["task"] == "ner"
        assert d["model"] == "bert-large"
        assert isinstance(d["cold_start_ms"], float)
        assert isinstance(d["warm_latency_ms"], float)
        assert isinstance(d["throughput_samples_per_sec"], float)
        assert d["num_warm_runs"] == 5
        # Check rounding
        assert d["cold_start_ms"] == round(300.123456, 2)
        assert d["warm_latency_ms"] == round(100.987654, 2)

    def test_to_dict_has_all_keys(self):
        result = BenchmarkResult("t", "m", 1.0, 2.0, 3.0, 5)
        keys = set(result.to_dict().keys())
        assert keys == {
            "task",
            "model",
            "cold_start_ms",
            "warm_latency_ms",
            "throughput_samples_per_sec",
            "num_warm_runs",
        }


class TestBenchmarkPipeline:
    def _make_mock_factory(self, return_value):
        """Create a mock pipe factory that returns a callable mock pipeline."""
        mock_pipe = MagicMock()
        mock_pipe.return_value = return_value

        def factory():
            return mock_pipe

        return factory, mock_pipe

    def test_returns_benchmark_result(self):
        factory, _ = self._make_mock_factory([{"label": "POSITIVE", "score": 0.99}])
        result = benchmark_pipeline(
            pipe_factory=factory,
            inputs="test input",
            task_name="text-classification",
            model_name="test-model",
            n_warm_runs=3,
        )
        assert isinstance(result, BenchmarkResult)

    def test_task_and_model_preserved(self):
        factory, _ = self._make_mock_factory("some output")
        result = benchmark_pipeline(
            pipe_factory=factory,
            inputs="hello",
            task_name="my-task",
            model_name="my-model",
            n_warm_runs=2,
        )
        assert result.task == "my-task"
        assert result.model == "my-model"
        assert result.num_warm_runs == 2

    def test_cold_start_ms_is_nonnegative(self):
        factory, _ = self._make_mock_factory([])
        result = benchmark_pipeline(factory, "x", "t", "m", n_warm_runs=1)
        assert result.cold_start_ms >= 0

    def test_warm_latency_ms_is_nonnegative(self):
        factory, _ = self._make_mock_factory([])
        result = benchmark_pipeline(factory, "x", "t", "m", n_warm_runs=1)
        assert result.warm_latency_ms >= 0

    def test_throughput_is_nonnegative(self):
        factory, _ = self._make_mock_factory([])
        result = benchmark_pipeline(factory, "x", "t", "m", n_warm_runs=1)
        assert result.throughput_samples_per_sec >= 0

    def test_pipe_called_correct_number_of_times(self):
        """Pipe should be called: 1 warm-up + n_warm_runs timed runs."""
        factory, mock_pipe = self._make_mock_factory("output")
        n = 4
        benchmark_pipeline(factory, "input", "t", "m", n_warm_runs=n)
        # 1 warm-up + n timed calls
        assert mock_pipe.call_count == 1 + n

    def test_pipe_called_with_correct_inputs(self):
        factory, mock_pipe = self._make_mock_factory("output")
        benchmark_pipeline(factory, "my-input", "t", "m", n_warm_runs=2)
        for c in mock_pipe.call_args_list:
            assert c == call("my-input")

    def test_single_input_throughput(self):
        """With a single string input, throughput = 1000 / warm_latency."""
        factory, mock_pipe = self._make_mock_factory("out")
        result = benchmark_pipeline(factory, "single", "t", "m", n_warm_runs=1)
        if result.warm_latency_ms > 0:
            expected = 1000.0 / result.warm_latency_ms
            assert abs(result.throughput_samples_per_sec - expected) < 1e-6

    def test_list_input_throughput(self):
        """With a list of N inputs, throughput = N * 1000 / warm_latency."""
        inputs = ["a", "b", "c"]
        factory, mock_pipe = self._make_mock_factory(["out", "out", "out"])
        result = benchmark_pipeline(factory, inputs, "t", "m", n_warm_runs=1)
        if result.warm_latency_ms > 0:
            expected = len(inputs) * 1000.0 / result.warm_latency_ms
            assert abs(result.throughput_samples_per_sec - expected) < 1e-6

    def test_factory_called_once(self):
        """pipe_factory should be called exactly once (for cold-start)."""
        call_count = [0]
        mock_pipe = MagicMock()
        mock_pipe.return_value = "output"

        def factory():
            call_count[0] += 1
            return mock_pipe

        benchmark_pipeline(factory, "x", "t", "m", n_warm_runs=3)
        assert call_count[0] == 1

    def test_zero_warm_runs_zero_latency(self):
        """With n_warm_runs=0, warm_latency and throughput should be 0."""
        factory, _ = self._make_mock_factory("out")
        result = benchmark_pipeline(factory, "x", "t", "m", n_warm_runs=0)
        assert result.warm_latency_ms == 0.0
        assert result.throughput_samples_per_sec == 0.0
        assert result.num_warm_runs == 0
