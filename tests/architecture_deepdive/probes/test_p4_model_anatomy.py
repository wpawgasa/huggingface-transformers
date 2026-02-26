"""Tests for src/architecture_deepdive/probes/p4_model_anatomy.py."""

from unittest.mock import MagicMock, patch

from src.architecture_deepdive.probes.p4_model_anatomy import run_experiment


def _mock_inspect_model(model_name, family):
    """Return a mock ModelAnatomy."""
    mock = MagicMock()
    mock.name = model_name
    mock.architecture_class = f"{family.title()}Model"
    mock.family = family
    mock.num_parameters = 110_000_000
    mock.num_trainable_parameters = 110_000_000
    mock.num_layers = 12
    mock.hidden_size = 768
    mock.num_attention_heads = 12
    mock.intermediate_size = 3072
    mock.vocab_size = 30522
    mock.max_position_embeddings = 512
    mock.has_encoder = "encoder" in family
    mock.has_decoder = "decoder" in family
    mock.layer_breakdown = {
        "embeddings": {"count": 5, "params": 23_000_000},
        "encoder": {"count": 72, "params": 85_000_000},
    }
    return mock


@patch("src.architecture_deepdive.probes.p4_model_anatomy.plot_parameter_comparison")
@patch("src.architecture_deepdive.probes.p4_model_anatomy.inspect_model")
class TestRunExperiment:
    def test_returns_dict(self, mock_inspect, mock_plot):
        mock_inspect.side_effect = _mock_inspect_model
        result = run_experiment(device="cpu")
        assert isinstance(result, dict)

    def test_has_task_key(self, mock_inspect, mock_plot):
        mock_inspect.side_effect = _mock_inspect_model
        result = run_experiment(device="cpu")
        assert result["task"] == "model_anatomy"

    def test_has_all_families(self, mock_inspect, mock_plot):
        mock_inspect.side_effect = _mock_inspect_model
        result = run_experiment(device="cpu")
        assert "encoder_only" in result
        assert "decoder_only" in result
        assert "encoder_decoder" in result

    def test_family_has_expected_keys(self, mock_inspect, mock_plot):
        mock_inspect.side_effect = _mock_inspect_model
        result = run_experiment(device="cpu")
        for family in ("encoder_only", "decoder_only", "encoder_decoder"):
            entry = result[family]
            assert "model" in entry
            assert "architecture_class" in entry
            assert "parameters" in entry
            assert "num_parameters_raw" in entry
            assert "num_layers" in entry
            assert "hidden_size" in entry
            assert "num_attention_heads" in entry
            assert "layer_breakdown" in entry

    def test_has_comparison(self, mock_inspect, mock_plot):
        mock_inspect.side_effect = _mock_inspect_model
        result = run_experiment(device="cpu")
        assert "comparison" in result
        comp = result["comparison"]
        assert "architecture_vs_checkpoint_note" in comp
        assert "structural_comparison" in comp

    def test_structural_comparison_has_all_families(self, mock_inspect, mock_plot):
        mock_inspect.side_effect = _mock_inspect_model
        result = run_experiment(device="cpu")
        sc = result["comparison"]["structural_comparison"]
        assert "encoder_only (BERT)" in sc
        assert "decoder_only (GPT-2)" in sc
        assert "encoder_decoder (T5)" in sc

    def test_calls_inspect_for_each_family(self, mock_inspect, mock_plot):
        mock_inspect.side_effect = _mock_inspect_model
        run_experiment(device="cpu")
        assert mock_inspect.call_count == 3

    def test_calls_plot(self, mock_inspect, mock_plot):
        mock_inspect.side_effect = _mock_inspect_model
        run_experiment(device="cpu")
        mock_plot.assert_called_once()

    def test_plot_receives_correct_data(self, mock_inspect, mock_plot):
        mock_inspect.side_effect = _mock_inspect_model
        run_experiment(device="cpu")
        plot_data = mock_plot.call_args[0][0]
        assert len(plot_data) == 3
        for entry in plot_data:
            assert "name" in entry
            assert "family" in entry
            assert "params" in entry
