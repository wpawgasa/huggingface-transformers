"""Tests for src/architecture_deepdive/probes/p5_attention_viz.py."""

from unittest.mock import patch

import numpy as np

from src.architecture_deepdive.probes.p5_attention_viz import run_experiment


def _mock_attn_data(tokens=None, num_layers=12, num_heads=12, seq_len=8):
    """Create a mock attention extraction result."""
    if tokens is None:
        tokens = [f"tok{i}" for i in range(seq_len)]
    attn = np.random.rand(num_layers, num_heads, seq_len, seq_len)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    return {
        "tokens": tokens,
        "attentions": attn,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "seq_len": seq_len,
    }


@patch("src.architecture_deepdive.probes.p5_attention_viz.plot_attention_matrix")
@patch("src.architecture_deepdive.probes.p5_attention_viz.plot_attention_mask_comparison")
@patch("src.architecture_deepdive.probes.p5_attention_viz.get_attention_to_token")
@patch("src.architecture_deepdive.probes.p5_attention_viz.extract_attention_weights")
class TestRunExperiment:
    def _setup(self, mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn):
        # extract_attention_weights returns mock data
        mock_extract.return_value = _mock_attn_data(
            tokens=[
                "[CLS]",
                "the",
                "animal",
                "didn",
                "'",
                "t",
                "cross",
                "the",
                "street",
                "because",
                "it",
                "was",
                "too",
                "tired",
                ".",
                "[SEP]",
            ],
            seq_len=16,
        )
        # get_attention_to_token returns a simple dict
        mock_get_attn.return_value = {
            "[CLS]": 0.05,
            "the": 0.1,
            "animal": 0.3,
            "it": 0.2,
            "tired": 0.1,
        }

    def test_returns_dict(self, mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn):
        self._setup(mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn)
        result = run_experiment(device="cpu")
        assert isinstance(result, dict)

    def test_has_task_key(self, mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn):
        self._setup(mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn)
        result = run_experiment(device="cpu")
        assert result["task"] == "attention_visualization"

    def test_has_mask_comparison(self, mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn):
        self._setup(mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn)
        result = run_experiment(device="cpu")
        assert "mask_comparison" in result
        mc = result["mask_comparison"]
        assert "tokens" in mc
        assert "causal_masked_positions" in mc

    def test_has_figures(self, mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn):
        self._setup(mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn)
        result = run_experiment(device="cpu")
        assert "figures" in result
        assert len(result["figures"]) > 0

    def test_has_coreference_test(
        self, mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn
    ):
        self._setup(mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn)
        result = run_experiment(device="cpu")
        assert "coreference_test" in result
        ct = result["coreference_test"]
        assert "hypothesis" in ct
        assert "tired_sentence" in ct
        assert "wide_sentence" in ct

    def test_has_causal_observation(
        self, mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn
    ):
        self._setup(mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn)
        result = run_experiment(device="cpu")
        assert "causal_observation" in result

    def test_calls_extract_for_bert_and_gpt(
        self, mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn
    ):
        self._setup(mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn)
        run_experiment(device="cpu")
        # Called for: 6 attention sentences + 2 coreference + 1 gpt2 = 9
        assert mock_extract.call_count == 9

    def test_mask_comparison_plot_called(
        self, mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn
    ):
        self._setup(mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn)
        run_experiment(device="cpu")
        mock_plot_mask.assert_called_once()

    def test_attention_matrix_plots_called(
        self, mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn
    ):
        self._setup(mock_extract, mock_get_attn, mock_plot_mask, mock_plot_attn)
        run_experiment(device="cpu")
        # 6 sentences * 2 (first + last layer) + 1 gpt2 = 13
        assert mock_plot_attn.call_count == 13
