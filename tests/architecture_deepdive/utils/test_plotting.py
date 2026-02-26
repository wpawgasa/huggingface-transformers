"""Tests for src/architecture_deepdive/utils/plotting.py."""

from unittest.mock import MagicMock, patch

import numpy as np

from src.architecture_deepdive.utils.plotting import (
    plot_attention_mask_comparison,
    plot_attention_matrix,
    plot_parameter_comparison,
    plot_transfer_learning_comparison,
)


@patch("src.architecture_deepdive.utils.plotting.plt")
@patch("src.architecture_deepdive.utils.plotting.sns")
@patch("src.architecture_deepdive.utils.plotting._ensure_figure_dir")
class TestPlotAttentionMatrix:
    def _setup(self, mock_plt):
        mock_plt.subplots.return_value = (MagicMock(), MagicMock())

    def test_calls_savefig(self, mock_ensure, mock_sns, mock_plt):
        self._setup(mock_plt)
        attn = np.random.rand(2, 2, 4, 4)
        tokens = ["a", "b", "c", "d"]
        plot_attention_matrix(attn, tokens, save_name="test.png")
        mock_plt.savefig.assert_called_once()

    def test_calls_close(self, mock_ensure, mock_sns, mock_plt):
        self._setup(mock_plt)
        attn = np.random.rand(2, 2, 4, 4)
        tokens = ["a", "b", "c", "d"]
        plot_attention_matrix(attn, tokens)
        mock_plt.close.assert_called_once()

    def test_uses_heatmap(self, mock_ensure, mock_sns, mock_plt):
        self._setup(mock_plt)
        attn = np.random.rand(2, 2, 4, 4)
        tokens = ["a", "b", "c", "d"]
        plot_attention_matrix(attn, tokens)
        mock_sns.heatmap.assert_called_once()

    def test_respects_layer_head_params(self, mock_ensure, mock_sns, mock_plt):
        self._setup(mock_plt)
        attn = np.random.rand(3, 4, 4, 4)
        tokens = ["a", "b", "c", "d"]
        plot_attention_matrix(attn, tokens, layer=1, head=2)
        call_args = mock_sns.heatmap.call_args
        # First positional arg is the attention data
        actual_attn = call_args[0][0]
        np.testing.assert_array_equal(actual_attn, attn[1, 2])


@patch("src.architecture_deepdive.utils.plotting.plt")
@patch("src.architecture_deepdive.utils.plotting.sns")
@patch("src.architecture_deepdive.utils.plotting._ensure_figure_dir")
class TestPlotAttentionMaskComparison:
    def test_calls_savefig(self, mock_ensure, mock_sns, mock_plt):
        bi = np.ones((4, 4))
        causal = np.tril(np.ones((4, 4)))
        tokens = ["a", "b", "c", "d"]
        mock_plt.subplots.return_value = (MagicMock(), [MagicMock(), MagicMock()])
        plot_attention_mask_comparison(bi, causal, tokens)
        mock_plt.savefig.assert_called_once()

    def test_creates_two_subplots(self, mock_ensure, mock_sns, mock_plt):
        bi = np.ones((4, 4))
        causal = np.tril(np.ones((4, 4)))
        tokens = ["a", "b", "c", "d"]
        mock_plt.subplots.return_value = (MagicMock(), [MagicMock(), MagicMock()])
        plot_attention_mask_comparison(bi, causal, tokens)
        mock_plt.subplots.assert_called_once_with(1, 2, figsize=(14, 5))


@patch("src.architecture_deepdive.utils.plotting.plt")
@patch("src.architecture_deepdive.utils.plotting._ensure_figure_dir")
class TestPlotParameterComparison:
    def test_calls_savefig(self, mock_ensure, mock_plt):
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)
        mock_ax.barh.return_value = [MagicMock()]
        data = [
            {"name": "bert", "family": "encoder_only", "params": 110_000_000},
        ]
        plot_parameter_comparison(data)
        mock_plt.savefig.assert_called_once()

    def test_handles_multiple_models(self, mock_ensure, mock_plt):
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)
        bars = [MagicMock() for _ in range(3)]
        mock_ax.barh.return_value = bars
        data = [
            {"name": "bert", "family": "encoder_only", "params": 110_000_000},
            {"name": "gpt2", "family": "decoder_only", "params": 124_000_000},
            {"name": "t5", "family": "encoder_decoder", "params": 60_000_000},
        ]
        plot_parameter_comparison(data)
        mock_plt.savefig.assert_called_once()


@patch("src.architecture_deepdive.utils.plotting.plt")
@patch("src.architecture_deepdive.utils.plotting._ensure_figure_dir")
class TestPlotTransferLearningComparison:
    def test_calls_savefig(self, mock_ensure, mock_plt):
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)
        plot_transfer_learning_comparison(
            pretrained_scores=[0.5, 0.8, 0.9],
            scratch_scores=[0.5, 0.55, 0.6],
            epochs=[1, 2, 3],
        )
        mock_plt.savefig.assert_called_once()

    def test_calls_close(self, mock_ensure, mock_plt):
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (MagicMock(), mock_ax)
        plot_transfer_learning_comparison(
            pretrained_scores=[0.5, 0.8],
            scratch_scores=[0.5, 0.55],
            epochs=[1, 2],
        )
        mock_plt.close.assert_called_once()
