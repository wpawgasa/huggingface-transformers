"""Tests for src/architecture_deepdive/probes/p2_language_modeling.py."""

from unittest.mock import MagicMock, patch

import torch

from src.architecture_deepdive.probes.p2_language_modeling import (
    run_causal_lm_probe,
    run_experiment,
    run_masked_lm_probe,
)


def _mock_tokenizer(vocab_size=50257, mask_token_id=103):
    """Create a mock tokenizer."""
    mock = MagicMock()
    mock.mask_token_id = mask_token_id
    mock.return_value = MagicMock()
    mock.return_value.to.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    # For masked sentences, embed [MASK] at position 3
    mask_input = MagicMock()
    mask_input.__getitem__ = MagicMock(
        side_effect=lambda key: (
            torch.tensor([[1, 2, 3, mask_token_id, 5]])
            if key == "input_ids"
            else torch.tensor([[1, 1, 1, 1, 1]])
        )
    )
    mock.decode = MagicMock(side_effect=lambda x: f"token_{x}")
    return mock


def _mock_causal_model(vocab_size=50257, seq_len=5):
    """Create a mock causal LM model."""
    mock = MagicMock()
    mock.to.return_value = mock
    mock.eval.return_value = mock

    outputs = MagicMock()
    outputs.logits = torch.randn(1, seq_len, vocab_size)
    mock.return_value = outputs

    return mock


def _mock_masked_model(vocab_size=30522, seq_len=5):
    """Create a mock masked LM model."""
    mock = MagicMock()
    mock.to.return_value = mock
    mock.eval.return_value = mock

    outputs = MagicMock()
    outputs.logits = torch.randn(1, seq_len, vocab_size)
    mock.return_value = outputs

    return mock


@patch("src.architecture_deepdive.probes.p2_language_modeling.AutoModelForCausalLM")
@patch("src.architecture_deepdive.probes.p2_language_modeling.AutoTokenizer")
class TestRunCausalLmProbe:
    def test_returns_dict(self, mock_tok_cls, mock_model_cls):
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_model_cls.from_pretrained.return_value = _mock_causal_model()
        result = run_causal_lm_probe(device="cpu")
        assert isinstance(result, dict)

    def test_has_model_key(self, mock_tok_cls, mock_model_cls):
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_model_cls.from_pretrained.return_value = _mock_causal_model()
        result = run_causal_lm_probe(device="cpu")
        assert result["model"] == "gpt2"

    def test_has_objective(self, mock_tok_cls, mock_model_cls):
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_model_cls.from_pretrained.return_value = _mock_causal_model()
        result = run_causal_lm_probe(device="cpu")
        assert result["objective"] == "Causal LM (CLM)"

    def test_results_count_matches_prompts(self, mock_tok_cls, mock_model_cls):
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_model_cls.from_pretrained.return_value = _mock_causal_model()
        result = run_causal_lm_probe(device="cpu")
        assert len(result["results"]) == 3  # 3 causal_prompts

    def test_each_result_has_top5(self, mock_tok_cls, mock_model_cls):
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_model_cls.from_pretrained.return_value = _mock_causal_model()
        result = run_causal_lm_probe(device="cpu")
        for r in result["results"]:
            assert "top_5_predictions" in r
            assert len(r["top_5_predictions"]) == 5

    def test_predictions_have_token_and_probability(self, mock_tok_cls, mock_model_cls):
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        mock_model_cls.from_pretrained.return_value = _mock_causal_model()
        result = run_causal_lm_probe(device="cpu")
        for r in result["results"]:
            for pred in r["top_5_predictions"]:
                assert "token" in pred
                assert "probability" in pred
                assert 0 <= pred["probability"] <= 1


@patch("src.architecture_deepdive.probes.p2_language_modeling.AutoModelForMaskedLM")
@patch("src.architecture_deepdive.probes.p2_language_modeling.AutoTokenizer")
class TestRunMaskedLmProbe:
    def _setup(self, mock_tok_cls, mock_model_cls):
        mock_tok = MagicMock()
        mock_tok.mask_token_id = 103

        def tokenize_side_effect(text, return_tensors=None):
            """Return input_ids with [MASK] at position 3."""
            result = MagicMock()
            result.to.return_value = result
            ids = torch.tensor([[101, 1, 2, 103, 4, 102]])
            result.__getitem__ = MagicMock(
                side_effect=lambda key: (
                    ids if key == "input_ids" else torch.tensor([[1, 1, 1, 1, 1, 1]])
                )
            )
            return result

        mock_tok.side_effect = tokenize_side_effect
        mock_tok.decode = MagicMock(return_value="paris")
        mock_tok_cls.from_pretrained.return_value = mock_tok

        mock_model = _mock_masked_model(seq_len=6)
        mock_model_cls.from_pretrained.return_value = mock_model

        return mock_tok, mock_model

    def test_returns_dict(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = run_masked_lm_probe(device="cpu")
        assert isinstance(result, dict)

    def test_has_correct_objective(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = run_masked_lm_probe(device="cpu")
        assert result["objective"] == "Masked LM (MLM)"

    def test_has_results(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = run_masked_lm_probe(device="cpu")
        assert "results" in result
        assert len(result["results"]) > 0


@patch("src.architecture_deepdive.probes.p2_language_modeling.AutoModelForMaskedLM")
@patch("src.architecture_deepdive.probes.p2_language_modeling.AutoModelForCausalLM")
@patch("src.architecture_deepdive.probes.p2_language_modeling.AutoTokenizer")
class TestRunExperiment:
    def _setup(self, mock_tok_cls, mock_clm_cls, mock_mlm_cls):
        mock_tok = MagicMock()
        mock_tok.mask_token_id = 103

        def tokenize_side_effect(text, return_tensors=None):
            result = MagicMock()
            result.to.return_value = result
            ids = torch.tensor([[101, 1, 2, 103, 4, 102]])
            result.__getitem__ = MagicMock(
                side_effect=lambda key: (
                    ids if key == "input_ids" else torch.tensor([[1, 1, 1, 1, 1, 1]])
                )
            )
            return result

        mock_tok.side_effect = tokenize_side_effect
        mock_tok.decode = MagicMock(return_value="token")
        mock_tok_cls.from_pretrained.return_value = mock_tok

        mock_clm_cls.from_pretrained.return_value = _mock_causal_model(seq_len=6)
        mock_mlm_cls.from_pretrained.return_value = _mock_masked_model(seq_len=6)

    def test_returns_dict(self, mock_tok_cls, mock_clm_cls, mock_mlm_cls):
        self._setup(mock_tok_cls, mock_clm_cls, mock_mlm_cls)
        result = run_experiment(device="cpu")
        assert isinstance(result, dict)

    def test_has_task_key(self, mock_tok_cls, mock_clm_cls, mock_mlm_cls):
        self._setup(mock_tok_cls, mock_clm_cls, mock_mlm_cls)
        result = run_experiment(device="cpu")
        assert result["task"] == "language_modeling_comparison"

    def test_has_both_probes(self, mock_tok_cls, mock_clm_cls, mock_mlm_cls):
        self._setup(mock_tok_cls, mock_clm_cls, mock_mlm_cls)
        result = run_experiment(device="cpu")
        assert "causal_lm" in result
        assert "masked_lm" in result

    def test_has_comparison(self, mock_tok_cls, mock_clm_cls, mock_mlm_cls):
        self._setup(mock_tok_cls, mock_clm_cls, mock_mlm_cls)
        result = run_experiment(device="cpu")
        assert "comparison_analysis" in result
        analysis = result["comparison_analysis"]
        assert "key_difference" in analysis
