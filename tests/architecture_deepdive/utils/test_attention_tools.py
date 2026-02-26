"""Tests for src/architecture_deepdive/utils/attention_tools.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.architecture_deepdive.utils.attention_tools import (
    compare_causal_vs_bidirectional_mask,
    extract_attention_weights,
    get_attention_to_token,
)


class TestCompareCausalVsBidirectionalMask:
    def test_returns_expected_keys(self):
        result = compare_causal_vs_bidirectional_mask(4)
        assert "bidirectional" in result
        assert "causal" in result
        assert "difference" in result
        assert "causal_masked_positions" in result
        assert "note" in result

    def test_bidirectional_is_all_ones(self):
        result = compare_causal_vs_bidirectional_mask(4)
        np.testing.assert_array_equal(result["bidirectional"], np.ones((4, 4)))

    def test_causal_is_lower_triangular(self):
        result = compare_causal_vs_bidirectional_mask(4)
        expected = np.tril(np.ones((4, 4)))
        np.testing.assert_array_equal(result["causal"], expected)

    def test_masked_positions_count(self):
        result = compare_causal_vs_bidirectional_mask(4)
        # Upper triangle: 4*3/2 = 6 positions
        assert result["causal_masked_positions"] == 6

    def test_seq_len_1(self):
        result = compare_causal_vs_bidirectional_mask(1)
        assert result["causal_masked_positions"] == 0

    def test_note_contains_seq_len(self):
        result = compare_causal_vs_bidirectional_mask(5)
        assert "seq_len=5" in result["note"]


class TestGetAttentionToToken:
    def _make_attention(self, seq_len=5, num_layers=2, num_heads=2):
        """Create a random attention array."""
        attn = np.random.rand(num_layers, num_heads, seq_len, seq_len)
        # Normalize rows to sum to 1 (like softmax output)
        attn = attn / attn.sum(axis=-1, keepdims=True)
        return attn

    def test_returns_dict_for_valid_token(self):
        tokens = ["[CLS]", "the", "cat", "sat", "[SEP]"]
        attn = self._make_attention(seq_len=5)
        result = get_attention_to_token(attn, tokens, "cat")
        assert isinstance(result, dict)
        assert len(result) == 5

    def test_all_tokens_present_in_result(self):
        tokens = ["[CLS]", "the", "cat", "sat", "[SEP]"]
        attn = self._make_attention(seq_len=5)
        result = get_attention_to_token(attn, tokens, "cat")
        for tok in tokens:
            assert tok in result

    def test_values_are_floats(self):
        tokens = ["[CLS]", "the", "cat", "sat", "[SEP]"]
        attn = self._make_attention(seq_len=5)
        result = get_attention_to_token(attn, tokens, "cat")
        for val in result.values():
            assert isinstance(val, float)

    def test_token_not_found_returns_error(self):
        tokens = ["[CLS]", "the", "cat", "[SEP]"]
        attn = self._make_attention(seq_len=4)
        result = get_attention_to_token(attn, tokens, "zebra")
        assert "error" in result

    def test_case_insensitive_matching(self):
        tokens = ["[CLS]", "The", "Cat", "[SEP]"]
        attn = self._make_attention(seq_len=4)
        result = get_attention_to_token(attn, tokens, "cat")
        assert "error" not in result

    def test_uses_last_layer_by_default(self):
        tokens = ["a", "b", "c"]
        attn = np.zeros((3, 2, 3, 3))
        # Set last layer's values to something distinguishable
        attn[-1, :, :, 1] = 0.5
        result = get_attention_to_token(attn, tokens, "b", layer=-1)
        assert all(v >= 0 for v in result.values())


@patch("src.architecture_deepdive.utils.attention_tools.AutoModel")
@patch("src.architecture_deepdive.utils.attention_tools.AutoTokenizer")
class TestExtractAttentionWeights:
    def _setup_mocks(self, mock_tokenizer_cls, mock_model_cls, seq_len=5):
        # Tokenizer mock - must return object supporting .to() and [] access
        mock_tokenizer = MagicMock()
        inputs_data = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
        }
        mock_inputs = MagicMock()
        mock_inputs.to.return_value = mock_inputs
        mock_inputs.__getitem__ = lambda self, key: inputs_data[key]
        mock_inputs.items = lambda: inputs_data.items()
        mock_inputs.keys = lambda: inputs_data.keys()
        # Support **inputs unpacking
        for key, val in inputs_data.items():
            setattr(mock_inputs, key, val)
        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.convert_ids_to_tokens.return_value = [
            "[CLS]",
            "the",
            "cat",
            "sat",
            "[SEP]",
        ]
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Model mock
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        # Create fake attention tensors: 2 layers, 2 heads
        attn_layer = torch.rand(1, 2, seq_len, seq_len)
        mock_outputs = MagicMock()
        mock_outputs.attentions = (attn_layer, attn_layer)
        mock_model.__call__ = MagicMock(return_value=mock_outputs)
        mock_model.return_value = mock_outputs
        mock_model_cls.from_pretrained.return_value = mock_model

        return mock_tokenizer, mock_model

    def test_returns_expected_keys(self, mock_tokenizer_cls, mock_model_cls):
        self._setup_mocks(mock_tokenizer_cls, mock_model_cls)
        result = extract_attention_weights("bert-base-uncased", "test text")
        assert "tokens" in result
        assert "attentions" in result
        assert "num_layers" in result
        assert "num_heads" in result
        assert "seq_len" in result

    def test_tokens_list(self, mock_tokenizer_cls, mock_model_cls):
        self._setup_mocks(mock_tokenizer_cls, mock_model_cls)
        result = extract_attention_weights("bert-base-uncased", "test text")
        assert result["tokens"] == ["[CLS]", "the", "cat", "sat", "[SEP]"]

    def test_attention_shape(self, mock_tokenizer_cls, mock_model_cls):
        self._setup_mocks(mock_tokenizer_cls, mock_model_cls)
        result = extract_attention_weights("bert-base-uncased", "test text")
        attn = result["attentions"]
        assert isinstance(attn, np.ndarray)
        assert attn.shape == (2, 2, 5, 5)  # layers, heads, seq, seq

    def test_num_layers(self, mock_tokenizer_cls, mock_model_cls):
        self._setup_mocks(mock_tokenizer_cls, mock_model_cls)
        result = extract_attention_weights("bert-base-uncased", "test text")
        assert result["num_layers"] == 2

    def test_num_heads(self, mock_tokenizer_cls, mock_model_cls):
        self._setup_mocks(mock_tokenizer_cls, mock_model_cls)
        result = extract_attention_weights("bert-base-uncased", "test text")
        assert result["num_heads"] == 2

    def test_seq_len(self, mock_tokenizer_cls, mock_model_cls):
        self._setup_mocks(mock_tokenizer_cls, mock_model_cls)
        result = extract_attention_weights("bert-base-uncased", "test text")
        assert result["seq_len"] == 5

    def test_model_loaded_with_output_attentions(self, mock_tokenizer_cls, mock_model_cls):
        self._setup_mocks(mock_tokenizer_cls, mock_model_cls)
        extract_attention_weights("bert-base-uncased", "test text")
        mock_model_cls.from_pretrained.assert_called_once_with(
            "bert-base-uncased", output_attentions=True
        )
