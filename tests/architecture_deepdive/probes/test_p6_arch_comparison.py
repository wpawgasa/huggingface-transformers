"""Tests for src/architecture_deepdive/probes/p6_arch_comparison.py."""

from unittest.mock import MagicMock, patch

import torch

from src.architecture_deepdive.probes.p6_arch_comparison import (
    SHARED_INPUT,
    probe_decoder_only,
    probe_encoder_decoder,
    probe_encoder_only,
    run_experiment,
)


def _mock_tokenizer(seq_len=10):
    """Create a mock tokenizer."""
    mock = MagicMock()
    result = MagicMock()
    result.to.return_value = result
    result.__getitem__ = MagicMock(
        side_effect=lambda key: (
            torch.tensor([list(range(seq_len))])
            if key == "input_ids"
            else torch.tensor([[1] * seq_len])
        )
    )
    mock.return_value = result
    mock.convert_ids_to_tokens.return_value = [f"tok{i}" for i in range(seq_len)]
    mock.decode = MagicMock(return_value="Generated text output")
    return mock


@patch("src.architecture_deepdive.probes.p6_arch_comparison.AutoModel")
@patch("src.architecture_deepdive.probes.p6_arch_comparison.AutoTokenizer")
class TestProbeEncoderOnly:
    def _setup(self, mock_tok_cls, mock_model_cls, seq_len=10):
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer(seq_len)

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        # Bidirectional attention: all values > 0
        attn = torch.rand(1, 12, seq_len, seq_len) + 0.01
        outputs = MagicMock()
        outputs.last_hidden_state = torch.randn(1, seq_len, 768)
        outputs.attentions = (attn,)
        mock_model.return_value = outputs
        mock_model_cls.from_pretrained.return_value = mock_model

    def test_returns_dict(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = probe_encoder_only(device="cpu")
        assert isinstance(result, dict)

    def test_correct_architecture(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = probe_encoder_only(device="cpu")
        assert result["architecture"] == "encoder-only"

    def test_has_hidden_state_shape(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = probe_encoder_only(device="cpu")
        assert "hidden_state_shape" in result
        assert len(result["hidden_state_shape"]) == 3

    def test_bidirectional_attention(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = probe_encoder_only(device="cpu")
        assert result["attention_is_bidirectional"] is True

    def test_has_cls_embedding_norm(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = probe_encoder_only(device="cpu")
        assert "cls_embedding_norm" in result
        assert isinstance(result["cls_embedding_norm"], float)


@patch("src.architecture_deepdive.probes.p6_arch_comparison.AutoModelForCausalLM")
@patch("src.architecture_deepdive.probes.p6_arch_comparison.AutoTokenizer")
class TestProbeDecoderOnly:
    def _setup(self, mock_tok_cls, mock_model_cls, seq_len=10):
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer(seq_len)

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        # Causal attention: lower triangular
        attn = torch.tril(torch.rand(1, 12, seq_len, seq_len) + 0.01)
        outputs = MagicMock()
        outputs.logits = torch.randn(1, seq_len, 50257)
        outputs.attentions = (attn,)
        mock_model.return_value = outputs
        mock_model_cls.from_pretrained.return_value = mock_model

    def test_returns_dict(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = probe_decoder_only(device="cpu")
        assert isinstance(result, dict)

    def test_correct_architecture(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = probe_decoder_only(device="cpu")
        assert result["architecture"] == "decoder-only"

    def test_causal_attention(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = probe_decoder_only(device="cpu")
        assert result["attention_is_causal"] is True
        assert result["upper_triangle_attn_sum"] < 1e-6

    def test_has_next_token_predictions(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = probe_decoder_only(device="cpu")
        assert "next_token_predictions" in result
        assert len(result["next_token_predictions"]) == 5

    def test_has_logits_shape(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = probe_decoder_only(device="cpu")
        assert "logits_shape" in result
        assert result["vocab_size"] == 50257


@patch("src.architecture_deepdive.probes.p6_arch_comparison.AutoModelForSeq2SeqLM")
@patch("src.architecture_deepdive.probes.p6_arch_comparison.AutoTokenizer")
class TestProbeEncoderDecoder:
    def _setup(self, mock_tok_cls, mock_model_cls, seq_len=12):
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer(seq_len)

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        # Generate returns token ids
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        # Encoder outputs
        mock_encoder = MagicMock()
        encoder_out = MagicMock()
        encoder_out.last_hidden_state = torch.randn(1, seq_len, 512)
        mock_encoder.return_value = encoder_out
        mock_model.get_encoder.return_value = mock_encoder

        mock_model_cls.from_pretrained.return_value = mock_model

    def test_returns_dict(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = probe_encoder_decoder(device="cpu")
        assert isinstance(result, dict)

    def test_correct_architecture(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = probe_encoder_decoder(device="cpu")
        assert result["architecture"] == "encoder-decoder"

    def test_has_generated_text(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = probe_encoder_decoder(device="cpu")
        assert "generated_text" in result
        assert isinstance(result["generated_text"], str)

    def test_has_encoder_hidden_shape(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = probe_encoder_decoder(device="cpu")
        assert "encoder_hidden_shape" in result
        assert len(result["encoder_hidden_shape"]) == 3

    def test_has_key_feature(self, mock_tok_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_model_cls)
        result = probe_encoder_decoder(device="cpu")
        assert "key_feature" in result


@patch("src.architecture_deepdive.probes.p6_arch_comparison.AutoModelForSeq2SeqLM")
@patch("src.architecture_deepdive.probes.p6_arch_comparison.AutoModelForCausalLM")
@patch("src.architecture_deepdive.probes.p6_arch_comparison.AutoModel")
@patch("src.architecture_deepdive.probes.p6_arch_comparison.AutoTokenizer")
class TestRunExperiment:
    def _setup(self, mock_tok_cls, mock_model_cls, mock_clm_cls, mock_s2s_cls):
        seq_len = 10
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer(seq_len)

        # Encoder-only (AutoModel)
        enc_model = MagicMock()
        enc_model.to.return_value = enc_model
        enc_model.eval.return_value = enc_model
        enc_attn = torch.rand(1, 12, seq_len, seq_len) + 0.01
        enc_out = MagicMock()
        enc_out.last_hidden_state = torch.randn(1, seq_len, 768)
        enc_out.attentions = (enc_attn,)
        enc_model.return_value = enc_out
        mock_model_cls.from_pretrained.return_value = enc_model

        # Decoder-only (AutoModelForCausalLM)
        dec_model = MagicMock()
        dec_model.to.return_value = dec_model
        dec_model.eval.return_value = dec_model
        dec_attn = torch.tril(torch.rand(1, 12, seq_len, seq_len) + 0.01)
        dec_out = MagicMock()
        dec_out.logits = torch.randn(1, seq_len, 50257)
        dec_out.attentions = (dec_attn,)
        dec_model.return_value = dec_out
        mock_clm_cls.from_pretrained.return_value = dec_model

        # Encoder-decoder (AutoModelForSeq2SeqLM)
        s2s_model = MagicMock()
        s2s_model.to.return_value = s2s_model
        s2s_model.eval.return_value = s2s_model
        s2s_model.generate.return_value = torch.tensor([[1, 2, 3]])
        mock_enc = MagicMock()
        enc_output = MagicMock()
        enc_output.last_hidden_state = torch.randn(1, seq_len, 512)
        mock_enc.return_value = enc_output
        s2s_model.get_encoder.return_value = mock_enc
        mock_s2s_cls.from_pretrained.return_value = s2s_model

    def test_returns_dict(self, mock_tok_cls, mock_model_cls, mock_clm_cls, mock_s2s_cls):
        self._setup(mock_tok_cls, mock_model_cls, mock_clm_cls, mock_s2s_cls)
        result = run_experiment(device="cpu")
        assert isinstance(result, dict)

    def test_has_task_key(self, mock_tok_cls, mock_model_cls, mock_clm_cls, mock_s2s_cls):
        self._setup(mock_tok_cls, mock_model_cls, mock_clm_cls, mock_s2s_cls)
        result = run_experiment(device="cpu")
        assert result["task"] == "architecture_comparison"

    def test_has_all_three_architectures(
        self, mock_tok_cls, mock_model_cls, mock_clm_cls, mock_s2s_cls
    ):
        self._setup(mock_tok_cls, mock_model_cls, mock_clm_cls, mock_s2s_cls)
        result = run_experiment(device="cpu")
        assert "encoder_only" in result
        assert "decoder_only" in result
        assert "encoder_decoder" in result

    def test_has_synthesis(self, mock_tok_cls, mock_model_cls, mock_clm_cls, mock_s2s_cls):
        self._setup(mock_tok_cls, mock_model_cls, mock_clm_cls, mock_s2s_cls)
        result = run_experiment(device="cpu")
        assert "synthesis" in result
        syn = result["synthesis"]
        assert "attention_patterns" in syn
        assert "output_format" in syn
        assert "task_suitability" in syn
        assert "course_quote" in syn

    def test_shared_input(self, mock_tok_cls, mock_model_cls, mock_clm_cls, mock_s2s_cls):
        self._setup(mock_tok_cls, mock_model_cls, mock_clm_cls, mock_s2s_cls)
        result = run_experiment(device="cpu")
        assert result["shared_input"] == SHARED_INPUT
