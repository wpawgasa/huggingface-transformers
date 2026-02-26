"""Tests for src/architecture_deepdive/utils/model_inspector.py."""

from unittest.mock import MagicMock, patch

import torch

from src.architecture_deepdive.utils.model_inspector import (
    ModelAnatomy,
    format_param_count,
    inspect_model,
)


class TestFormatParamCount:
    def test_billions(self):
        assert format_param_count(1_500_000_000) == "1.5B"

    def test_millions(self):
        assert format_param_count(110_000_000) == "110.0M"

    def test_thousands(self):
        assert format_param_count(5_000) == "5.0K"

    def test_small(self):
        assert format_param_count(42) == "42"

    def test_one_billion(self):
        assert format_param_count(1_000_000_000) == "1.0B"


class TestModelAnatomy:
    def test_dataclass_creation(self):
        anatomy = ModelAnatomy(
            name="test-model",
            architecture_class="TestModel",
            family="encoder_only",
            num_parameters=1000,
            num_trainable_parameters=1000,
            num_layers=2,
            hidden_size=64,
            num_attention_heads=2,
            intermediate_size=256,
            vocab_size=30000,
            max_position_embeddings=512,
            has_encoder=True,
            has_decoder=False,
            layer_breakdown={"embeddings": {"count": 3, "params": 500}},
        )
        assert anatomy.name == "test-model"
        assert anatomy.num_parameters == 1000
        assert anatomy.has_encoder is True
        assert anatomy.has_decoder is False


@patch("src.architecture_deepdive.utils.model_inspector.AutoModel")
@patch("src.architecture_deepdive.utils.model_inspector.AutoConfig")
class TestInspectModel:
    def _setup_mocks(self, mock_config_cls, mock_model_cls):
        # Config mock
        mock_config = MagicMock()
        mock_config.num_hidden_layers = 12
        mock_config.hidden_size = 768
        mock_config.num_attention_heads = 12
        mock_config.intermediate_size = 3072
        mock_config.vocab_size = 30522
        mock_config.max_position_embeddings = 512
        mock_config_cls.from_pretrained.return_value = mock_config

        # Model mock with parameters
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "BertModel"

        # Create fake parameters
        param1 = torch.nn.Parameter(torch.randn(768, 768))
        param1.requires_grad = True
        param2 = torch.nn.Parameter(torch.randn(768))
        param2.requires_grad = True

        mock_model.parameters.return_value = [param1, param2]
        mock_model.named_parameters.return_value = [
            ("embeddings.weight", param1),
            ("encoder.bias", param2),
        ]
        mock_model_cls.from_pretrained.return_value = mock_model

        return mock_config, mock_model

    def test_returns_model_anatomy(self, mock_config_cls, mock_model_cls):
        self._setup_mocks(mock_config_cls, mock_model_cls)
        result = inspect_model("bert-base-uncased", "encoder_only")
        assert isinstance(result, ModelAnatomy)

    def test_correct_name(self, mock_config_cls, mock_model_cls):
        self._setup_mocks(mock_config_cls, mock_model_cls)
        result = inspect_model("bert-base-uncased", "encoder_only")
        assert result.name == "bert-base-uncased"

    def test_correct_family(self, mock_config_cls, mock_model_cls):
        self._setup_mocks(mock_config_cls, mock_model_cls)
        result = inspect_model("bert-base-uncased", "encoder_only")
        assert result.family == "encoder_only"

    def test_parameter_count(self, mock_config_cls, mock_model_cls):
        self._setup_mocks(mock_config_cls, mock_model_cls)
        result = inspect_model("bert-base-uncased", "encoder_only")
        # 768*768 + 768 = 590592 + 768 = 591360
        assert result.num_parameters == 768 * 768 + 768
        assert result.num_trainable_parameters == 768 * 768 + 768

    def test_config_values(self, mock_config_cls, mock_model_cls):
        self._setup_mocks(mock_config_cls, mock_model_cls)
        result = inspect_model("bert-base-uncased", "encoder_only")
        assert result.num_layers == 12
        assert result.hidden_size == 768
        assert result.num_attention_heads == 12
        assert result.vocab_size == 30522

    def test_layer_breakdown(self, mock_config_cls, mock_model_cls):
        self._setup_mocks(mock_config_cls, mock_model_cls)
        result = inspect_model("bert-base-uncased", "encoder_only")
        assert "embeddings" in result.layer_breakdown
        assert "encoder" in result.layer_breakdown
        assert result.layer_breakdown["embeddings"]["count"] == 1
        assert result.layer_breakdown["encoder"]["count"] == 1

    def test_encoder_only_flags(self, mock_config_cls, mock_model_cls):
        self._setup_mocks(mock_config_cls, mock_model_cls)
        result = inspect_model("bert-base-uncased", "encoder_only")
        assert result.has_encoder is True
        assert result.has_decoder is False

    def test_decoder_only_flags(self, mock_config_cls, mock_model_cls):
        self._setup_mocks(mock_config_cls, mock_model_cls)
        result = inspect_model("gpt2", "decoder_only")
        assert result.has_encoder is False
        assert result.has_decoder is True

    def test_encoder_decoder_flags(self, mock_config_cls, mock_model_cls):
        self._setup_mocks(mock_config_cls, mock_model_cls)
        result = inspect_model("t5-small", "encoder_decoder")
        assert result.has_encoder is True
        assert result.has_decoder is True

    def test_fallback_config_fields(self, mock_config_cls, mock_model_cls):
        """Test getattr chains for GPT-style config field names."""
        mock_config = MagicMock(spec=[])
        mock_config.n_layer = 12
        mock_config.n_embd = 768
        mock_config.n_head = 12
        mock_config.n_inner = 3072
        mock_config.vocab_size = 50257
        mock_config.n_positions = 1024
        mock_config_cls.from_pretrained.return_value = mock_config

        mock_model = MagicMock()
        mock_model.__class__.__name__ = "GPT2Model"
        param = torch.nn.Parameter(torch.randn(10))
        mock_model.parameters.return_value = [param]
        mock_model.named_parameters.return_value = [("wte.weight", param)]
        mock_model_cls.from_pretrained.return_value = mock_model

        result = inspect_model("gpt2", "decoder_only")
        assert result.num_layers == 12
        assert result.hidden_size == 768
        assert result.max_position_embeddings == 1024
