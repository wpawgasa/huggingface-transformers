"""Tests for src/architecture_deepdive/probes/p3_transfer_learning.py."""

from unittest.mock import MagicMock, patch

import torch

from src.architecture_deepdive.probes.p3_transfer_learning import (
    SimpleClassifier,
    prepare_data,
    run_experiment,
    train_and_evaluate,
)


def _mock_tokenizer():
    """Create a mock tokenizer that returns padded tensors."""
    mock = MagicMock()

    def tokenize(texts, padding=True, truncation=True, return_tensors=None):
        n = len(texts)
        return {
            "input_ids": torch.randint(0, 1000, (n, 16)),
            "attention_mask": torch.ones(n, 16, dtype=torch.long),
        }

    mock.side_effect = tokenize
    return mock


def _mock_base_model(hidden_size=128):
    """Create a mock base model that returns fake hidden states."""
    mock = MagicMock()
    mock.to.return_value = mock

    def forward(input_ids=None, attention_mask=None):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        outputs = MagicMock()
        outputs.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
        return outputs

    mock.side_effect = forward
    mock.__call__ = forward
    # Ensure it can be used in nn.Module
    mock.parameters = MagicMock(return_value=iter([]))
    mock.train = MagicMock(return_value=mock)
    mock.eval = MagicMock(return_value=mock)
    return mock


class TestSimpleClassifier:
    def test_forward_shape(self):
        base = _mock_base_model(hidden_size=128)
        classifier = SimpleClassifier(base, hidden_size=128, num_labels=2)
        input_ids = torch.randint(0, 1000, (4, 16))
        attention_mask = torch.ones(4, 16, dtype=torch.long)
        output = classifier(input_ids, attention_mask)
        assert output.shape == (4, 2)

    def test_forward_with_single_sample(self):
        base = _mock_base_model(hidden_size=64)
        classifier = SimpleClassifier(base, hidden_size=64, num_labels=2)
        input_ids = torch.randint(0, 1000, (1, 8))
        attention_mask = torch.ones(1, 8, dtype=torch.long)
        output = classifier(input_ids, attention_mask)
        assert output.shape == (1, 2)


@patch("src.architecture_deepdive.probes.p3_transfer_learning.AutoTokenizer")
class TestPrepareData:
    def test_returns_tensor_dataset(self, mock_tok_cls):
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        ds = prepare_data("test-model", "train")
        assert len(ds) == 8  # 8 training examples

    def test_test_split(self, mock_tok_cls):
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        ds = prepare_data("test-model", "test")
        assert len(ds) == 4  # 4 test examples

    def test_dataset_has_three_tensors(self, mock_tok_cls):
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()
        ds = prepare_data("test-model", "train")
        # input_ids, attention_mask, labels
        assert len(ds.tensors) == 3


@patch("src.architecture_deepdive.probes.p3_transfer_learning.AutoModel")
@patch("src.architecture_deepdive.probes.p3_transfer_learning.AutoConfig")
@patch("src.architecture_deepdive.probes.p3_transfer_learning.AutoTokenizer")
class TestTrainAndEvaluate:
    def _setup(self, mock_tok_cls, mock_cfg_cls, mock_model_cls):
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()

        mock_config = MagicMock()
        mock_config.hidden_size = 128
        mock_cfg_cls.from_pretrained.return_value = mock_config

        # Wrap in something that returns .last_hidden_state
        class FakeBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 128)

            def forward(self, input_ids=None, attention_mask=None):
                # Simple: embed input_ids as float, project
                x = input_ids.float()
                out = self.linear(x)
                result = MagicMock()
                result.last_hidden_state = out.unsqueeze(0) if out.dim() == 1 else out
                # Ensure shape is (batch, seq, hidden)
                if result.last_hidden_state.dim() == 2:
                    result.last_hidden_state = result.last_hidden_state.unsqueeze(1)
                return result

        fake_base = FakeBase()
        mock_model_cls.from_pretrained.return_value = fake_base
        mock_model_cls.from_config.return_value = FakeBase()

    def test_pretrained_returns_dict(self, mock_tok_cls, mock_cfg_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_cfg_cls, mock_model_cls)
        result = train_and_evaluate("test-model", from_scratch=False, num_epochs=2, device="cpu")
        assert isinstance(result, dict)

    def test_pretrained_mode(self, mock_tok_cls, mock_cfg_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_cfg_cls, mock_model_cls)
        result = train_and_evaluate("test-model", from_scratch=False, num_epochs=2)
        assert result["mode"] == "pretrained"

    def test_scratch_mode(self, mock_tok_cls, mock_cfg_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_cfg_cls, mock_model_cls)
        result = train_and_evaluate("test-model", from_scratch=True, num_epochs=2)
        assert result["mode"] == "from_scratch"

    def test_has_history(self, mock_tok_cls, mock_cfg_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_cfg_cls, mock_model_cls)
        result = train_and_evaluate("test-model", num_epochs=3)
        assert "history" in result
        assert len(result["history"]["epochs"]) == 3
        assert len(result["history"]["train_loss"]) == 3
        assert len(result["history"]["test_accuracy"]) == 3

    def test_has_final_accuracy(self, mock_tok_cls, mock_cfg_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_cfg_cls, mock_model_cls)
        result = train_and_evaluate("test-model", num_epochs=2)
        assert "final_accuracy" in result
        assert 0 <= result["final_accuracy"] <= 1

    def test_accuracy_values_in_range(self, mock_tok_cls, mock_cfg_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_cfg_cls, mock_model_cls)
        result = train_and_evaluate("test-model", num_epochs=2)
        for acc in result["history"]["test_accuracy"]:
            assert 0 <= acc <= 1


@patch("src.architecture_deepdive.probes.p3_transfer_learning.AutoModel")
@patch("src.architecture_deepdive.probes.p3_transfer_learning.AutoConfig")
@patch("src.architecture_deepdive.probes.p3_transfer_learning.AutoTokenizer")
class TestRunExperiment:
    def _setup(self, mock_tok_cls, mock_cfg_cls, mock_model_cls):
        mock_tok_cls.from_pretrained.return_value = _mock_tokenizer()

        mock_config = MagicMock()
        mock_config.hidden_size = 128
        mock_cfg_cls.from_pretrained.return_value = mock_config

        class FakeBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(16, 128)

            def forward(self, input_ids=None, attention_mask=None):
                x = input_ids.float()
                out = self.linear(x)
                result = MagicMock()
                result.last_hidden_state = out.unsqueeze(0) if out.dim() == 1 else out
                if result.last_hidden_state.dim() == 2:
                    result.last_hidden_state = result.last_hidden_state.unsqueeze(1)
                return result

        mock_model_cls.from_pretrained.return_value = FakeBase()
        mock_model_cls.from_config.return_value = FakeBase()

    def test_returns_dict(self, mock_tok_cls, mock_cfg_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_cfg_cls, mock_model_cls)
        result = run_experiment(device="cpu")
        assert isinstance(result, dict)

    def test_has_task_key(self, mock_tok_cls, mock_cfg_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_cfg_cls, mock_model_cls)
        result = run_experiment(device="cpu")
        assert result["task"] == "transfer_learning"

    def test_has_both_modes(self, mock_tok_cls, mock_cfg_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_cfg_cls, mock_model_cls)
        result = run_experiment(device="cpu")
        assert "pretrained" in result
        assert "from_scratch" in result

    def test_has_analysis(self, mock_tok_cls, mock_cfg_cls, mock_model_cls):
        self._setup(mock_tok_cls, mock_cfg_cls, mock_model_cls)
        result = run_experiment(device="cpu")
        assert "analysis" in result
        analysis = result["analysis"]
        assert "accuracy_gap" in analysis
        assert "conclusion" in analysis
