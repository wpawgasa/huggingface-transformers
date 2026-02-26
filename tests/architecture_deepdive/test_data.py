"""Tests for src/architecture_deepdive/data.py."""

from src.architecture_deepdive.data import (
    ATTENTION_SENTENCES,
    COURSE_SENTENCES,
    LM_INPUTS,
    MODEL_REGISTRY,
    TRANSFER_LEARNING_DATA,
)


class TestCourseSentences:
    def test_has_required_keys(self):
        assert "translation_example" in COURSE_SENTENCES
        assert "french_target" in COURSE_SENTENCES
        assert "attention_demo" in COURSE_SENTENCES

    def test_values_are_nonempty_strings(self):
        for key, val in COURSE_SENTENCES.items():
            assert isinstance(val, str), f"{key} should be a string"
            assert len(val) > 0, f"{key} should not be empty"


class TestAttentionSentences:
    def test_has_agreement_sentences(self):
        assert "agreement_short" in ATTENTION_SENTENCES
        assert "agreement_long" in ATTENTION_SENTENCES

    def test_has_coreference_sentences(self):
        assert "coref_animal" in ATTENTION_SENTENCES
        assert "coref_street" in ATTENTION_SENTENCES

    def test_has_order_sentences(self):
        assert "order_matters" in ATTENTION_SENTENCES
        assert "order_reversed" in ATTENTION_SENTENCES

    def test_all_nonempty_strings(self):
        for key, val in ATTENTION_SENTENCES.items():
            assert isinstance(val, str) and len(val) > 0, key


class TestLMInputs:
    def test_has_causal_prompts(self):
        assert "causal_prompts" in LM_INPUTS
        assert len(LM_INPUTS["causal_prompts"]) == 3

    def test_has_masked_sentences(self):
        assert "masked_sentences" in LM_INPUTS
        assert len(LM_INPUTS["masked_sentences"]) == 3

    def test_masked_sentences_contain_mask_token(self):
        for s in LM_INPUTS["masked_sentences"]:
            assert "[MASK]" in s


class TestTransferLearningData:
    def test_has_train_and_test(self):
        assert "train" in TRANSFER_LEARNING_DATA
        assert "test" in TRANSFER_LEARNING_DATA

    def test_train_has_8_examples(self):
        assert len(TRANSFER_LEARNING_DATA["train"]) == 8

    def test_test_has_4_examples(self):
        assert len(TRANSFER_LEARNING_DATA["test"]) == 4

    def test_labels_are_binary(self):
        for split in ("train", "test"):
            for text, label in TRANSFER_LEARNING_DATA[split]:
                assert isinstance(text, str)
                assert label in (0, 1)

    def test_balanced_labels(self):
        for split in ("train", "test"):
            labels = [label for _, label in TRANSFER_LEARNING_DATA[split]]
            assert labels.count(0) == labels.count(1)


class TestModelRegistry:
    def test_has_three_families(self):
        assert set(MODEL_REGISTRY.keys()) == {
            "encoder_only",
            "decoder_only",
            "encoder_decoder",
        }

    def test_each_family_has_required_keys(self):
        required = {"primary", "small", "family", "objective", "attention"}
        for family, info in MODEL_REGISTRY.items():
            assert required.issubset(set(info.keys())), f"{family} missing keys"

    def test_primary_models(self):
        assert MODEL_REGISTRY["encoder_only"]["primary"] == "bert-base-uncased"
        assert MODEL_REGISTRY["decoder_only"]["primary"] == "gpt2"
        assert MODEL_REGISTRY["encoder_decoder"]["primary"] == "t5-small"
