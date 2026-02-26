"""Tests for src/evaluate.py — output schema validators."""

import pytest

from src.pipeline_exploration.evaluate import validate_output

# ---------------------------------------------------------------------------
# validate_output dispatcher
# ---------------------------------------------------------------------------


class TestValidateOutputDispatcher:
    def test_unknown_task_raises(self):
        with pytest.raises(ValueError, match="Unknown task"):
            validate_output("unsupported-task", {"label": "X", "score": 0.9})

    def test_dispatches_text_classification(self):
        # Should not raise
        validate_output("text-classification", {"label": "POSITIVE", "score": 0.99})

    def test_dispatches_zero_shot(self):
        validate_output(
            "zero-shot-classification",
            {"sequence": "text", "labels": ["a"], "scores": [1.0]},
        )

    def test_dispatches_text_generation(self):
        validate_output("text-generation", {"generated_text": "hello world"})

    def test_dispatches_fill_mask(self):
        validate_output(
            "fill-mask",
            {"sequence": "test", "score": 0.8, "token": 42, "token_str": " model"},
        )

    def test_dispatches_ner(self):
        validate_output(
            "ner",
            {"entity_group": "PER", "word": "Alice", "score": 0.99, "start": 0, "end": 5},
        )

    def test_dispatches_question_answering(self):
        validate_output(
            "question-answering",
            {"answer": "Brooklyn", "score": 0.98, "start": 40, "end": 48},
        )

    def test_dispatches_summarization(self):
        validate_output("summarization", {"summary_text": "A summary."})

    def test_dispatches_translation(self):
        validate_output("translation", {"translation_text": "Translated text."})

    def test_dispatches_image_classification(self):
        validate_output("image-classification", {"label": "cat", "score": 0.95})

    def test_dispatches_asr(self):
        validate_output("automatic-speech-recognition", {"text": "Hello world."})


# ---------------------------------------------------------------------------
# text-classification
# ---------------------------------------------------------------------------


class TestTextClassification:
    TASK = "text-classification"

    def test_valid_positive(self):
        validate_output(self.TASK, {"label": "POSITIVE", "score": 0.9998})

    def test_valid_negative(self):
        validate_output(self.TASK, {"label": "NEGATIVE", "score": 0.0})

    def test_score_boundary_zero(self):
        validate_output(self.TASK, {"label": "POSITIVE", "score": 0.0})

    def test_score_boundary_one(self):
        validate_output(self.TASK, {"label": "POSITIVE", "score": 1.0})

    def test_missing_label_raises(self):
        with pytest.raises(ValueError, match="Missing keys"):
            validate_output(self.TASK, {"score": 0.9})

    def test_missing_score_raises(self):
        with pytest.raises(ValueError, match="Missing keys"):
            validate_output(self.TASK, {"label": "POSITIVE"})

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            validate_output(self.TASK, {"label": "POSITIVE", "score": 1.01})

    def test_score_below_zero_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            validate_output(self.TASK, {"label": "POSITIVE", "score": -0.01})

    def test_empty_label_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_output(self.TASK, {"label": "", "score": 0.9})


# ---------------------------------------------------------------------------
# zero-shot-classification
# ---------------------------------------------------------------------------


class TestZeroShot:
    TASK = "zero-shot-classification"

    def _valid(self):
        return {"sequence": "hello", "labels": ["a", "b"], "scores": [0.7, 0.3]}

    def test_valid(self):
        validate_output(self.TASK, self._valid())

    def test_valid_three_labels(self):
        validate_output(
            self.TASK,
            {"sequence": "x", "labels": ["a", "b", "c"], "scores": [0.5, 0.3, 0.2]},
        )

    def test_missing_sequence_raises(self):
        with pytest.raises(ValueError, match="Missing keys"):
            validate_output(self.TASK, {"labels": ["a"], "scores": [1.0]})

    def test_missing_labels_raises(self):
        with pytest.raises(ValueError, match="Missing keys"):
            validate_output(self.TASK, {"sequence": "x", "scores": [1.0]})

    def test_missing_scores_raises(self):
        with pytest.raises(ValueError, match="Missing keys"):
            validate_output(self.TASK, {"sequence": "x", "labels": ["a"]})

    def test_scores_not_summing_raises(self):
        with pytest.raises(ValueError, match="sum"):
            validate_output(
                self.TASK,
                {"sequence": "x", "labels": ["a", "b"], "scores": [0.5, 0.1]},
            )

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            validate_output(
                self.TASK,
                {"sequence": "x", "labels": ["a", "b"], "scores": [1.0]},
            )

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            validate_output(
                self.TASK,
                {"sequence": "x", "labels": ["a", "b"], "scores": [1.5, -0.5]},
            )

    def test_sequence_not_string_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            validate_output(
                self.TASK,
                {"sequence": 123, "labels": ["a"], "scores": [1.0]},
            )

    def test_labels_not_list_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            validate_output(
                self.TASK,
                {"sequence": "x", "labels": "a", "scores": [1.0]},
            )

    def test_scores_not_list_raises(self):
        with pytest.raises(ValueError, match="non-empty list"):
            validate_output(
                self.TASK,
                {"sequence": "x", "labels": ["a"], "scores": 1.0},
            )


# ---------------------------------------------------------------------------
# text-generation
# ---------------------------------------------------------------------------


class TestTextGeneration:
    TASK = "text-generation"

    def test_valid(self):
        validate_output(self.TASK, {"generated_text": "In this course we learn"})

    def test_empty_generated_text_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_output(self.TASK, {"generated_text": ""})

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_output(self.TASK, {"generated_text": "   "})

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="Missing keys"):
            validate_output(self.TASK, {"text": "hello"})


# ---------------------------------------------------------------------------
# fill-mask
# ---------------------------------------------------------------------------


class TestFillMask:
    TASK = "fill-mask"

    def _valid(self):
        return {"sequence": "This is a model.", "score": 0.85, "token": 2235, "token_str": " model"}

    def test_valid(self):
        validate_output(self.TASK, self._valid())

    def test_missing_sequence_raises(self):
        with pytest.raises(ValueError, match="Missing keys"):
            validate_output(self.TASK, {"score": 0.5, "token": 1, "token_str": "x"})

    def test_missing_token_raises(self):
        with pytest.raises(ValueError, match="Missing keys"):
            validate_output(self.TASK, {"sequence": "x", "score": 0.5, "token_str": "x"})

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            validate_output(
                self.TASK, {"sequence": "x", "score": 1.5, "token": 1, "token_str": "x"}
            )

    def test_token_not_int_raises(self):
        with pytest.raises(ValueError, match="must be an int"):
            validate_output(
                self.TASK, {"sequence": "x", "score": 0.5, "token": "abc", "token_str": "x"}
            )

    def test_sequence_not_string_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            validate_output(self.TASK, {"sequence": 42, "score": 0.5, "token": 1, "token_str": "x"})

    def test_token_str_not_string_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            validate_output(self.TASK, {"sequence": "x", "score": 0.5, "token": 1, "token_str": 99})


# ---------------------------------------------------------------------------
# ner
# ---------------------------------------------------------------------------


class TestNER:
    TASK = "ner"

    def _valid_grouped(self):
        return {"entity_group": "PER", "word": "Sylvain", "score": 0.99, "start": 11, "end": 18}

    def _valid_ungrouped(self):
        return {"entity": "B-PER", "word": "Sylvain", "score": 0.99, "start": 11, "end": 18}

    def test_valid_grouped(self):
        validate_output(self.TASK, self._valid_grouped())

    def test_valid_ungrouped(self):
        validate_output(self.TASK, self._valid_ungrouped())

    def test_no_entity_key_raises(self):
        with pytest.raises(ValueError, match="entity_group.*entity"):
            validate_output(self.TASK, {"word": "x", "score": 0.9, "start": 0, "end": 1})

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            validate_output(
                self.TASK, {"entity_group": "PER", "word": "x", "score": 1.5, "start": 0, "end": 1}
            )

    def test_invalid_span_raises(self):
        with pytest.raises(ValueError, match="Invalid span"):
            validate_output(
                self.TASK, {"entity_group": "PER", "word": "x", "score": 0.9, "start": 5, "end": 3}
            )

    def test_missing_word_raises(self):
        with pytest.raises(ValueError, match="Missing keys"):
            validate_output(self.TASK, {"entity_group": "PER", "score": 0.9, "start": 0, "end": 1})

    def test_empty_entity_group_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_output(
                self.TASK, {"entity_group": "", "word": "x", "score": 0.9, "start": 0, "end": 1}
            )

    def test_word_not_string_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            validate_output(
                self.TASK, {"entity_group": "PER", "word": 42, "score": 0.9, "start": 0, "end": 1}
            )

    def test_start_end_not_int_raises(self):
        with pytest.raises(ValueError, match="must be ints"):
            validate_output(
                self.TASK,
                {"entity_group": "PER", "word": "x", "score": 0.9, "start": "a", "end": "b"},
            )

    def test_start_negative_raises(self):
        with pytest.raises(ValueError, match="Invalid span"):
            validate_output(
                self.TASK, {"entity_group": "PER", "word": "x", "score": 0.9, "start": -1, "end": 1}
            )


# ---------------------------------------------------------------------------
# question-answering
# ---------------------------------------------------------------------------


class TestQuestionAnswering:
    TASK = "question-answering"

    def test_valid(self):
        validate_output(self.TASK, {"answer": "Brooklyn", "score": 0.98, "start": 40, "end": 48})

    def test_empty_answer_is_valid(self):
        # Model may return empty answer for unanswerable questions
        validate_output(self.TASK, {"answer": "", "score": 0.01, "start": 0, "end": 0})

    def test_missing_answer_raises(self):
        with pytest.raises(ValueError, match="Missing keys"):
            validate_output(self.TASK, {"score": 0.9, "start": 0, "end": 5})

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            validate_output(self.TASK, {"answer": "x", "score": -0.1, "start": 0, "end": 1})

    def test_start_end_not_int_raises(self):
        with pytest.raises(ValueError, match="must be ints"):
            validate_output(self.TASK, {"answer": "x", "score": 0.9, "start": "a", "end": "b"})

    def test_answer_not_string_raises(self):
        with pytest.raises(ValueError, match="must be a string"):
            validate_output(self.TASK, {"answer": 42, "score": 0.9, "start": 0, "end": 1})


# ---------------------------------------------------------------------------
# summarization
# ---------------------------------------------------------------------------


class TestSummarization:
    TASK = "summarization"

    def test_valid(self):
        validate_output(self.TASK, {"summary_text": "A short summary of the article."})

    def test_empty_summary_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_output(self.TASK, {"summary_text": ""})

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_output(self.TASK, {"summary_text": "  \n  "})

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="Missing keys"):
            validate_output(self.TASK, {"text": "something"})


# ---------------------------------------------------------------------------
# translation
# ---------------------------------------------------------------------------


class TestTranslation:
    TASK = "translation"

    def test_valid(self):
        validate_output(self.TASK, {"translation_text": "This course is produced by Hugging Face."})

    def test_empty_translation_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_output(self.TASK, {"translation_text": ""})

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="Missing keys"):
            validate_output(self.TASK, {"text": "translated"})


# ---------------------------------------------------------------------------
# image-classification
# ---------------------------------------------------------------------------


class TestImageClassification:
    TASK = "image-classification"

    def test_valid(self):
        validate_output(self.TASK, {"label": "Egyptian cat", "score": 0.9879})

    def test_empty_label_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_output(self.TASK, {"label": "", "score": 0.9})

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            validate_output(self.TASK, {"label": "cat", "score": 1.5})

    def test_missing_keys_raises(self):
        with pytest.raises(ValueError, match="Missing keys"):
            validate_output(self.TASK, {"label": "cat"})


# ---------------------------------------------------------------------------
# automatic-speech-recognition
# ---------------------------------------------------------------------------


class TestASR:
    TASK = "automatic-speech-recognition"

    def test_valid(self):
        validate_output(self.TASK, {"text": "My fellow Americans."})

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_output(self.TASK, {"text": ""})

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_output(self.TASK, {"text": "   "})

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="Missing keys"):
            validate_output(self.TASK, {"transcript": "hello"})
