"""Tests for src/data.py — verify all task input structures."""

from src.pipeline_exploration.data import (
    FILL_MASK_INPUTS,
    IMAGE_CLASSIFICATION_INPUTS,
    NER_INPUTS,
    QA_INPUTS,
    SPEECH_RECOGNITION_INPUTS,
    SUMMARIZATION_INPUTS,
    TEXT_CLASSIFICATION_INPUTS,
    TEXT_GENERATION_INPUTS,
    TRANSLATION_INPUTS,
    ZERO_SHOT_INPUTS,
)


class TestTextClassificationInputs:
    def test_has_course_examples(self):
        assert "course_examples" in TEXT_CLASSIFICATION_INPUTS

    def test_course_examples_is_list(self):
        assert isinstance(TEXT_CLASSIFICATION_INPUTS["course_examples"], list)

    def test_course_examples_nonempty(self):
        assert len(TEXT_CLASSIFICATION_INPUTS["course_examples"]) >= 1

    def test_course_examples_are_strings(self):
        for s in TEXT_CLASSIFICATION_INPUTS["course_examples"]:
            assert isinstance(s, str)

    def test_has_edge_cases(self):
        assert "edge_cases" in TEXT_CLASSIFICATION_INPUTS
        assert isinstance(TEXT_CLASSIFICATION_INPUTS["edge_cases"], dict)

    def test_known_course_text(self):
        texts = TEXT_CLASSIFICATION_INPUTS["course_examples"]
        assert any("HuggingFace" in t for t in texts)


class TestZeroShotInputs:
    def test_has_course_examples(self):
        assert "course_examples" in ZERO_SHOT_INPUTS

    def test_course_examples_has_sequence_and_labels(self):
        ce = ZERO_SHOT_INPUTS["course_examples"]
        assert "sequence" in ce
        assert "candidate_labels" in ce

    def test_candidate_labels_is_list(self):
        assert isinstance(ZERO_SHOT_INPUTS["course_examples"]["candidate_labels"], list)

    def test_has_edge_cases(self):
        assert "edge_cases" in ZERO_SHOT_INPUTS

    def test_edge_cases_have_required_keys(self):
        for name, item in ZERO_SHOT_INPUTS["edge_cases"].items():
            assert "sequence" in item, f"Missing 'sequence' in edge case {name!r}"
            assert "candidate_labels" in item, f"Missing 'candidate_labels' in edge case {name!r}"


class TestTextGenerationInputs:
    def test_has_course_examples(self):
        assert "course_examples" in TEXT_GENERATION_INPUTS

    def test_course_examples_is_list_of_strings(self):
        for s in TEXT_GENERATION_INPUTS["course_examples"]:
            assert isinstance(s, str)

    def test_has_ablation(self):
        assert "ablation" in TEXT_GENERATION_INPUTS

    def test_ablation_has_temperatures(self):
        assert "temperatures" in TEXT_GENERATION_INPUTS["ablation"]
        temps = TEXT_GENERATION_INPUTS["ablation"]["temperatures"]
        assert set(temps) == {0.7, 1.0, 1.5}

    def test_ablation_has_models(self):
        assert "models" in TEXT_GENERATION_INPUTS["ablation"]
        models = TEXT_GENERATION_INPUTS["ablation"]["models"]
        assert "default" in models
        assert "alternative" in models


class TestFillMaskInputs:
    def test_has_course_examples(self):
        assert "course_examples" in FILL_MASK_INPUTS

    def test_distilroberta_uses_mask_token(self):
        text = FILL_MASK_INPUTS["course_examples"]["distilroberta"]
        assert "<mask>" in text

    def test_bert_uses_mask_token(self):
        text = FILL_MASK_INPUTS["course_examples"]["bert"]
        assert "[MASK]" in text

    def test_has_edge_cases(self):
        assert "edge_cases" in FILL_MASK_INPUTS
        for text in FILL_MASK_INPUTS["edge_cases"].values():
            assert "<mask>" in text


class TestNERInputs:
    def test_has_course_examples(self):
        assert "course_examples" in NER_INPUTS

    def test_course_examples_is_list(self):
        assert isinstance(NER_INPUTS["course_examples"], list)

    def test_course_example_mentions_sylvain(self):
        assert any("Sylvain" in s for s in NER_INPUTS["course_examples"])

    def test_has_edge_cases(self):
        assert "edge_cases" in NER_INPUTS


class TestQAInputs:
    def test_has_course_examples(self):
        assert "course_examples" in QA_INPUTS

    def test_course_example_has_question_and_context(self):
        ce = QA_INPUTS["course_examples"]
        assert "question" in ce
        assert "context" in ce

    def test_edge_cases_have_question_and_context(self):
        for name, item in QA_INPUTS["edge_cases"].items():
            assert "question" in item, f"Missing 'question' in {name!r}"
            assert "context" in item, f"Missing 'context' in {name!r}"


class TestSummarizationInputs:
    def test_has_course_examples(self):
        assert "course_examples" in SUMMARIZATION_INPUTS

    def test_course_examples_are_long_strings(self):
        for text in SUMMARIZATION_INPUTS["course_examples"]:
            assert len(text) > 100

    def test_has_ablation_max_lengths(self):
        assert "ablation" in SUMMARIZATION_INPUTS
        assert "max_lengths" in SUMMARIZATION_INPUTS["ablation"]
        assert set(SUMMARIZATION_INPUTS["ablation"]["max_lengths"]) == {50, 100, 200}


class TestTranslationInputs:
    def test_has_course_examples(self):
        assert "course_examples" in TRANSLATION_INPUTS

    def test_course_example_is_french(self):
        # "Ce cours" is French
        assert any("Ce cours" in s for s in TRANSLATION_INPUTS["course_examples"])

    def test_has_models(self):
        assert "models" in TRANSLATION_INPUTS
        assert "fr_en" in TRANSLATION_INPUTS["models"]

    def test_has_edge_cases(self):
        assert "edge_cases" in TRANSLATION_INPUTS


class TestImageClassificationInputs:
    def test_has_course_examples(self):
        assert "course_examples" in IMAGE_CLASSIFICATION_INPUTS

    def test_course_example_is_url(self):
        url = IMAGE_CLASSIFICATION_INPUTS["course_examples"][0]
        assert url.startswith("http")

    def test_has_edge_cases(self):
        assert "edge_cases" in IMAGE_CLASSIFICATION_INPUTS


class TestSpeechRecognitionInputs:
    def test_has_course_examples(self):
        assert "course_examples" in SPEECH_RECOGNITION_INPUTS

    def test_course_example_is_url(self):
        url = SPEECH_RECOGNITION_INPUTS["course_examples"][0]
        assert url.startswith("http")

    def test_has_models(self):
        assert "models" in SPEECH_RECOGNITION_INPUTS
        assert "large_gpu" in SPEECH_RECOGNITION_INPUTS["models"]
        assert "tiny_cpu" in SPEECH_RECOGNITION_INPUTS["models"]

    def test_has_edge_cases(self):
        assert "edge_cases" in SPEECH_RECOGNITION_INPUTS
