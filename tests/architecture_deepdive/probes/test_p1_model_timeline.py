"""Tests for src/architecture_deepdive/probes/p1_model_timeline.py."""

from src.architecture_deepdive.probes.p1_model_timeline import (
    TIMELINE,
    TransformerMilestone,
    run_experiment,
)


class TestTimeline:
    def test_has_11_entries(self):
        assert len(TIMELINE) == 11

    def test_all_are_milestones(self):
        for m in TIMELINE:
            assert isinstance(m, TransformerMilestone)

    def test_required_fields_nonempty(self):
        for m in TIMELINE:
            assert m.name
            assert m.date
            assert m.params
            assert m.family
            assert m.architecture
            assert m.key_innovation

    def test_valid_families(self):
        valid = {"encoder-only", "decoder-only", "encoder-decoder"}
        for m in TIMELINE:
            assert m.family in valid, f"{m.name} has invalid family"

    def test_first_is_original_transformer(self):
        assert TIMELINE[0].name == "Transformer (Original)"
        assert TIMELINE[0].date == "June 2017"


class TestRunExperiment:
    def test_returns_dict(self):
        result = run_experiment()
        assert isinstance(result, dict)

    def test_has_task_key(self):
        result = run_experiment()
        assert result["task"] == "transformer_timeline"

    def test_has_timeline(self):
        result = run_experiment()
        assert "timeline" in result
        assert len(result["timeline"]) == 11

    def test_timeline_entries_have_keys(self):
        result = run_experiment()
        required = {"name", "date", "params", "family", "key_innovation", "hf_checkpoint"}
        for entry in result["timeline"]:
            assert required.issubset(set(entry.keys()))

    def test_has_family_distribution(self):
        result = run_experiment()
        dist = result["family_distribution"]
        assert isinstance(dist, dict)
        assert sum(dist.values()) == 11

    def test_decoder_only_dominates(self):
        result = run_experiment()
        dist = result["family_distribution"]
        assert dist["decoder-only"] > dist.get("encoder-only", 0)
        assert dist["decoder-only"] > dist.get("encoder-decoder", 0)

    def test_has_observation(self):
        result = run_experiment()
        assert "observation" in result
        assert isinstance(result["observation"], str)
        assert len(result["observation"]) > 0
