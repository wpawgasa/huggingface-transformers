# Pipeline Experiment Implementation Summary

**Date**: 2026-02-22
**Branch**: `feature/pipeline-experiment-implementation`

## Overview

Full implementation of the Transformers Pipeline Exploration experiment, as specified in
[`docs/transformers_pipeline_experiment.md`](transformers_pipeline_experiment.md). All 10 pipeline
modules, benchmarking utilities, output validators, a CLI orchestrator, and a shell wrapper were
implemented from scratch in an empty `src/` directory.

## Changes

### New Files

| File | Purpose |
|------|---------|
| `src/__init__.py` | Package marker |
| `src/data.py` | Course + edge-case test inputs for all 10 tasks |
| `src/benchmarks.py` | `BenchmarkResult` dataclass + `benchmark_pipeline()` function |
| `src/evaluate.py` | Per-task `validate_output()` with full schema checks |
| `src/experiment_runner.py` | CLI orchestrator (`--device`, `--tasks`, `--output`) |
| `src/pipelines/__init__.py` | Pipeline subpackage marker |
| `src/pipelines/text_classification.py` | Experiment 1 — sentiment analysis |
| `src/pipelines/zero_shot.py` | Experiment 2 — zero-shot classification |
| `src/pipelines/text_generation.py` | Experiment 3 — text generation |
| `src/pipelines/fill_mask.py` | Experiment 4 — mask filling |
| `src/pipelines/ner.py` | Experiment 5 — named entity recognition |
| `src/pipelines/question_answering.py` | Experiment 6 — extractive QA |
| `src/pipelines/summarization.py` | Experiment 7 — summarization |
| `src/pipelines/translation.py` | Experiment 8 — translation (fr→en) |
| `src/pipelines/image_classification.py` | Experiment 9 — image classification |
| `src/pipelines/speech_recognition.py` | Experiment 10 — ASR |
| `scripts/run_experiment.sh` | One-command runner with GPU auto-detection |
| `tests/__init__.py` | Test package marker |
| `tests/pipelines/__init__.py` | Pipeline tests package marker |
| `tests/test_data.py` | Data module tests |
| `tests/test_benchmarks.py` | Benchmark utility tests |
| `tests/test_evaluate.py` | Validator tests |
| `tests/test_experiment_runner.py` | CLI orchestrator tests |
| `tests/pipelines/test_*.py` | Per-pipeline tests (10 files) |

## Technical Details

### Architecture

Each pipeline module follows the same pattern:
```
load pipeline → run course example → run custom tests → validate output schema → benchmark
```

### `src/benchmarks.py`

Measures three metrics by wrapping pipeline calls:
- **cold_start_ms**: time to load the model from disk into memory
- **warm_latency_ms**: mean of N timed inference calls after a warm-up run
- **throughput_samples_per_sec**: N_inputs × 1000 / warm_latency_ms

The `pipe_factory` callable pattern allows zero-shot and other tasks to wrap their
pipelines with fixed keyword arguments before passing to `benchmark_pipeline`.

### `src/evaluate.py`

Dispatches to per-task validators that check:
- Required output keys are present
- Score values are in [0, 1]
- Non-empty strings where required
- NER accepts both `entity_group` (grouped) and `entity` (ungrouped) formats
- Zero-shot scores sum ≈ 1.0 (±0.01 tolerance)

### Ablation Studies Implemented

| Study | Module | What varies |
|-------|--------|-------------|
| Temperature sweep | `text_generation.py` | `temperature` ∈ {0.7, 1.0, 1.5} |
| Default vs custom model | `text_generation.py` | GPT-2 vs SmolLM2-360M |
| Label count scaling | `zero_shot.py` | `n_labels` ∈ {2, 3, 5, 10} |
| Mask token / model | `fill_mask.py` | DistilRoBERTa (`<mask>`) vs BERT-base-cased (`[MASK]`) |
| `grouped_entities` | `ner.py` | `True` vs `False` |
| Summary length | `summarization.py` | `max_length` ∈ {50, 100, 200} |
| CPU model substitution | `speech_recognition.py` | whisper-tiny (CPU) vs whisper-large-v3 (GPU) |

### Model Selection

| Task | Model |
|------|-------|
| Text classification | `distilbert-base-uncased-finetuned-sst-2-english` |
| Zero-shot | `facebook/bart-large-mnli` |
| Text generation | `openai-community/gpt2` |
| Fill-mask | `distilroberta-base` |
| NER | `dbmdz/bert-large-cased-finetuned-conll03-english` |
| QA | `distilbert-base-cased-distilled-squad` |
| Summarization | `sshleifer/distilbart-cnn-12-6` |
| Translation | `Helsinki-NLP/opus-mt-fr-en` |
| Image classification | `google/vit-base-patch16-224` |
| ASR (CPU) | `openai/whisper-tiny` |
| ASR (GPU) | `openai/whisper-large-v3` |

## Testing

- **Framework**: pytest with pytest-cov
- **Test count**: 236 tests
- **Coverage**: 99% overall (100% for all files except `zero_shot.py` at 90%, which has
  an inner closure that is intentionally not executed in the mocked test environment)
- All tests run without downloading models (HuggingFace pipeline is fully mocked)

### Coverage by module

| Module | Coverage |
|--------|---------|
| `src/__init__.py` | 100% |
| `src/benchmarks.py` | 100% |
| `src/data.py` | 100% |
| `src/evaluate.py` | 100% |
| `src/experiment_runner.py` | 100% |
| All 10 pipeline modules | 100% (except zero_shot at 90%) |

## Usage

```bash
# Run all 10 tasks (auto-detect GPU)
python -m src.experiment_runner --device auto --tasks all --output results/outputs.json

# Run subset
python -m src.experiment_runner --device cpu \
  --tasks text-classification,fill-mask,ner \
  --output results/text_only.json

# One-command via shell script
bash scripts/run_experiment.sh
bash scripts/run_experiment.sh --tasks text-classification --output results/test.json
```
