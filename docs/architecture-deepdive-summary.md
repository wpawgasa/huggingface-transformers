# Architecture Deep Dive — Implementation Summary

**Date**: 2026-02-26
**Branch**: `feature/reorganize-src-and-architecture-experiment`

## Overview

Full implementation of the Transformer Architecture Deep Dive experiment (Ch1.4),
covering 6 probes that empirically verify the theoretical claims from the HuggingFace
LLM Course Chapter 1.4 about attention mechanisms, architecture families, language
modeling objectives, and transfer learning.

## Changes

### New Files — Source (`src/architecture_deepdive/`)

| File | Purpose |
|------|---------|
| `__init__.py` | Package docstring |
| `__main__.py` | `python -m src.architecture_deepdive` entry point |
| `data.py` | Shared test sentences, LM inputs, transfer learning data, model registry |
| `experiment_runner.py` | CLI orchestrator with `--device`, `--probes`, `--output` args |
| `utils/__init__.py` | Utils package exports |
| `utils/model_inspector.py` | `ModelAnatomy` dataclass + `inspect_model()` for parameter/layer extraction |
| `utils/attention_tools.py` | `extract_attention_weights()`, `get_attention_to_token()`, mask comparison |
| `utils/plotting.py` | Matplotlib/seaborn helpers for attention heatmaps, parameter charts, learning curves |
| `probes/__init__.py` | Probe package exports |
| `probes/p1_model_timeline.py` | Transformer history timeline (data-only, no model downloads) |
| `probes/p2_language_modeling.py` | Causal vs Masked LM comparison (GPT-2 vs BERT) |
| `probes/p3_transfer_learning.py` | Pretrained vs from-scratch training on tiny sentiment dataset |
| `probes/p4_model_anatomy.py` | Architecture inspection across BERT, GPT-2, T5 |
| `probes/p5_attention_viz.py` | Attention weight extraction and visualization |
| `probes/p6_arch_comparison.py` | Full encoder vs decoder vs encoder-decoder comparison |

### New Files — Tests (`tests/architecture_deepdive/`)

| File | Tests |
|------|-------|
| `test_data.py` | 15 tests — validates all data constants |
| `test_experiment_runner.py` | 24 tests — CLI parsing, probe execution, error handling |
| `utils/test_model_inspector.py` | 15 tests — ModelAnatomy, inspect_model(), format_param_count() |
| `utils/test_attention_tools.py` | 14 tests — mask generation, attention extraction, token analysis |
| `utils/test_plotting.py` | 10 tests — all 4 plot functions (mocked matplotlib) |
| `probes/test_p1_model_timeline.py` | 12 tests — timeline data + run_experiment() |
| `probes/test_p2_language_modeling.py` | 13 tests — CLM probe, MLM probe, full experiment |
| `probes/test_p3_transfer_learning.py` | 14 tests — SimpleClassifier, prepare_data, train loop |
| `probes/test_p4_model_anatomy.py` | 9 tests — inspect + plot integration |
| `probes/test_p5_attention_viz.py` | 9 tests — attention extraction, plotting, coreference |
| `probes/test_p6_arch_comparison.py` | 15 tests — all 3 architecture probes + synthesis |

### New Files — Scripts

| File | Purpose |
|------|---------|
| `scripts/run_architecture_experiment.sh` | One-command runner with dep install + GPU detection |

## Testing

- **168 new tests**, all passing
- **99%+ coverage** on `src/architecture_deepdive/` (only `__main__.py` and 1 line in `plotting.py` uncovered)
- All models fully mocked — no internet or GPU required for tests
- Full suite: 405 tests (237 existing + 168 new), all passing

## Usage

```bash
# Run all 6 probes
python -m src.architecture_deepdive.experiment_runner --device auto --probes all

# Run specific probes
python -m src.architecture_deepdive.experiment_runner --device cpu --probes p1_timeline,p4_model_anatomy

# One-command via shell script
bash scripts/run_architecture_experiment.sh
```

## Architecture Decisions

- Follows the same patterns as `pipeline_exploration`: registry-based runner, mocked tests, per-module `run_experiment()` exports
- Probes 2-6 accept `device` parameter; Probe 1 is data-only
- All figures saved to `results/figures/` using matplotlib Agg (non-interactive) backend
- Transfer learning probe uses `SimpleClassifier(nn.Module)` wrapper around base model
- Config field name differences handled via `getattr` chains (e.g., `num_hidden_layers` vs `n_layer`)
