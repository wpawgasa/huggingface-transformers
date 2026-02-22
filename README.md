# HuggingFace Transformers — Experiments & Tutorials

A hands-on repository for exploring the [Hugging Face Transformers](https://huggingface.co/docs/transformers) library through structured experiments and reproducible notebooks. Covers text, vision, and audio modalities using pre-trained models and the `pipeline()` API.

---

## Experiments

### Transformers Pipeline Exploration

> **Reference**: [HuggingFace LLM Course — Chapter 1.3](https://huggingface.co/learn/llm-course/chapter1/3)

A systematic benchmark of all major `pipeline()` task types. The goal is to understand how the 3-stage pipeline abstraction (preprocessing → model forward → postprocessing) behaves across tasks, what the default models are, and how latency differs across modalities.

| # | Task | Pipeline | Default model |
|---|------|----------|---------------|
| 1 | Sentiment analysis | `text-classification` | DistilBERT (SST-2) |
| 2 | Zero-shot classification | `zero-shot-classification` | BART-large-MNLI |
| 3 | Text generation | `text-generation` | GPT-2 |
| 4 | Mask filling | `fill-mask` | DistilRoBERTa |
| 5 | Named entity recognition | `ner` | BERT-large (CoNLL-03) |
| 6 | Question answering | `question-answering` | DistilBERT (SQuAD) |
| 7 | Summarization | `summarization` | DistilBART-CNN |
| 8 | Translation | `translation` | Helsinki-NLP/opus-mt |
| 9 | Image classification | `image-classification` | ViT-base-patch16-224 |
| 10 | Speech recognition | `automatic-speech-recognition` | Whisper-large-v3 |

Full design doc: [docs/transformers_pipeline_experiment.md](docs/transformers_pipeline_experiment.md)

---

## Project Structure

```
.
├── src/
│   ├── experiment_runner.py        # CLI orchestrator
│   ├── data.py                     # Test inputs for every task
│   ├── benchmarks.py               # Latency/throughput utilities
│   ├── evaluate.py                 # Output schema validation
│   └── pipelines/                  # One module per pipeline task
├── scripts/
│   └── run_experiment.sh           # One-command runner
├── docs/                           # Experiment design notes
├── notebooks/                      # Jupyter notebooks
├── datasets/                       # Dataset storage (gitignored)
├── models/                         # Model checkpoints (gitignored)
├── results/                        # Experiment outputs (gitignored)
└── pyproject.toml
```

---

## Getting Started

**Requirements**: Python 3.12+, ~10 GB disk for model cache, GPU optional.

```bash
# 1. Clone
git clone <repo-url>
cd huggingface-transformers

# 2. Install dependencies
uv sync
# or: pip install -e ".[dev]"

# 3. Run all pipeline experiments
python -m src.experiment_runner --device auto --tasks all --output results/outputs.json

# 4. Or use the shell script
bash scripts/run_experiment.sh
```

### DevContainer (recommended)

Open in VS Code with the Dev Containers extension. Two configurations are available:

- **CPU** — `Dockerfile` (default base image)
- **GPU** — `Dockerfile.cuda12` (CUDA 12.6, used by `devcontainer.json`)

The GPU container sets `HF_HUB_ENABLE_HF_TRANSFER=1` for faster model downloads.

---

## Running a Subset of Tasks

```bash
# Text tasks only
python -m src.experiment_runner --device cpu \
  --tasks text-classification,zero-shot,text-generation,fill-mask,ner,question-answering \
  --output results/text_only.json

# Single task
python -m src.experiment_runner --device auto --tasks summarization --output results/summarization.json
```

---

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `transformers>=4.46.0` | Core model library |
| `torch>=2.5.0` | PyTorch backend |
| `datasets>=3.2.0` | Dataset loading |
| `accelerate>=1.2.0` | Large model loading |
| `peft>=0.13.0` | LoRA / parameter-efficient fine-tuning |
| `bitsandbytes>=0.44.0` | 4-bit / 8-bit quantization |
| `optimum>=1.23.0` | ONNX export and optimized inference |
| `trl>=0.12.0` | RLHF and alignment training |
| `wandb`, `tensorboard` | Experiment tracking |

---

## Code Quality

```bash
# Format
black . && isort . && ruff format .

# Lint & type-check
ruff check . && mypy .

# Tests
pytest

# All at once (pre-commit)
pre-commit run --all-files
```

Pre-commit hooks (black, ruff, isort, mypy) run automatically on every commit.

---

## Resources

- [Hugging Face Transformers docs](https://huggingface.co/docs/transformers)
- [HuggingFace LLM Course](https://huggingface.co/learn/llm-course)
- [Model Hub](https://huggingface.co/models)
- [Datasets Hub](https://huggingface.co/datasets)
