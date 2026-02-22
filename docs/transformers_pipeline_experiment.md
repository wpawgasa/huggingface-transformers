# Experiment: Exploring 🤗 Transformers Pipelines

**Source**: [HuggingFace LLM Course — Chapter 1.3: Transformers, what can they do?](https://huggingface.co/learn/llm-course/chapter1/3)
**Experiment Design**: AI Researcher Reproduction Framework
**Scope**: Systematic exploration of all `pipeline()` task types across text, vision, and audio modalities
**Success Criterion**: All pipelines execute correctly, outputs match expected schema, and performance is benchmarked

---

## 1. Problem & Motivation

**Research Question**: How does the HuggingFace `pipeline()` abstraction unify inference across diverse NLP/CV/Audio tasks, and what are the practical performance characteristics (latency, output quality) of default vs. custom models?

**Context**: The `pipeline()` API is the highest-level entry point in 🤗 Transformers. It encapsulates a 3-stage process — **preprocessing → model forward pass → postprocessing** — hiding tokenization, tensor handling, and decoding behind a single callable. Understanding its behavior across tasks is foundational before diving into fine-tuning or custom architectures.

**Prior Work Limitations**:
- Most tutorials run pipelines in isolation without systematic comparison
- Latency and output schema differences across tasks are rarely documented
- Default model selection is opaque to beginners

**Gap Addressed**: This experiment provides a structured, reproducible benchmark of all major pipeline tasks taught in the course, with consistent evaluation methodology.

---

## 2. Technical Approach

### 2.1 Core Concept: The `pipeline()` 3-Stage Architecture

```
Input (raw text/image/audio)
        │
        ▼
┌─────────────────┐
│  Preprocessing   │  ← Tokenizer / Feature Extractor / Processor
│  (AutoTokenizer) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Model Forward  │  ← AutoModel / AutoModelForXxx
│   Pass (logits)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Postprocessing   │  ← Decode tokens, apply softmax, format output
│ (task-specific)  │
└─────────────────┘
         │
         ▼
Output (human-readable dict/list)
```

### 2.2 Experiment Taxonomy

The course covers **9 pipeline tasks** across **4 modalities**:

| # | Task | Modality | Pipeline Name | Default Model Family |
|---|------|----------|---------------|---------------------|
| 1 | Sentiment Analysis | Text | `text-classification` | DistilBERT (SST-2) |
| 2 | Zero-shot Classification | Text | `zero-shot-classification` | BART-large-MNLI |
| 3 | Text Generation | Text | `text-generation` | GPT-2 |
| 4 | Mask Filling | Text | `fill-mask` | DistilRoBERTa |
| 5 | Named Entity Recognition | Text | `ner` | BERT-base (CoNLL-03) |
| 6 | Question Answering | Text | `question-answering` | DistilBERT (SQuAD) |
| 7 | Summarization | Text | `summarization` | DistilBART-CNN |
| 8 | Translation | Text | `translation` | Helsinki-NLP/opus-mt |
| 9 | Image Classification | Vision | `image-classification` | ViT-base-patch16-224 |
| 10 | Speech Recognition | Audio | `automatic-speech-recognition` | Whisper-large-v3 |

---

## 3. Experimental Setup

### 3.1 Environment

```yaml
# config.yaml
environment:
  python: ">=3.9"
  framework: "transformers>=4.40.0"
  backend: "torch>=2.0"
  device: "auto"  # cuda if available, else cpu

dependencies:
  core:
    - transformers
    - torch
    - sentencepiece      # for translation models
    - sacremoses         # for Helsinki-NLP models
    - protobuf           # for some tokenizers
  optional:
    - accelerate         # for large model loading
    - soundfile          # for audio pipelines
    - librosa            # for audio resampling
    - Pillow             # for image pipelines
```

### 3.2 Project Structure

```
transformers_pipeline_experiment/
├── README.md                   # This document
├── requirements.txt            # Pinned dependencies
├── config.yaml                 # All hyperparameters and settings
├── src/
│   ├── __init__.py
│   ├── experiment_runner.py    # Main orchestrator
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── text_classification.py
│   │   ├── zero_shot.py
│   │   ├── text_generation.py
│   │   ├── fill_mask.py
│   │   ├── ner.py
│   │   ├── question_answering.py
│   │   ├── summarization.py
│   │   ├── translation.py
│   │   ├── image_classification.py
│   │   └── speech_recognition.py
│   ├── benchmarks.py           # Latency and throughput measurement
│   ├── data.py                 # Test inputs for each task
│   └── evaluate.py             # Output validation and comparison
├── results/
│   ├── outputs.json            # Raw pipeline outputs
│   ├── benchmarks.csv          # Latency metrics
│   └── comparison.md           # Default vs custom model comparison
└── scripts/
    └── run_experiment.sh        # One-command execution
```

---

## 4. Implementation Guide

### 4.1 Test Data Module — `src/data.py`

```python
"""
Test inputs for each pipeline task.
Sourced from HuggingFace LLM Course Chapter 1.3 examples
with additional edge cases for robustness testing.
"""

# ──────────────────────────────────────────────
# Text Classification (Sentiment Analysis)
# ──────────────────────────────────────────────
SENTIMENT_INPUTS = {
    "course_examples": [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ],
    "edge_cases": [
        "",                                    # empty string
        "This is fine.",                       # neutral / ambiguous
        "Not bad, not great either.",          # double negation
        "สวัสดีครับ ผมชอบคอร์สนี้มาก",           # Thai (out-of-distribution)
    ],
}

# ──────────────────────────────────────────────
# Zero-shot Classification
# ──────────────────────────────────────────────
ZERO_SHOT_INPUTS = {
    "course_example": {
        "sequence": "This is a course about the Transformers library",
        "candidate_labels": ["education", "politics", "business"],
    },
    "custom_tests": [
        {
            "sequence": "The stock market crashed after the Fed raised interest rates",
            "candidate_labels": ["finance", "technology", "sports", "politics"],
        },
        {
            "sequence": "The patient was diagnosed with pneumonia and prescribed antibiotics",
            "candidate_labels": ["medical", "legal", "education", "cooking"],
        },
    ],
}

# ──────────────────────────────────────────────
# Text Generation
# ──────────────────────────────────────────────
TEXT_GEN_INPUTS = {
    "course_example": "In this course, we will teach you how to",
    "custom_prompts": [
        "The future of artificial intelligence is",
        "Once upon a time in a galaxy far away,",
        "def fibonacci(n):",                    # code completion test
    ],
    "params": {
        "max_length": 50,
        "num_return_sequences": 2,
        "temperature": [0.7, 1.0, 1.5],        # ablation on randomness
        "do_sample": True,
    },
}

# ──────────────────────────────────────────────
# Fill-Mask
# ──────────────────────────────────────────────
FILL_MASK_INPUTS = {
    "course_example": "This course will teach you all about <mask> models.",
    "custom_tests": [
        "The capital of France is <mask>.",
        "Machine learning requires large amounts of <mask>.",
        "<mask> is the best programming language.",  # mask at start
    ],
    "params": {"top_k": 5},
}

# ──────────────────────────────────────────────
# Named Entity Recognition
# ──────────────────────────────────────────────
NER_INPUTS = {
    "course_example": "My name is Sylvain and I work at Hugging Face in Brooklyn.",
    "custom_tests": [
        "Elon Musk founded SpaceX in Hawthorne, California.",
        "The United Nations headquarters is in New York City.",
        "Dr. Sarah Chen published her findings in Nature last Tuesday.",
    ],
}

# ──────────────────────────────────────────────
# Question Answering
# ──────────────────────────────────────────────
QA_INPUTS = {
    "course_example": {
        "question": "Where do I work?",
        "context": "My name is Sylvain and I work at Hugging Face in Brooklyn",
    },
    "custom_tests": [
        {
            "question": "What is the transformer architecture based on?",
            "context": (
                "The Transformer architecture was introduced in 2017 by Vaswani et al. "
                "It is based on the self-attention mechanism, which allows the model to "
                "weigh the importance of different parts of the input sequence."
            ),
        },
        {
            "question": "When was the company founded?",
            "context": "Anthropic was founded in 2021 by Dario Amodei and Daniela Amodei.",
        },
    ],
}

# ──────────────────────────────────────────────
# Summarization
# ──────────────────────────────────────────────
SUMMARIZATION_INPUTS = {
    "course_example": """
    America has changed dramatically during recent years. Not only has the number of
    graduates in traditional engineering disciplines such as mechanical, civil,
    electrical, chemical, and aeronautical engineering declined, but in most of
    the premier American universities engineering curricula now concentrate on
    and encourage largely the study of engineering science. As a result, there
    are declining offerings in engineering subjects dealing with infrastructure,
    the environment, and related issues, and greater concentration on high
    technology subjects, largely supporting increasingly complex scientific
    developments. While the latter is important, it should not be at the expense
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other
    industrial countries in Europe and Asia, continue to encourage and advance
    the teaching of engineering. Both China and India, respectively, graduate
    six and eight times as many traditional engineers as does the United States.
    Other industrial countries at minimum maintain their output, while America
    suffers an increasingly serious decline in the number of engineering graduates
    and a lack of well-educated engineers.
    """,
    "params": {
        "max_length": 150,
        "min_length": 30,
    },
}

# ──────────────────────────────────────────────
# Translation
# ──────────────────────────────────────────────
TRANSLATION_INPUTS = {
    "course_example": {
        "text": "Ce cours est produit par Hugging Face.",
        "model": "Helsinki-NLP/opus-mt-fr-en",
        "src_lang": "fr",
        "tgt_lang": "en",
    },
    "custom_tests": [
        {
            "text": "Ich liebe maschinelles Lernen.",
            "model": "Helsinki-NLP/opus-mt-de-en",
            "src_lang": "de",
            "tgt_lang": "en",
        },
    ],
}

# ──────────────────────────────────────────────
# Image Classification
# ──────────────────────────────────────────────
IMAGE_INPUTS = {
    "course_example": {
        "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
        "model": "google/vit-base-patch16-224",
    },
}

# ──────────────────────────────────────────────
# Automatic Speech Recognition
# ──────────────────────────────────────────────
ASR_INPUTS = {
    "course_example": {
        "url": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
        "model": "openai/whisper-large-v3",
    },
}
```

### 4.2 Benchmark Utility — `src/benchmarks.py`

```python
"""
Latency and throughput benchmarking for pipeline experiments.
Measures: cold start (model load), warm inference, batch throughput.
"""

import time
import json
import statistics
from dataclasses import dataclass, asdict
from typing import Callable, Any


@dataclass
class BenchmarkResult:
    task: str
    model: str
    cold_start_ms: float          # Time to load pipeline (first call)
    warm_inference_ms: float      # Average inference time (after warmup)
    warm_inference_std_ms: float  # Std dev of inference time
    throughput_samples_sec: float # For batch inputs
    num_warmup_runs: int
    num_timed_runs: int
    device: str


def benchmark_pipeline(
    pipeline_fn: Callable,
    input_data: Any,
    task: str,
    model: str,
    num_warmup: int = 3,
    num_runs: int = 10,
    device: str = "cpu",
) -> BenchmarkResult:
    """
    Benchmark a pipeline with warmup and timed runs.

    Args:
        pipeline_fn: Instantiated pipeline callable
        input_data: Single input to pass to the pipeline
        task: Task name for logging
        model: Model name for logging
        num_warmup: Number of warmup iterations (not timed)
        num_runs: Number of timed iterations
        device: Device string ("cpu" or "cuda")

    Returns:
        BenchmarkResult with timing statistics
    """
    # Warmup
    for _ in range(num_warmup):
        _ = pipeline_fn(input_data)

    # Timed runs
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = pipeline_fn(input_data)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    avg_ms = statistics.mean(latencies)
    std_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    return BenchmarkResult(
        task=task,
        model=model,
        cold_start_ms=0.0,   # Measured separately at pipeline creation
        warm_inference_ms=round(avg_ms, 2),
        warm_inference_std_ms=round(std_ms, 2),
        throughput_samples_sec=round(1000.0 / avg_ms, 2) if avg_ms > 0 else 0.0,
        num_warmup_runs=num_warmup,
        num_timed_runs=num_runs,
        device=device,
    )


def save_benchmarks(results: list[BenchmarkResult], path: str = "results/benchmarks.json"):
    """Save benchmark results to JSON."""
    with open(path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
```

### 4.3 Individual Pipeline Experiments

Each module follows a consistent pattern: **load → run course example → run custom tests → validate output → benchmark**.

#### 4.3.1 Text Classification — `src/pipelines/text_classification.py`

```python
"""
Experiment 1: Sentiment Analysis (Text Classification)
Reference: HF LLM Course Ch1.3 — "Working with pipelines"

Default model: distilbert-base-uncased-finetuned-sst-2-english
Architecture: DistilBERT (encoder-only)
Training data: SST-2 (Stanford Sentiment Treebank)
"""

from transformers import pipeline
from src.data import SENTIMENT_INPUTS
from src.benchmarks import benchmark_pipeline


def validate_output(result: dict) -> bool:
    """Validate output schema: {'label': str, 'score': float}"""
    return (
        isinstance(result, dict)
        and "label" in result
        and "score" in result
        and result["label"] in ("POSITIVE", "NEGATIVE")
        and 0.0 <= result["score"] <= 1.0
    )


def run_experiment(device: str = "cpu") -> dict:
    # ── 1. Load pipeline (measure cold start) ──
    import time
    start = time.perf_counter()
    classifier = pipeline("sentiment-analysis", device=device)
    cold_start_ms = (time.perf_counter() - start) * 1000

    results = {"task": "text-classification", "cold_start_ms": cold_start_ms}

    # ── 2. Course examples ──
    course_results = classifier(SENTIMENT_INPUTS["course_examples"])
    results["course_examples"] = {
        "inputs": SENTIMENT_INPUTS["course_examples"],
        "outputs": course_results,
        "all_valid": all(validate_output(r) for r in course_results),
    }

    # ── 3. Edge cases ──
    edge_results = []
    for text in SENTIMENT_INPUTS["edge_cases"]:
        try:
            out = classifier(text)
            edge_results.append({"input": text, "output": out, "error": None})
        except Exception as e:
            edge_results.append({"input": text, "output": None, "error": str(e)})
    results["edge_cases"] = edge_results

    # ── 4. Batch processing ──
    batch_result = classifier(SENTIMENT_INPUTS["course_examples"])
    results["batch_vs_single"] = "consistent" if len(batch_result) == 2 else "mismatch"

    # ── 5. Benchmark ──
    bench = benchmark_pipeline(
        pipeline_fn=classifier,
        input_data="I've been waiting for a HuggingFace course my whole life.",
        task="sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
    )
    bench.cold_start_ms = cold_start_ms
    results["benchmark"] = bench.__dict__

    return results
```

#### 4.3.2 Zero-Shot Classification — `src/pipelines/zero_shot.py`

```python
"""
Experiment 2: Zero-shot Classification
Reference: HF LLM Course Ch1.3 — "Zero-shot classification"

Default model: facebook/bart-large-mnli
Architecture: BART (encoder-decoder)
Key insight: No fine-tuning required — labels provided at inference time.
"""

from transformers import pipeline
from src.data import ZERO_SHOT_INPUTS


def validate_output(result: dict) -> bool:
    """Validate: {'sequence': str, 'labels': list, 'scores': list}"""
    return (
        "sequence" in result
        and "labels" in result
        and "scores" in result
        and len(result["labels"]) == len(result["scores"])
        and abs(sum(result["scores"]) - 1.0) < 0.01  # scores sum to ~1.0
    )


def run_experiment(device: str = "cpu") -> dict:
    classifier = pipeline("zero-shot-classification", device=device)
    results = {"task": "zero-shot-classification"}

    # ── Course example ──
    ex = ZERO_SHOT_INPUTS["course_example"]
    out = classifier(ex["sequence"], candidate_labels=ex["candidate_labels"])
    results["course_example"] = {
        "output": out,
        "valid": validate_output(out),
        "top_label": out["labels"][0],
        "top_score": out["scores"][0],
    }

    # ── Ablation: varying number of labels ──
    label_ablation = []
    for n_labels in [2, 3, 5, 10]:
        labels = ex["candidate_labels"][:min(n_labels, len(ex["candidate_labels"]))]
        if n_labels > len(ex["candidate_labels"]):
            labels += [f"label_{i}" for i in range(n_labels - len(labels))]
        out = classifier(ex["sequence"], candidate_labels=labels)
        label_ablation.append({
            "num_labels": n_labels,
            "top_label": out["labels"][0],
            "top_score": out["scores"][0],
        })
    results["label_count_ablation"] = label_ablation

    # ── Custom domain tests ──
    for test in ZERO_SHOT_INPUTS["custom_tests"]:
        out = classifier(test["sequence"], candidate_labels=test["candidate_labels"])
        results[f"custom_{test['candidate_labels'][0]}"] = {
            "input": test["sequence"],
            "top_label": out["labels"][0],
            "scores": dict(zip(out["labels"], out["scores"])),
        }

    return results
```

#### 4.3.3 Text Generation — `src/pipelines/text_generation.py`

```python
"""
Experiment 3: Text Generation
Reference: HF LLM Course Ch1.3 — "Text generation" + "Using any model from the Hub"

Default model: openai-community/gpt2
Custom model: HuggingFaceTB/SmolLM2-360M (from course)
Architecture: GPT-2 / SmolLM2 (decoder-only, autoregressive)

Key ablations:
  - Default vs custom model
  - Temperature sweep (creativity vs coherence)
  - num_return_sequences control
"""

from transformers import pipeline
from src.data import TEXT_GEN_INPUTS


def validate_output(result: list) -> bool:
    """Each result: {'generated_text': str}"""
    return all(
        isinstance(r, dict) and "generated_text" in r and len(r["generated_text"]) > 0
        for r in result
    )


def run_experiment(device: str = "cpu") -> dict:
    results = {"task": "text-generation"}

    # ── Default model (GPT-2) ──
    generator_default = pipeline("text-generation", device=device)
    out = generator_default(
        TEXT_GEN_INPUTS["course_example"],
        max_length=50,
        num_return_sequences=1,
    )
    results["default_model"] = {
        "model": "gpt2",
        "output": out,
        "valid": validate_output(out),
    }

    # ── Custom model from Hub (SmolLM2-360M, as in the course) ──
    generator_custom = pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M",
        device=device,
    )
    out = generator_custom(
        TEXT_GEN_INPUTS["course_example"],
        max_length=30,
        num_return_sequences=2,
    )
    results["custom_model"] = {
        "model": "HuggingFaceTB/SmolLM2-360M",
        "output": out,
        "valid": validate_output(out),
        "num_sequences_returned": len(out),
    }

    # ── Temperature ablation ──
    temp_ablation = []
    for temp in TEXT_GEN_INPUTS["params"]["temperature"]:
        out = generator_default(
            "The meaning of life is",
            max_length=40,
            temperature=temp,
            do_sample=True,
            num_return_sequences=1,
        )
        temp_ablation.append({
            "temperature": temp,
            "generated_text": out[0]["generated_text"],
        })
    results["temperature_ablation"] = temp_ablation

    return results
```

#### 4.3.4 Fill-Mask — `src/pipelines/fill_mask.py`

```python
"""
Experiment 4: Mask Filling
Reference: HF LLM Course Ch1.3 — "Mask filling"

Default model: distilroberta-base
Architecture: DistilRoBERTa (encoder-only, MLM objective)

Key investigation:
  - Mask token varies by model (<mask> for RoBERTa, [MASK] for BERT)
  - top_k controls number of predictions
"""

from transformers import pipeline
from src.data import FILL_MASK_INPUTS


def run_experiment(device: str = "cpu") -> dict:
    results = {"task": "fill-mask"}

    # ── Default model ──
    unmasker = pipeline("fill-mask", device=device)
    out = unmasker(
        FILL_MASK_INPUTS["course_example"],
        top_k=FILL_MASK_INPUTS["params"]["top_k"],
    )
    results["course_example"] = {
        "input": FILL_MASK_INPUTS["course_example"],
        "predictions": [
            {"token": r["token_str"], "score": r["score"]}
            for r in out
        ],
    }

    # ── Comparison: DistilRoBERTa vs BERT-base-cased ──
    # NOTE: BERT uses [MASK] token, not <mask>
    unmasker_bert = pipeline("fill-mask", model="bert-base-cased", device=device)
    bert_input = "This course will teach you all about [MASK] models."
    out_bert = unmasker_bert(bert_input, top_k=5)
    results["model_comparison"] = {
        "distilroberta": results["course_example"]["predictions"][:3],
        "bert_base_cased": [
            {"token": r["token_str"], "score": r["score"]}
            for r in out_bert[:3]
        ],
        "note": "Different mask tokens: <mask> vs [MASK]",
    }

    return results
```

#### 4.3.5 Named Entity Recognition — `src/pipelines/ner.py`

```python
"""
Experiment 5: Named Entity Recognition (NER)
Reference: HF LLM Course Ch1.3 — "Named entity recognition"

Default model: dbmdz/bert-large-cased-finetuned-conll03-english
Entity types: PER (person), ORG (organization), LOC (location), MISC

Key feature: grouped_entities=True merges sub-word tokens (e.g., "Hug" + "##ging")
"""

from transformers import pipeline
from src.data import NER_INPUTS


def run_experiment(device: str = "cpu") -> dict:
    results = {"task": "ner"}

    # ── With grouped_entities (as taught in course) ──
    ner_grouped = pipeline("ner", grouped_entities=True, device=device)
    out = ner_grouped(NER_INPUTS["course_example"])
    results["course_example_grouped"] = {
        "input": NER_INPUTS["course_example"],
        "entities": [
            {
                "word": e["word"],
                "entity_group": e["entity_group"],
                "score": round(e["score"], 4),
                "span": (e["start"], e["end"]),
            }
            for e in out
        ],
    }

    # ── Without grouped_entities (raw sub-word tokens) ──
    ner_raw = pipeline("ner", grouped_entities=False, device=device)
    out_raw = ner_raw(NER_INPUTS["course_example"])
    results["course_example_raw"] = {
        "num_raw_entities": len(out_raw),
        "num_grouped_entities": len(results["course_example_grouped"]["entities"]),
        "note": "Raw mode shows sub-word level predictions (e.g., S, ##yl, ##va, ##in)",
    }

    # ── Custom tests ──
    for i, text in enumerate(NER_INPUTS["custom_tests"]):
        out = ner_grouped(text)
        results[f"custom_{i}"] = {
            "input": text,
            "entities": [
                {"word": e["word"], "type": e["entity_group"], "score": round(e["score"], 4)}
                for e in out
            ],
        }

    return results
```

#### 4.3.6 Question Answering — `src/pipelines/question_answering.py`

```python
"""
Experiment 6: Extractive Question Answering
Reference: HF LLM Course Ch1.3 — "Question answering"

Default model: distilbert-base-cased-distilled-squad
Architecture: DistilBERT fine-tuned on SQuAD
Key: Extracts answer spans from context — does NOT generate new text.
"""

from transformers import pipeline
from src.data import QA_INPUTS


def run_experiment(device: str = "cpu") -> dict:
    results = {"task": "question-answering"}

    qa = pipeline("question-answering", device=device)

    # ── Course example ──
    ex = QA_INPUTS["course_example"]
    out = qa(question=ex["question"], context=ex["context"])
    results["course_example"] = {
        "question": ex["question"],
        "context": ex["context"],
        "answer": out["answer"],
        "score": round(out["score"], 4),
        "span": (out["start"], out["end"]),
        "answer_in_context": ex["context"][out["start"]:out["end"]] == out["answer"],
    }

    # ── Custom tests ──
    for i, test in enumerate(QA_INPUTS["custom_tests"]):
        out = qa(question=test["question"], context=test["context"])
        results[f"custom_{i}"] = {
            "question": test["question"],
            "answer": out["answer"],
            "score": round(out["score"], 4),
        }

    # ── Ablation: unanswerable question ──
    out = qa(
        question="What is the meaning of life?",
        context="My name is Sylvain and I work at Hugging Face in Brooklyn",
    )
    results["unanswerable"] = {
        "question": "What is the meaning of life?",
        "answer": out["answer"],
        "score": round(out["score"], 4),
        "note": "Extractive QA always returns a span — low score indicates uncertainty",
    }

    return results
```

#### 4.3.7 Summarization — `src/pipelines/summarization.py`

```python
"""
Experiment 7: Summarization
Reference: HF LLM Course Ch1.3 — "Summarization"

Default model: sshleifer/distilbart-cnn-12-6
Architecture: DistilBART (encoder-decoder)
"""

from transformers import pipeline
from src.data import SUMMARIZATION_INPUTS


def run_experiment(device: str = "cpu") -> dict:
    results = {"task": "summarization"}

    summarizer = pipeline("summarization", device=device)

    # ── Course example ──
    text = SUMMARIZATION_INPUTS["course_example"]
    params = SUMMARIZATION_INPUTS["params"]
    out = summarizer(text, max_length=params["max_length"], min_length=params["min_length"])

    results["course_example"] = {
        "input_length_chars": len(text.strip()),
        "output_length_chars": len(out[0]["summary_text"]),
        "compression_ratio": round(len(out[0]["summary_text"]) / len(text.strip()), 3),
        "summary": out[0]["summary_text"],
    }

    # ── Length ablation ──
    length_ablation = []
    for max_len in [50, 100, 200]:
        out = summarizer(text, max_length=max_len, min_length=20)
        length_ablation.append({
            "max_length": max_len,
            "actual_length": len(out[0]["summary_text"]),
            "summary_preview": out[0]["summary_text"][:80] + "...",
        })
    results["length_ablation"] = length_ablation

    return results
```

#### 4.3.8 Translation — `src/pipelines/translation.py`

```python
"""
Experiment 8: Translation
Reference: HF LLM Course Ch1.3 — "Translation"

Model: Helsinki-NLP/opus-mt-fr-en (French → English)
Architecture: MarianMT (encoder-decoder)
Note: Requires sentencepiece and sacremoses packages.
"""

from transformers import pipeline
from src.data import TRANSLATION_INPUTS


def run_experiment(device: str = "cpu") -> dict:
    results = {"task": "translation"}

    # ── Course example (FR → EN) ──
    ex = TRANSLATION_INPUTS["course_example"]
    translator = pipeline("translation", model=ex["model"], device=device)
    out = translator(ex["text"])
    results["course_example"] = {
        "source": ex["text"],
        "translation": out[0]["translation_text"],
        "direction": f"{ex['src_lang']} → {ex['tgt_lang']}",
    }

    # ── Additional languages ──
    for test in TRANSLATION_INPUTS["custom_tests"]:
        translator = pipeline("translation", model=test["model"], device=device)
        out = translator(test["text"])
        results[f"{test['src_lang']}_to_{test['tgt_lang']}"] = {
            "source": test["text"],
            "translation": out[0]["translation_text"],
        }

    return results
```

#### 4.3.9 Image Classification — `src/pipelines/image_classification.py`

```python
"""
Experiment 9: Image Classification
Reference: HF LLM Course Ch1.3 — "Image classification"

Model: google/vit-base-patch16-224
Architecture: Vision Transformer (ViT)
"""

from transformers import pipeline
from src.data import IMAGE_INPUTS


def run_experiment(device: str = "cpu") -> dict:
    results = {"task": "image-classification"}

    ex = IMAGE_INPUTS["course_example"]
    classifier = pipeline("image-classification", model=ex["model"], device=device)
    out = classifier(ex["url"])

    results["course_example"] = {
        "url": ex["url"],
        "top_5_predictions": [
            {"label": r["label"], "score": round(r["score"], 4)}
            for r in out[:5]
        ],
    }

    return results
```

#### 4.3.10 Automatic Speech Recognition — `src/pipelines/speech_recognition.py`

```python
"""
Experiment 10: Automatic Speech Recognition
Reference: HF LLM Course Ch1.3 — "Automatic speech recognition"

Model: openai/whisper-large-v3
Architecture: Whisper (encoder-decoder)
Note: Requires soundfile/librosa. Use smaller model (whisper-tiny) for CPU.
"""

from transformers import pipeline
from src.data import ASR_INPUTS


def run_experiment(device: str = "cpu") -> dict:
    results = {"task": "automatic-speech-recognition"}

    ex = ASR_INPUTS["course_example"]

    # Use whisper-tiny for CPU environments, large-v3 for GPU
    model = ex["model"] if device != "cpu" else "openai/whisper-tiny"
    transcriber = pipeline("automatic-speech-recognition", model=model, device=device)
    out = transcriber(ex["url"])

    results["course_example"] = {
        "audio_url": ex["url"],
        "model_used": model,
        "transcription": out["text"],
    }

    return results
```

### 4.4 Main Orchestrator — `src/experiment_runner.py`

```python
"""
Main experiment orchestrator.
Runs all pipeline experiments, collects results, saves outputs.

Usage:
    python -m src.experiment_runner [--device cpu|cuda] [--tasks all|task1,task2]
"""

import argparse
import json
import time
import torch
from pathlib import Path

from src.pipelines import (
    text_classification,
    zero_shot,
    text_generation,
    fill_mask,
    ner,
    question_answering,
    summarization,
    translation,
    image_classification,
    speech_recognition,
)


TASK_REGISTRY = {
    "text-classification": text_classification,
    "zero-shot": zero_shot,
    "text-generation": text_generation,
    "fill-mask": fill_mask,
    "ner": ner,
    "question-answering": question_answering,
    "summarization": summarization,
    "translation": translation,
    "image-classification": image_classification,
    "speech-recognition": speech_recognition,
}


def main():
    parser = argparse.ArgumentParser(description="Transformers Pipeline Experiment")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--tasks", default="all", help="Comma-separated task names or 'all'")
    parser.add_argument("--output", default="results/outputs.json")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = 0 if args.device == "cuda" else "cpu"

    device_name = f"cuda ({torch.cuda.get_device_name(0)})" if device == 0 else "cpu"

    # Select tasks
    if args.tasks == "all":
        tasks = list(TASK_REGISTRY.keys())
    else:
        tasks = [t.strip() for t in args.tasks.split(",")]

    # ── Run experiments ──
    all_results = {
        "metadata": {
            "device": device_name,
            "torch_version": torch.__version__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "experiments": {},
    }

    for task_name in tasks:
        if task_name not in TASK_REGISTRY:
            print(f"⚠️  Unknown task: {task_name}, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  Running: {task_name}")
        print(f"{'='*60}")

        module = TASK_REGISTRY[task_name]
        start = time.perf_counter()

        try:
            result = module.run_experiment(device=device)
            result["status"] = "success"
            result["duration_sec"] = round(time.perf_counter() - start, 2)
        except Exception as e:
            result = {
                "task": task_name,
                "status": "failed",
                "error": str(e),
                "duration_sec": round(time.perf_counter() - start, 2),
            }
            print(f"  ❌ Failed: {e}")

        all_results["experiments"][task_name] = result
        print(f"  ✅ Done in {result['duration_sec']}s")

    # ── Save results ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  Results saved to {output_path}")
    print(f"{'='*60}")

    # ── Summary ──
    succeeded = sum(1 for r in all_results["experiments"].values() if r["status"] == "success")
    total = len(all_results["experiments"])
    print(f"\n  Summary: {succeeded}/{total} experiments passed")


if __name__ == "__main__":
    main()
```

### 4.5 Run Script — `scripts/run_experiment.sh`

```bash
#!/bin/bash
# ──────────────────────────────────────────────
# Transformers Pipeline Experiment Runner
# ──────────────────────────────────────────────
set -euo pipefail

echo "╔══════════════════════════════════════════╗"
echo "║  🤗 Transformers Pipeline Experiment     ║"
echo "╚══════════════════════════════════════════╝"

# 1. Setup virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi
source .venv/bin/activate

# 2. Install dependencies
pip install -q -r requirements.txt

# 3. Detect device
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    DEVICE="cuda"
    echo "🎮 GPU detected: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
    DEVICE="cpu"
    echo "💻 Running on CPU (some large models will be slow)"
fi

# 4. Run all experiments
python -m src.experiment_runner --device "$DEVICE" --tasks all --output results/outputs.json

# 5. Optional: run only text tasks (faster)
# python -m src.experiment_runner --device "$DEVICE" \
#   --tasks text-classification,zero-shot,text-generation,fill-mask,ner,question-answering \
#   --output results/text_only.json

echo ""
echo "Done! Check results/ directory for outputs."
```

### 4.6 Requirements — `requirements.txt`

```
transformers>=4.40.0
torch>=2.0.0
sentencepiece>=0.2.0
sacremoses>=0.1.1
protobuf>=4.25.0
accelerate>=0.30.0
soundfile>=0.12.0
librosa>=0.10.0
Pillow>=10.0.0
```

---

## 5. Expected Results

### 5.1 Output Schema Validation

| Task | Expected Output Keys | Score Range |
|------|---------------------|-------------|
| Sentiment Analysis | `label`, `score` | [0, 1] |
| Zero-shot | `sequence`, `labels`, `scores` | scores sum ≈ 1.0 |
| Text Generation | `generated_text` | N/A |
| Fill-Mask | `sequence`, `score`, `token`, `token_str` | [0, 1] |
| NER | `entity_group`, `word`, `score`, `start`, `end` | [0, 1] |
| Question Answering | `answer`, `score`, `start`, `end` | [0, 1] |
| Summarization | `summary_text` | N/A |
| Translation | `translation_text` | N/A |
| Image Classification | `label`, `score` | [0, 1] |
| ASR | `text` | N/A |

### 5.2 Expected Course Output Reproduction

| Task | Course Expected Output | Match Criterion |
|------|----------------------|-----------------|
| Sentiment: positive | `POSITIVE`, score ≈ 0.9598 | Label match, score within ±0.05 |
| Zero-shot: education | `education` top label, score ≈ 0.845 | Label match, score within ±0.1 |
| NER: 3 entities | PER(Sylvain), ORG(Hugging Face), LOC(Brooklyn) | All entities detected |
| QA: workplace | `Hugging Face` | Exact match |
| Translation: FR→EN | `This course is produced by Hugging Face.` | Semantic equivalence |

### 5.3 Approximate Latency Expectations (CPU)

| Task | Model Size | Expected Latency |
|------|-----------|-----------------|
| Sentiment Analysis | ~260MB | 50–100ms |
| Zero-shot | ~1.6GB | 200–500ms |
| Text Generation (GPT-2) | ~500MB | 100–300ms |
| Fill-Mask | ~330MB | 50–100ms |
| NER | ~1.2GB | 80–200ms |
| QA | ~260MB | 50–100ms |
| Summarization | ~1.2GB | 500–2000ms |
| Translation | ~300MB | 100–300ms |
| Image Classification | ~350MB | 100–200ms |
| ASR (whisper-tiny) | ~150MB | 1000–5000ms |

---

## 6. Ablation Studies

### 6.1 Text Generation — Temperature Sweep

| Temperature | Expected Behavior |
|------------|-------------------|
| 0.7 | More focused, coherent, less creative |
| 1.0 | Default — balanced creativity vs coherence |
| 1.5 | More random, creative, potentially incoherent |

### 6.2 Fill-Mask — Model Comparison

| Model | Mask Token | Training Objective | Expected Differences |
|-------|-----------|-------------------|---------------------|
| DistilRoBERTa | `<mask>` | Dynamic MLM | More diverse predictions |
| BERT-base-cased | `[MASK]` | Static MLM | More conservative, case-sensitive |

### 6.3 Zero-shot — Label Count Scaling

As the number of candidate labels increases, per-label scores should decrease (probability mass distributed among more options). Top-label accuracy may degrade with many labels.

### 6.4 Default vs Custom Model (Text Generation)

| Model | Params | Expected Quality |
|-------|--------|-----------------|
| GPT-2 | 124M | Decent general text, may lose coherence |
| SmolLM2-360M | 360M | More coherent, larger context window |

---

## 7. Architecture Reference

The experiment covers all three major transformer architecture families:

```
┌────────────────────────────────────────────────────────────┐
│                    ENCODER-ONLY                            │
│  (Bidirectional attention — understanding tasks)           │
│                                                            │
│  Models:  BERT, DistilBERT, RoBERTa                       │
│  Tasks:   Sentiment, NER, QA, Fill-Mask                   │
│  Reason:  Full context access for classification/extraction│
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                    DECODER-ONLY                            │
│  (Causal/autoregressive attention — generation tasks)      │
│                                                            │
│  Models:  GPT-2, SmolLM2                                   │
│  Tasks:   Text Generation                                  │
│  Reason:  Left-to-right generation, next token prediction  │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│                    ENCODER-DECODER                         │
│  (Cross-attention — sequence-to-sequence tasks)            │
│                                                            │
│  Models:  BART, MarianMT, DistilBART, Whisper              │
│  Tasks:   Summarization, Translation, Zero-shot, ASR       │
│  Reason:  Encode input fully, then generate output sequence│
└────────────────────────────────────────────────────────────┘
```

---

## 8. Reproducibility Checklist

- [ ] Python >=3.9 installed
- [ ] All packages from `requirements.txt` installed
- [ ] Internet access for model downloads (first run)
- [ ] ~10GB disk for model cache (`~/.cache/huggingface/`)
- [ ] GPU optional but recommended for ASR/summarization
- [ ] Random seeds set for text generation (note: some randomness is expected)
- [ ] All 10 pipeline tasks return valid output schema
- [ ] Course examples reproduce expected labels/answers
- [ ] Benchmark timings collected and saved
- [ ] Edge cases documented (empty strings, OOD languages)

---

## 9. Key Takeaways & Extension Ideas

### What This Experiment Demonstrates
1. **Unified API**: A single `pipeline()` call handles tokenization, inference, and decoding for any supported task
2. **Model Hub flexibility**: Swap `model=` parameter to use any compatible model from 500K+ options
3. **Architecture-task mapping**: Encoder-only for understanding, decoder-only for generation, encoder-decoder for seq2seq
4. **Zero-shot capability**: NLI-based classification without task-specific training data

### Extension Ideas
1. **Batch throughput scaling**: Measure throughput with batch sizes [1, 4, 8, 16, 32]
2. **Quantization impact**: Compare FP32 vs INT8 vs INT4 inference (using `bitsandbytes`)
3. **Cross-lingual evaluation**: Run NER and QA on Thai text to test multilingual models
4. **Pipeline internals**: Decompose `pipeline()` into manual tokenizer → model → postprocess steps (Chapter 1.5)
5. **Streaming generation**: Use `TextStreamer` for real-time token-by-token output
6. **ONNX export**: Convert pipelines to ONNX and compare latency

---

*Experiment designed following the [AI Researcher Reproduction Framework](https://docs.anthropic.com) and [HuggingFace LLM Course Chapter 1.3](https://huggingface.co/learn/llm-course/chapter1/3).*
