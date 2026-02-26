"""Test inputs for all pipeline tasks.

Includes course examples from HuggingFace LLM Course Chapter 1.3
and edge cases for each task type.
"""

# ---------------------------------------------------------------------------
# 1. Text Classification (sentiment-analysis)
# ---------------------------------------------------------------------------
TEXT_CLASSIFICATION_INPUTS: dict = {
    "course_examples": [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ],
    "edge_cases": {
        "neutral": "The weather is somewhat acceptable today.",
        "very_long": "This is a review. " * 50,
        "non_english": "Je suis très heureux d'être ici.",
        "single_word": "Excellent!",
        "empty": "",
    },
}

# ---------------------------------------------------------------------------
# 2. Zero-Shot Classification
# ---------------------------------------------------------------------------
ZERO_SHOT_INPUTS: dict = {
    "course_examples": {
        "sequence": "This is a course about the Transformers library",
        "candidate_labels": ["education", "politics", "business"],
    },
    "edge_cases": {
        "two_labels": {
            "sequence": "The stock market crashed today",
            "candidate_labels": ["economics", "sports"],
        },
        "five_labels": {
            "sequence": "Apple released a new iPhone model",
            "candidate_labels": ["technology", "business", "design", "marketing", "manufacturing"],
        },
        "ten_labels": {
            "sequence": "Scientists discovered a new exoplanet orbiting a distant star",
            "candidate_labels": [
                "astronomy",
                "physics",
                "biology",
                "chemistry",
                "geology",
                "medicine",
                "technology",
                "arts",
                "sports",
                "politics",
            ],
        },
    },
}

# ---------------------------------------------------------------------------
# 3. Text Generation
# ---------------------------------------------------------------------------
TEXT_GENERATION_INPUTS: dict = {
    "course_examples": [
        "In this course, we will teach you how to",
    ],
    "edge_cases": {
        "short_prompt": "The future of AI is",
        "question_prompt": "What is the meaning of",
        "continuation": "Once upon a time, there was a",
    },
    "ablation": {
        "temperatures": [0.7, 1.0, 1.5],
        "models": {
            "default": "openai-community/gpt2",
            "alternative": "HuggingFaceTB/SmolLM2-360M",
        },
    },
}

# ---------------------------------------------------------------------------
# 4. Fill-Mask
# ---------------------------------------------------------------------------
FILL_MASK_INPUTS: dict = {
    "course_examples": {
        "distilroberta": "This course will teach you all about <mask> models.",
        "bert": "This course will teach you all about [MASK] models.",
    },
    "edge_cases": {
        "mask_at_start": "<mask> is the most important concept in NLP.",
        "mask_at_end": "The best way to learn machine learning is by <mask>.",
    },
}

# ---------------------------------------------------------------------------
# 5. Named Entity Recognition
# ---------------------------------------------------------------------------
NER_INPUTS: dict = {
    "course_examples": [
        "My name is Sylvain and I work at Hugging Face in Brooklyn.",
    ],
    "edge_cases": {
        "no_entities": "The weather is nice today.",
        "many_entities": (
            "Barack Obama visited Microsoft headquarters in Seattle with Tim Cook from Apple."
        ),
        "dates_and_money": "On January 15, 2024, the company earned $5 million.",
    },
}

# ---------------------------------------------------------------------------
# 6. Question Answering
# ---------------------------------------------------------------------------
QA_INPUTS: dict = {
    "course_examples": {
        "question": "Where do I work?",
        "context": "My name is Sylvain and I work at Hugging Face in Brooklyn",
    },
    "edge_cases": {
        "multi_sentence": {
            "question": "What is the capital of France?",
            "context": (
                "France is a country in Western Europe. Its capital city is Paris. "
                "The Eiffel Tower is located there."
            ),
        },
        "technical": {
            "question": "What does the pipeline() function do?",
            "context": (
                "The pipeline() function in HuggingFace Transformers is the highest-level "
                "entry point for NLP tasks. It encapsulates preprocessing, model forward pass, "
                "and postprocessing in a single callable."
            ),
        },
    },
}

# ---------------------------------------------------------------------------
# 7. Summarization
# ---------------------------------------------------------------------------
SUMMARIZATION_INPUTS: dict = {
    "course_examples": [
        (
            "America has changed dramatically during recent years. Not only has the number of "
            "graduates in traditional engineering disciplines such as mechanical, civil, "
            "electrical, chemical, and aeronautical engineering declined, but in most of "
            "the premier universities in the United States, many of the valedictorians "
            "and high-ranking students are girls. Increasingly, these students are "
            "particularly in programs such as health and medical related fields, "
            "including premedical programs, medicine, law, business, and computer "
            "sciences. For women, the attractiveness of these majors is that they "
            "offer the opportunity to combine intellectual rigor and career potential "
            "with an altruistic contribution to society. This particular article, "
            "however, is about another major -- the STEM fields, which stands for "
            "science, technology, engineering, and mathematics."
        )
    ],
    "edge_cases": {
        "short_text": (
            "Artificial intelligence is rapidly transforming many industries. "
            "Companies are investing heavily in machine learning research."
        ),
        "technical": (
            "The transformer architecture, introduced in the paper 'Attention Is All You Need', "
            "relies entirely on self-attention mechanisms to compute representations of its "
            "input and output without using sequence-aligned RNNs or convolutions. "
            "It has become the dominant architecture for NLP tasks, achieving state-of-the-art "
            "results on machine translation, text summarization, and language modeling."
        ),
    },
    "ablation": {
        "max_lengths": [50, 100, 200],
    },
}

# ---------------------------------------------------------------------------
# 8. Translation (French → English)
# ---------------------------------------------------------------------------
TRANSLATION_INPUTS: dict = {
    "course_examples": [
        "Ce cours est produit par Hugging Face.",
    ],
    "edge_cases": {
        "greeting": "Bonjour, comment allez-vous?",
        "question": "Qu'est-ce que l'intelligence artificielle?",
        "technical": (
            "Les modèles de transformateur révolutionnent le traitement du langage naturel."
        ),
    },
    "models": {
        "fr_en": "Helsinki-NLP/opus-mt-fr-en",
        "en_fr": "Helsinki-NLP/opus-mt-en-fr",
    },
}

# ---------------------------------------------------------------------------
# 9. Image Classification
# ---------------------------------------------------------------------------
IMAGE_CLASSIFICATION_INPUTS: dict = {
    "course_examples": [
        (
            "https://huggingface.co/datasets/huggingface/documentation-images"
            "/resolve/main/pipeline-cat-chonk.jpeg"
        ),
    ],
    "edge_cases": {
        "landscape": (
            "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/"
            "24701-nature-natural-beauty.jpg/320px-24701-nature-natural-beauty.jpg"
        ),
    },
}

# ---------------------------------------------------------------------------
# 10. Automatic Speech Recognition
# ---------------------------------------------------------------------------
SPEECH_RECOGNITION_INPUTS: dict = {
    "course_examples": [
        "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
    ],
    "edge_cases": {
        "short_audio": ("https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac"),
    },
    "models": {
        "large_gpu": "openai/whisper-large-v3",
        "tiny_cpu": "openai/whisper-tiny",
    },
}
