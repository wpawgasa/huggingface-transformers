"""Probe 5: Attention Layer Visualization.

Reference: Ch1.4 -- "Attention layers"

From the course:
  "This layer will tell the model to pay specific attention to certain words
   in the sentence you passed it (and more or less ignore the others)"

  "Given the input 'You like this course', a translation model will need
   to also attend to the adjacent word 'You' to get the proper translation
   for the word 'like', because in French the verb 'like' is conjugated
   differently depending on the subject."

This probe extracts and visualizes real attention weights to verify
these theoretical claims empirically.
"""

from src.architecture_deepdive.data import ATTENTION_SENTENCES
from src.architecture_deepdive.utils.attention_tools import (
    compare_causal_vs_bidirectional_mask,
    extract_attention_weights,
    get_attention_to_token,
)
from src.architecture_deepdive.utils.plotting import (
    plot_attention_mask_comparison,
    plot_attention_matrix,
)


def run_experiment(device: str = "cpu") -> dict:
    """Extract and visualize attention weights from BERT and GPT-2.

    Args:
        device: Device string ("cpu" or "cuda").

    Returns:
        Dict with mask comparison, per-sentence attention data,
        coreference test results, and figure paths.
    """
    results: dict = {"task": "attention_visualization"}

    # -- 5.1: Attention mask comparison (theoretical) --
    tokens = ["You", "like", "this", "course"]
    masks = compare_causal_vs_bidirectional_mask(seq_len=len(tokens))
    results["mask_comparison"] = {
        "tokens": tokens,
        "causal_masked_positions": masks["causal_masked_positions"],
        "note": masks["note"],
    }

    # Generate comparison figure
    plot_attention_mask_comparison(
        masks["bidirectional"],
        masks["causal"],
        tokens,
        save_name="01_mask_comparison.png",
    )
    results["figures"] = ["results/figures/01_mask_comparison.png"]

    # -- 5.2: BERT attention on coreference (bidirectional) --
    bert_model = "bert-base-uncased"

    for key, sentence in ATTENTION_SENTENCES.items():
        attn_data = extract_attention_weights(bert_model, sentence, device)

        # Plot first layer, first head
        plot_attention_matrix(
            attn_data["attentions"],
            attn_data["tokens"],
            title=f"BERT Attention: {key}",
            save_name=f"02_bert_{key}.png",
            layer=0,
            head=0,
        )
        results["figures"].append(f"results/figures/02_bert_{key}.png")

        # Also plot last layer (more task-specific patterns)
        plot_attention_matrix(
            attn_data["attentions"],
            attn_data["tokens"],
            title=f"BERT Attention (Last Layer): {key}",
            save_name=f"02_bert_{key}_last.png",
            layer=-1,
            head=0,
        )
        results["figures"].append(f"results/figures/02_bert_{key}_last.png")

    # -- 5.3: Course example -- "it" in coreference --
    coref_tired = extract_attention_weights(bert_model, ATTENTION_SENTENCES["coref_animal"], device)
    coref_wide = extract_attention_weights(bert_model, ATTENTION_SENTENCES["coref_street"], device)

    # Measure attention from "it" -> "animal" vs "it" -> "street"
    it_to_animal = get_attention_to_token(
        coref_tired["attentions"],
        coref_tired["tokens"],
        "animal",
        layer=-1,
    )
    it_to_street = get_attention_to_token(
        coref_wide["attentions"],
        coref_wide["tokens"],
        "street",
        layer=-1,
    )

    results["coreference_test"] = {
        "hypothesis": (
            "In 'it was too tired', 'it' should attend more to 'animal'. "
            "In 'it was too wide', 'it' should attend more to 'street'."
        ),
        "tired_sentence": {
            "attention_from_all_tokens_to_animal": it_to_animal,
        },
        "wide_sentence": {
            "attention_from_all_tokens_to_street": it_to_street,
        },
    }

    # -- 5.4: GPT-2 attention (causal) for comparison --
    gpt_model = "gpt2"
    gpt_attn = extract_attention_weights(gpt_model, ATTENTION_SENTENCES["agreement_short"], device)
    plot_attention_matrix(
        gpt_attn["attentions"],
        gpt_attn["tokens"],
        title="GPT-2 Attention (Causal)",
        save_name="03_gpt2_causal.png",
        layer=0,
        head=0,
    )
    results["figures"].append("results/figures/03_gpt2_causal.png")

    results["causal_observation"] = (
        "GPT-2 attention matrix shows a clear lower-triangular pattern: "
        "each token can only attend to itself and preceding tokens. "
        "BERT attention matrices are full (bidirectional). "
        "This is the defining structural difference between encoder"
        " and decoder."
    )

    return results
