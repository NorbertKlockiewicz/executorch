"""
HuggingFace → ExecuTorch weight key translation for LFM2.5-VL-1.6B.

The HF model stores weights under the LFM2 naming convention; the ET
construct_transformer() expects a different layout. This file handles
the translation — no ET pipeline imports, just plain dict manipulation.

Key mappings:
  HF                              ET (construct_transformer)
  ──────────────────────────────  ──────────────────────────────────────
  embed_tokens.weight          →  tok_embeddings.weight
  embedding_norm.weight        →  norm.weight
  lm_head.weight               →  output.weight
  layers.N.operator_norm.weight → layers.N.attention_norm.weight
  layers.N.conv.in_proj.weight  → split into B_proj / C_proj / x_proj
  layers.N.self_attn.q_proj    →  layers.N.attention.wq
  layers.N.self_attn.k_proj    →  layers.N.attention.wk
  layers.N.self_attn.v_proj    →  layers.N.attention.wv
  layers.N.self_attn.out_proj  →  layers.N.attention.wo
  layers.N.self_attn.q_layernorm → layers.N.attention.q_norm_fn
  layers.N.self_attn.k_layernorm → layers.N.attention.k_norm_fn
"""

import torch


def translate_weights(hf_model) -> dict:
    """
    Translates the HF LFM2.5-VL state dict into the format expected by
    construct_transformer(). Returns a plain Python dict.

    Call order matters in export.py:
      export_text_decoder() must run BEFORE export_token_embedding() because
      EmbeddingQuantHandler mutates hf_model.model.language_model.embed_tokens
      in-place, which would break load_state_dict in the decoder.
    """
    # Collect language model weights + lm_head
    raw = {}
    for k, v in hf_model.model.language_model.state_dict().items():
        raw[k] = v
    for k, v in hf_model.lm_head.state_dict().items():
        raw[f"lm_head.{k}"] = v

    out = {}
    for key, val in raw.items():

        # ── Top-level renames ──────────────────────────────────────────────
        if key == "embed_tokens.weight":
            out["tok_embeddings.weight"] = val
            continue
        if key == "embedding_norm.weight":
            out["norm.weight"] = val
            continue
        if key == "lm_head.weight":
            out["output.weight"] = val
            continue

        # ── Only layer weights below this point ───────────────────────────
        if not key.startswith("layers."):
            continue

        # layers.<N>.<rest>
        _, n, rest = key.split(".", 2)
        prefix = f"layers.{n}."

        # ── Feed-forward (same names in both) ─────────────────────────────
        if rest.startswith("feed_forward.") or rest == "ffn_norm.weight":
            out[prefix + rest] = val
            continue

        # ── Shared norm (HF: operator_norm → ET: attention_norm) ──────────
        if rest == "operator_norm.weight":
            out[prefix + "attention_norm.weight"] = val
            continue

        # ── Conv (SSM / ShortConv) layer weights ──────────────────────────
        if rest == "conv.conv.weight":
            out[prefix + "conv.conv.weight"] = val
            continue
        if rest == "conv.in_proj.weight":
            # in_proj [3*dim, dim] → three separate projections
            B, C, x = torch.chunk(val, 3, dim=0)
            out[prefix + "conv.B_proj.weight"] = B
            out[prefix + "conv.C_proj.weight"] = C
            out[prefix + "conv.x_proj.weight"] = x
            continue
        if rest == "conv.out_proj.weight":
            out[prefix + "conv.out_proj.weight"] = val
            continue

        # ── Attention layer weights ────────────────────────────────────────
        if rest == "self_attn.q_proj.weight":
            out[prefix + "attention.wq.weight"] = val
            continue
        if rest == "self_attn.k_proj.weight":
            out[prefix + "attention.wk.weight"] = val
            continue
        if rest == "self_attn.v_proj.weight":
            out[prefix + "attention.wv.weight"] = val
            continue
        if rest == "self_attn.out_proj.weight":
            out[prefix + "attention.wo.weight"] = val
            continue
        if rest == "self_attn.q_layernorm.weight":
            out[prefix + "attention.q_norm_fn.weight"] = val
            continue
        if rest == "self_attn.k_layernorm.weight":
            out[prefix + "attention.k_norm_fn.weight"] = val
            continue

        # Any key that falls through here is intentionally skipped
        # (e.g. KV cache buffers, conv_state — these are initialised by
        # construct_transformer and not present in the HF checkpoint)

    return out
