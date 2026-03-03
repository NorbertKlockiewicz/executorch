#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Export LFM2-VL text decoder (hybrid SSM+attention) to ExecuTorch PTE.

Uses the existing ET LFM2 infrastructure (ShortConvBlock, construct_transformer)
instead of trying to wrap the HF model directly, which avoids dynamic cache issues.

Weight mapping (HF -> ET):
  Conv layers:
    layers.N.conv.conv.weight            -> layers.N.conv.conv.weight       (direct)
    layers.N.conv.in_proj.weight         -> layers.N.conv.{B,C,x}_proj.weight  (split into 3)
    layers.N.conv.out_proj.weight        -> layers.N.conv.out_proj.weight    (direct)
    layers.N.operator_norm.weight        -> layers.N.attention_norm.weight   (rename)
    layers.N.ffn_norm.weight             -> layers.N.ffn_norm.weight         (direct)
    layers.N.feed_forward.{w1,w2,w3}    -> layers.N.feed_forward.{w1,w2,w3} (direct)
  Attention layers:
    layers.N.self_attn.q_proj.weight     -> layers.N.attention.wq.weight
    layers.N.self_attn.k_proj.weight     -> layers.N.attention.wk.weight
    layers.N.self_attn.v_proj.weight     -> layers.N.attention.wv.weight
    layers.N.self_attn.out_proj.weight   -> layers.N.attention.wo.weight
    layers.N.self_attn.q_layernorm.weight -> layers.N.attention.q_norm_fn.weight
    layers.N.self_attn.k_layernorm.weight -> layers.N.attention.k_norm_fn.weight
    layers.N.operator_norm.weight        -> layers.N.attention_norm.weight
    layers.N.ffn_norm.weight             -> layers.N.ffn_norm.weight
    layers.N.feed_forward.{w1,w2,w3}    -> layers.N.feed_forward.{w1,w2,w3}
  Top-level:
    embed_tokens.weight                  -> tok_embeddings.weight
    embedding_norm.weight                -> norm.weight
    lm_head.weight                       -> output.weight
"""

import torch
from torch.nn.attention import SDPBackend
from transformers import AutoModelForImageTextToText

from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    replace_kv_cache_with_custom_kv_cache,
)
from executorch.examples.models.llama.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.extension.llm.export.builder import LLMEdgeManager, DType


# LFM2-VL text backbone layer layout (from config.text_config.layer_types)
LAYER_TYPES = [
    "conv", "conv", "full_attention",
    "conv", "conv", "full_attention",
    "conv", "conv", "full_attention",
    "conv", "full_attention",
    "conv", "full_attention",
    "conv", "full_attention",
    "conv",
]


class Lfm2EdgeManager(LLMEdgeManager):
    def export(self) -> "Lfm2EdgeManager":
        dynamic_shape = self._get_dynamic_shape()
        print("  Tracing with strict=False...")
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            self.export_program = torch.export.export(
                self.model,
                self.example_inputs,
                dynamic_shapes=dynamic_shape,
                strict=False,
            )
            self.pre_autograd_graph_module = self.export_program.module()
        return self


class Lfm2TextDecoder(torch.nn.Module):
    """Thin wrapper matching LLaVA's pattern: forward(embeddings, input_pos)."""
    def __init__(self, text_model):
        super().__init__()
        self.text_model = text_model

    def forward(self, embeddings, input_pos):
        return self.text_model(None, {"input_pos": input_pos}, embeddings)


def translate_weights(hf_model) -> dict:
    """
    Convert LFM2-VL HF state dict to ET construct_transformer format.
    Handles both conv and attention layers correctly.
    """
    # Collect weights from language_model + lm_head
    sd = {}
    for k, v in hf_model.model.language_model.state_dict().items():
        sd[k] = v
    for k, v in hf_model.lm_head.state_dict().items():
        sd[f"lm_head.{k}"] = v

    out = {}

    for key, val in sd.items():
        # --- Top-level ---
        if key == "embed_tokens.weight":
            out["tok_embeddings.weight"] = val
            continue
        if key == "embedding_norm.weight":
            out["norm.weight"] = val
            continue
        if key == "lm_head.weight":
            out["output.weight"] = val
            continue

        # --- Layer keys ---
        if not key.startswith("layers."):
            # anything unmapped (rotary_emb buffers etc.) — skip
            continue

        parts = key.split(".", 2)   # ["layers", "N", "rest"]
        n = parts[1]
        rest = parts[2]
        prefix = f"layers.{n}."

        # Shared: feed_forward and ffn_norm are identical in both layer types
        if rest.startswith("feed_forward.") or rest == "ffn_norm.weight":
            out[prefix + rest] = val
            continue

        # operator_norm -> attention_norm (used in both conv and attention layers)
        if rest == "operator_norm.weight":
            out[prefix + "attention_norm.weight"] = val
            continue

        # --- Conv layer specifics ---
        if rest == "conv.conv.weight":
            out[prefix + "conv.conv.weight"] = val
            continue
        if rest == "conv.in_proj.weight":
            # Split [3*dim, dim] -> B_proj, C_proj, x_proj each [dim, dim]
            B, C, x = torch.chunk(val, 3, dim=0)
            out[prefix + "conv.B_proj.weight"] = B
            out[prefix + "conv.C_proj.weight"] = C
            out[prefix + "conv.x_proj.weight"] = x
            continue
        if rest == "conv.out_proj.weight":
            out[prefix + "conv.out_proj.weight"] = val
            continue

        # --- Attention layer specifics ---
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

        # Anything else (rotary buffers inside layers etc.) — skip
        print(f"  [skip] {key}")

    return out


def main():
    print("Loading LFM2-VL-450M from HuggingFace...")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        "LiquidAI/LFM2-VL-450M", device_map="cpu", torch_dtype=torch.float32
    )
    hf_model.eval()

    print("\nBuilding ET hybrid transformer (conv+attention)...")
    MAX_SEQ_LEN = 2048

    model_args = ModelArgs(
        dim=1024,
        n_layers=16,
        n_heads=16,
        n_kv_heads=8,
        vocab_size=65536,
        hidden_dim=4608,
        ffn_dim_multiplier=1,
        norm_eps=1e-5,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        max_context_len=MAX_SEQ_LEN,
        use_kv_cache=True,
        use_sdpa_with_kv_cache_op=True,
        use_hf_rope=True,
        enable_dynamic_shape=False,
        rope_theta=1000000.0,
        use_qk_norm=True,
        qk_norm_before_rope=True,
        layer_types=LAYER_TYPES,
    )

    et_model = construct_transformer(model_args)

    print("Translating weights...")
    state_dict = translate_weights(hf_model)
    missing, unexpected = et_model.load_state_dict(state_dict, strict=False, assign=True)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    print(f"  Loaded {len(state_dict)} tensors")

    decoder = Lfm2TextDecoder(et_model)
    decoder.eval()

    # --- Verify weights loaded correctly with a quick sanity check ---
    print("\nVerifying weight mapping...")
    et_embed = et_model.tok_embeddings.weight
    hf_embed = hf_model.model.language_model.embed_tokens.weight
    diff = (et_embed - hf_embed).abs().max().item()
    print(f"  embed_tokens max_diff: {diff:.6f}  {'✅' if diff < 1e-5 else '❌'}")

    et_conv0 = et_model.layers[0].conv.B_proj.weight
    hf_in_proj = hf_model.model.language_model.layers[0].conv.in_proj.weight
    hf_B = hf_in_proj[:1024]
    diff = (et_conv0 - hf_B).abs().max().item()
    print(f"  layer0 B_proj max_diff: {diff:.6f}  {'✅' if diff < 1e-5 else '❌'}")

    et_wq = et_model.layers[2].attention.wq.weight
    hf_wq = hf_model.model.language_model.layers[2].self_attn.q_proj.weight
    diff = (et_wq - hf_wq).abs().max().item()
    print(f"  layer2 wq max_diff: {diff:.6f}  {'✅' if diff < 1e-5 else '❌'}")

    # --- Export ---
    print("\nExporting...")
    dummy_seq_len = 8
    dummy_embeddings = torch.randn(1, dummy_seq_len, 1024)
    dummy_input_pos  = torch.arange(dummy_seq_len, dtype=torch.int64)

    source_transforms = [
        replace_kv_cache_with_custom_kv_cache,
        replace_sdpa_with_custom_op,
    ]

    from torch.export import Dim
    token_dim = Dim("token_dim", min=1, max=MAX_SEQ_LEN)
    dynamic_shapes = ({1: token_dim}, {0: token_dim})

    manager = Lfm2EdgeManager(
        model=decoder,
        modelname="lfm2_text_decoder",
        max_seq_len=MAX_SEQ_LEN,
        dtype=DType.fp32,
        use_kv_cache=True,
        example_inputs=(dummy_embeddings, dummy_input_pos),
        dynamic_shapes=dynamic_shapes,
    )

    print("1. Source transforms + export...")
    manager = manager.source_transform(source_transforms).export()

    print("2. Lowering to Edge IR + XNNPACK...")
    lowered = to_edge_transform_and_lower(
        manager.export_program,
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    print("3. Finalizing ExecuTorch program...")
    et_program = lowered.to_executorch(
        ExecutorchBackendConfig(extract_delegate_segments=True)
    )

    output_name = "lfm2_text_decoder_xnnpack_fp32.pte"
    print(f"Saving {output_name}...")
    with open(output_name, "wb") as f:
        et_program.write_to_file(f)

    print(f"\n✅ Saved {output_name}")
    print(f"   Methods: {et_program.methods}")


if __name__ == "__main__":
    main()
