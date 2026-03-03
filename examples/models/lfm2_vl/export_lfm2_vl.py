#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Export LFM2-VL-450M as a single multi-method PTE compatible with
ExecuTorch's generic MultimodalRunner (C++).

Produces: lfm2_vl_xnnpack.pte  with three named methods:
  - "vision_encoder"  : pixel_values [1, 3, 512, 512] f32 NCHW -> image_embeds [1, 256, 1024]
  - "token_embedding" : token_ids    [1, seq_len]      i64      -> embeds [1, seq_len, 1024]
  - "text_decoder"    : (embeds [1, seq_len, 1024], input_pos [seq_len]) -> logits [1, 65536]

The vision_encoder accepts raw NCHW float32 pixels in [0, 255] range.
It performs normalization ((x/255 - 0.5) / 0.5) and patch extraction
(16x16 patches -> [1, 1024, 768]) internally, so the C++ runner only
needs to resize to 512x512 and pass the raw pixel tensor.

Method names match the constants in extension/llm/runner/constants.h so the
generic MultimodalRunner can load and call them without any model-specific code.

Usage:
    python examples/models/lfm2_vl/export_lfm2_vl.py
    python examples/models/lfm2_vl/export_lfm2_vl.py --output lfm2_vl_xnnpack.pte
"""

import argparse
import math
import torch
import torch.nn.functional as F
from torch.export import Dim, export
from torch.nn.attention import SDPBackend
from transformers import AutoModelForImageTextToText
from PIL import Image

from executorch.backends.xnnpack.partition.config.xnnpack_config import (
    ConfigPrecisionType,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.examples.models.llama.export_llama_lib import (
    get_quantizer_and_quant_params,
)
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    replace_kv_cache_with_custom_kv_cache,
)
from executorch.examples.models.llama.source_transformation.quantize import (
    EmbeddingQuantHandler,
    get_quant_weight_transform,
)
from executorch.examples.models.llama.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import (
    ConstraintBasedSymShapeEvalPass,
    HintBasedSymShapeEvalPass,
)
from executorch.extension.llm.export.builder import LLMEdgeManager, DType
from executorch.extension.llm.export.config.llm_config import LlmConfig


# ---------------------------------------------------------------------------
# LFM2-VL architecture constants
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048
IMAGE_SIZE = 512  # 512x512 → 32x32 patches → 256 image tokens after projector
FIXED_H, FIXED_W = 32, 32

# Per-model architecture configs — keyed by short name
MODEL_CONFIGS = {
    "450M": {
        "model_id": "LiquidAI/LFM2-VL-450M",
        "dim": 1024,
        "n_heads": 16,
        "n_kv_heads": 8,
        "hidden_dim": 4608,
        "n_layers": 16,
        "layer_types": [
            "conv", "conv", "full_attention",
            "conv", "conv", "full_attention",
            "conv", "conv", "full_attention",
            "conv", "full_attention",
            "conv", "full_attention",
            "conv", "full_attention",
            "conv",
        ],
    },
    "1.6B": {
        "model_id": "LiquidAI/LFM2-VL-1.6B",
        "dim": 2048,
        "n_heads": 32,
        "n_kv_heads": 8,
        "hidden_dim": 8192,  # actual w1/w3 shape — config.intermediate_size=12288 is pre-SwiGLU (12288*2/3=8192)
        "n_layers": 16,
        "layer_types": [
            "conv", "conv", "full_attention",
            "conv", "conv", "full_attention",
            "conv", "conv", "full_attention",
            "conv", "full_attention",
            "conv", "full_attention",
            "conv", "full_attention",
            "conv",
        ],
    },
    "3B": {
        "model_id": "LiquidAI/LFM2-VL-3B",
        "dim": 2048,
        "n_heads": 32,
        "n_kv_heads": 8,
        "hidden_dim": 10752,  # block_ff_dim — block_auto_adjust_ff_dim=false so used as-is
        "n_layers": 30,
        "layer_types": [
            "conv", "conv", "full_attention",
            "conv", "conv", "full_attention",
            "conv", "conv", "conv", "full_attention",
            "conv", "conv", "conv", "full_attention",
            "conv", "conv", "conv", "full_attention",
            "conv", "conv", "conv", "full_attention",
            "conv", "conv", "full_attention",
            "conv", "conv", "full_attention",
            "conv", "conv",
        ],
    },
}


# ---------------------------------------------------------------------------
# Shared LLMEdgeManager subclass (used by both vision encoder and text decoder)
# ---------------------------------------------------------------------------


class Lfm2EdgeManager(LLMEdgeManager):
    def export(self) -> "Lfm2EdgeManager":
        dynamic_shape = self._get_dynamic_shape()
        with torch.nn.attention.sdpa_kernel([SDPBackend.MATH]), torch.no_grad():
            self.export_program = torch.export.export(
                self.model,
                self.example_inputs,
                dynamic_shapes=dynamic_shape,
                strict=False,
            )
            self.pre_autograd_graph_module = self.export_program.module()
        return self


# ---------------------------------------------------------------------------
# Vision encoder export
# ---------------------------------------------------------------------------

PATCH_SIZE = 16  # SigLIP ViT patch size


def export_vision_encoder(hf_model, model_cfg, dtype: DType = DType.fp32) -> torch.export.ExportedProgram:
    """Export SigLIP ViT + MLP projector as 'vision_encoder' method.

    Accepts raw NCHW float32 pixels in [0, 255] range — compatible with
    the C++ LLaVA/Multimodal runner which passes resized pixels directly.

    Bakes in:
      - Normalization: (x/255 - 0.5) / 0.5
      - Patch extraction: 16x16 patches -> [1, 1024, 768] (HW-major ordering)
      - Fixed 512x512 input (32x32 patches, all valid — no masking needed)
      - Positional embeddings pre-computed for 32x32 grid
      - Full attention mask and spatial_shapes as constants
    """
    print("  Pre-computing positional embeddings for 32x32 patches...")
    orig_embeddings = hf_model.model.vision_tower.vision_model.embeddings.position_embedding.weight.data
    num_positions, dim = orig_embeddings.shape
    sqrt_num = int(math.sqrt(num_positions))
    grid = orig_embeddings.reshape(sqrt_num, sqrt_num, dim)
    resized = F.interpolate(
        grid.permute(2, 0, 1).unsqueeze(0),
        size=(FIXED_H, FIXED_W),
        mode="bilinear",
        align_corners=False,
    )
    # .contiguous() prevents dim_order mismatches in the portable aten::add.out kernel
    precomputed_pos = (
        resized.squeeze(0).permute(1, 2, 0).reshape(FIXED_H * FIXED_W, dim).contiguous()
    )

    def patched_resize(positional_embeddings, height=None, width=None, max_length=None):
        return precomputed_pos

    hf_model.model.vision_tower.vision_model.embeddings.resize_positional_embeddings = (
        patched_resize
    )

    FULL_MASK = torch.ones(1, FIXED_H * FIXED_W, dtype=torch.int32)

    class VisionEncoder(torch.nn.Module):
        """Accepts [1, 3, 512, 512] pixels in [0, 255] range, in model dtype.

        Internally: normalize -> unfold 16x16 patches -> [1, 1024, 768] ->
        vision tower -> MLP projector -> [256, dim].
        The C++ runner must cast its float32 pixel buffer to the model dtype first.
        """

        def __init__(self, vision_tower, projector):
            super().__init__()
            self.vision_tower = vision_tower
            self.projector = projector

        def forward(self, nchw_pixels):
            # nchw_pixels: [1, 3, 512, 512] in model dtype, range [0, 255]
            # Normalize: (x/255 - 0.5) / 0.5
            x = nchw_pixels / 255.0
            x = (x - 0.5) / 0.5

            # Extract 16x16 patches in HW-major order -> [1, 1024, 768]
            # unfold: [1, 3, 32, 32, 16, 16]
            x = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
            # reorder to [1, 32, 32, 16, 16, 3] then flatten -> [1, 1024, 768]
            x = x.permute(0, 2, 3, 4, 5, 1).reshape(
                1, FIXED_H * FIXED_W, PATCH_SIZE * PATCH_SIZE * 3
            )

            out = self.vision_tower(
                pixel_values=x,
                pixel_attention_mask=FULL_MASK,
                spatial_shapes=torch.tensor([[FIXED_H, FIXED_W]], dtype=torch.int64),
                return_dict=True,
            )
            feats = out.last_hidden_state  # [1, 1024, 768]
            feats = feats.reshape(feats.shape[0], FIXED_H, FIXED_W, -1)
            projected = self.projector(feats)  # [1, 16, 16, 1024]
            return projected.reshape(1, -1, projected.shape[-1])  # [1, 256, 1024]

    torch_dtype = dtype.to_torch_dtype()
    # Example input: [1, 3, 512, 512] in model dtype, range [0, 255].
    # The C++ runner converts its float32 pixels to the model dtype before calling.
    pixel_values = torch.randint(
        0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch_dtype
    )

    encoder = VisionEncoder(
        hf_model.model.vision_tower,
        hf_model.model.multi_modal_projector,
    )
    encoder.eval()

    # Sanity check: compare eager output with processor-based output
    print("  Verifying preprocessing matches HF processor...")
    import os

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    img_path = os.path.join(repo_root, "dog.jpg")
    if os.path.exists(img_path):
        from transformers import AutoProcessor as _AP

        processor = _AP.from_pretrained(model_cfg["model_id"])
        image = Image.open(img_path).resize((IMAGE_SIZE, IMAGE_SIZE))
        inputs = processor(text=["<image>"], images=[image], return_tensors="pt")
        proc_pv = inputs["pixel_values"]  # [1, 1024, 768]

        import numpy as np

        arr = torch.tensor(np.array(image), dtype=torch.float32)  # [512, 512, 3]
        raw_nchw = arr.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 512, 512]

        with torch.no_grad():
            our_pv = raw_nchw / 255.0
            our_pv = (our_pv - 0.5) / 0.5
            our_pv = our_pv.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(
                3, PATCH_SIZE, PATCH_SIZE
            )
            our_pv = our_pv.permute(0, 2, 3, 4, 5, 1).reshape(
                1, FIXED_H * FIXED_W, PATCH_SIZE * PATCH_SIZE * 3
            )
        diff = (our_pv.float() - proc_pv.float()).abs().max().item()
        print(
            f"    Max pixel_values diff vs processor: {diff:.2e}  {'✓' if diff < 1e-4 else '✗ MISMATCH'}"
        )
        pixel_values = raw_nchw.to(torch_dtype)

    print("  Tracing vision encoder...")
    with torch.no_grad():
        ep = export(encoder, (pixel_values,), strict=False)

    return ep


# ---------------------------------------------------------------------------
# Token embedding export
# ---------------------------------------------------------------------------


def export_token_embedding(hf_model, quantize: bool) -> torch.export.ExportedProgram:
    """Export nn.Embedding(65536, 1024) as 'token_embedding' method.

    When quantize=True applies int8 group quantization (bitwidth=8, group_size=32).
    NOTE: EmbeddingQuantHandler mutates hf_model.model.language_model in-place,
    so export_text_decoder must be called before this function.

    Dynamic sequence length — the C++ runner passes actual (unpadded) token
    sequences at runtime.
    """
    language_model = hf_model.model.language_model

    example_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int64)
    token_dim = Dim("token_dim_1", min=1, max=MAX_SEQ_LEN)
    dynamic_shapes = [{1: token_dim}]

    if quantize:
        print("  Quantizing token embedding (int8, group_size=32)...")
        quantized_lm = EmbeddingQuantHandler(
            language_model,
            bitwidth=8,
            group_size=32,
            packed=False,
        ).quantized_model()
        embed_module = quantized_lm.embed_tokens
    else:
        embed_module = language_model.get_input_embeddings()

    print("  Tracing token embedding...")
    with torch.no_grad():
        ep = export(embed_module, (example_ids,), dynamic_shapes=dynamic_shapes, strict=True)

    return ep


# ---------------------------------------------------------------------------
# Text decoder export
# ---------------------------------------------------------------------------


def translate_weights(hf_model) -> dict:
    """Translate HF LFM2-VL state dict to ET construct_transformer format."""
    sd = {}
    for k, v in hf_model.model.language_model.state_dict().items():
        sd[k] = v
    for k, v in hf_model.lm_head.state_dict().items():
        sd[f"lm_head.{k}"] = v

    out = {}
    for key, val in sd.items():
        # Top-level
        if key == "embed_tokens.weight":
            out["tok_embeddings.weight"] = val
            continue
        if key == "embedding_norm.weight":
            out["norm.weight"] = val
            continue
        if key == "lm_head.weight":
            out["output.weight"] = val
            continue

        if not key.startswith("layers."):
            continue

        parts = key.split(".", 2)
        n, rest = parts[1], parts[2]
        prefix = f"layers.{n}."

        # Shared across both layer types
        if rest.startswith("feed_forward.") or rest == "ffn_norm.weight":
            out[prefix + rest] = val
            continue
        if rest == "operator_norm.weight":
            out[prefix + "attention_norm.weight"] = val
            continue

        # Conv layer specifics
        if rest == "conv.conv.weight":
            out[prefix + "conv.conv.weight"] = val
            continue
        if rest == "conv.in_proj.weight":
            # Split [3*dim, dim] → B_proj, C_proj, x_proj each [dim, dim]
            B, C, x = torch.chunk(val, 3, dim=0)
            out[prefix + "conv.B_proj.weight"] = B
            out[prefix + "conv.C_proj.weight"] = C
            out[prefix + "conv.x_proj.weight"] = x
            continue
        if rest == "conv.out_proj.weight":
            out[prefix + "conv.out_proj.weight"] = val
            continue

        # Attention layer specifics
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

    return out


def export_text_decoder(hf_model, model_cfg, quantize: bool, dtype: DType = DType.fp32) -> torch.export.ExportedProgram:
    """Export the hybrid LFM2 decoder (10 conv + 6 attention layers) as 'text_decoder'.

    Uses the ET LFM2 infrastructure (construct_transformer + ShortConvBlock)
    rather than wrapping the HF model directly, which avoids DynamicCache issues.

    enable_dynamic_shape=False avoids .item() in rope.get_freqs which is not
    traceable with FakeTensors. Dynamic shapes are still enforced via Dim().
    """
    print("  Building ET hybrid transformer...")
    model_args = ModelArgs(
        dim=model_cfg["dim"],
        n_layers=model_cfg["n_layers"],
        n_heads=model_cfg["n_heads"],
        n_kv_heads=model_cfg["n_kv_heads"],
        vocab_size=65536,
        hidden_dim=model_cfg["hidden_dim"],
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
        layer_types=model_cfg["layer_types"],
    )

    et_model = construct_transformer(model_args)

    print("  Translating weights...")
    state_dict = translate_weights(hf_model)
    missing, unexpected = et_model.load_state_dict(
        state_dict, strict=False, assign=True
    )
    print(
        f"    Loaded {len(state_dict)} tensors, {len(missing)} missing (expected: KV/conv buffers)"
    )

    # Cast all parameters AND buffers (conv_state, kv_cache, etc.) to the target
    # dtype now, before source transforms. This ensures conv_state buffers in
    # ShortConv are fp16 so the cat([conv_state, Bx]) doesn't produce a dtype
    # mismatch when Conv1d weights are fp16.
    if dtype != DType.fp32:
        et_model = et_model.to(dtype.to_torch_dtype())

    class Lfm2TextDecoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.text_model = model

        def forward(self, embeddings, input_pos):
            return self.text_model(None, {"input_pos": input_pos}, embeddings)

    decoder = Lfm2TextDecoder(et_model)
    decoder.eval()

    torch_dtype = dtype.to_torch_dtype()
    dummy_seq_len = 8
    dummy_embeddings = torch.randn(1, dummy_seq_len, model_cfg["dim"], dtype=torch_dtype)
    dummy_input_pos = torch.arange(dummy_seq_len, dtype=torch.int64)

    token_dim = Dim("token_dim", min=1, max=MAX_SEQ_LEN)
    dynamic_shapes = ({1: token_dim}, {0: token_dim})

    manager = Lfm2EdgeManager(
        model=decoder,
        modelname="lfm2_text_decoder",
        max_seq_len=MAX_SEQ_LEN,
        dtype=dtype,
        use_kv_cache=True,
        example_inputs=(dummy_embeddings, dummy_input_pos),
        dynamic_shapes=dynamic_shapes,
    )

    source_transforms = [
        replace_kv_cache_with_custom_kv_cache,
        replace_sdpa_with_custom_op,
    ]

    if quantize:
        # 8da4w: int8 dynamic activation + int4 weight, group_size=128
        llm_config = LlmConfig()
        llm_config.quantization.qmode = "8da4w"
        llm_config.quantization.group_size = 128
        quant_transform = get_quant_weight_transform(
            quantization_mode=llm_config.quantization.qmode,
            group_size=llm_config.quantization.group_size,
            computation_dtype=dtype,
            checkpoint_path=None,
            tokenizer_path=None,
            calibration_tasks=None,
            calibration_limit=None,
            calibration_seq_length=None,
        )
        _, quantizers, _ = get_quantizer_and_quant_params(llm_config)
        source_transforms.append(quant_transform)
        print("  Applying source transforms (KV cache, SDPA, 8da4w) + tracing...")
        manager = (
            manager.source_transform(source_transforms).export().pt2e_quantize(quantizers)
        )
    else:
        print("  Applying source transforms (KV cache, SDPA) + tracing...")
        manager = manager.source_transform(source_transforms).export()

    with torch.no_grad():
        decoder_ep = torch.export.export(
            manager.pre_autograd_graph_module,
            manager.example_inputs,
            dynamic_shapes=manager._get_dynamic_shape(),
            strict=True,
        )

    return decoder_ep


# ---------------------------------------------------------------------------
# Main — combine all three into one PTE
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Export LFM2-VL to a single multi-method PTE"
    )
    parser.add_argument(
        "--model",
        default="450M",
        choices=list(MODEL_CONFIGS.keys()),
        help="Which LFM2-VL variant to export",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help="Quantize decoder (8da4w) and token embedding (int8); default: fp32",
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16"],
        help="Model dtype (default: fp32)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PTE path (default: lfm2_vl_<model>[_fp16][_quantized]_xnnpack.pte)",
    )
    args = parser.parse_args()

    dtype = DType.fp16 if args.dtype == "fp16" else DType.fp32
    model_cfg = MODEL_CONFIGS[args.model]
    suffix = ("_fp16" if dtype == DType.fp16 else "") + ("_quantized" if args.quantize else "")
    output = args.output or f"lfm2_vl_{args.model}{suffix}_xnnpack.pte"

    print(f"Loading {model_cfg['model_id']} from HuggingFace ({args.dtype})...")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        model_cfg["model_id"], device_map="cpu", torch_dtype=dtype.to_torch_dtype()
    )
    hf_model.eval()

    print("\n[1/3] Vision encoder...")
    vision_ep = export_vision_encoder(hf_model, model_cfg, dtype)

    # Text decoder must run before token embedding: EmbeddingQuantHandler mutates
    # hf_model.model.language_model.embed_tokens in-place (replacing the fp32
    # weight with int8), which would cause load_state_dict to fail in the decoder.
    print("\n[2/3] Text decoder...")
    decoder_ep = export_text_decoder(hf_model, model_cfg, args.quantize, dtype)

    print("\n[3/3] Token embedding...")
    token_ep = export_token_embedding(hf_model, args.quantize)

    print("\nLowering all three methods to Edge IR + XNNPACK...")
    lowered = to_edge_transform_and_lower(
        {
            "vision_encoder": vision_ep,
            "token_embedding": token_ep,
            "text_decoder": decoder_ep,
        },
        partitioner={
            "vision_encoder": [XnnpackPartitioner()],
            "token_embedding": [XnnpackPartitioner()],
            # Quantized: two-pass pattern — DQLinear-only first (avoids holding both
            # packed and unpacked weight buffers in memory), then all remaining ops.
            # fp32/fp16: single partitioner (XNNPACK handles fp16 ops automatically).
            "text_decoder": [
                XnnpackPartitioner(
                    config_precisions=ConfigPrecisionType.DYNAMIC_QUANT,
                    per_op_mode=True,
                ),
                XnnpackPartitioner(),
            ] if args.quantize else [XnnpackPartitioner()],
        },
        constant_methods={
            "get_max_seq_len": MAX_SEQ_LEN,
            # EOS = <|im_end|> (token 7). Exported so the C++ runner
            # reads the correct stop token from the model rather than
            # relying on the tokenizer's eos_tok() default.
            "get_eos_ids": [7],
        },
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )

    print("Finalizing ExecuTorch program...")
    et_program = lowered.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            passes=[QuantFusionPass()] if args.quantize else [],
            sym_shape_eval_pass={
                "vision_encoder": ConstraintBasedSymShapeEvalPass(),
                "token_embedding": HintBasedSymShapeEvalPass(),
                "text_decoder": ConstraintBasedSymShapeEvalPass(),
            },
        )
    )

    print(f"\nSaving {output}...")
    with open(output, "wb") as f:
        et_program.write_to_file(f)

    print(f"\n✅ Saved {output}")
    print(f"   Methods: {et_program.methods}")
    print(f"   Compatible with ExecuTorch MultimodalRunner (C++)")


if __name__ == "__main__":
    main()
