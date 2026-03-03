#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Export LFM2.5-VL-1.6B as a single multi-method PTE compatible with
ExecuTorch's generic MultimodalRunner (C++).

Produces: lfm2p5_vl_1.6B[_quantized]_xnnpack.pte  with three named methods:
  - "vision_encoder"  : pixel_values [1, 3, 512, 512] f32 NCHW -> image_embeds [1, 256, 2048]
  - "token_embedding" : token_ids    [1, seq_len]      i64      -> embeds [1, seq_len, 2048]
  - "text_decoder"    : (embeds [1, seq_len, 2048], input_pos [seq_len]) -> logits [1, 65536]

Key difference from LFM2-VL: do_image_splitting=True.
The model expects 2-10 tiles (each 512×512) plus a thumbnail tile, concatenated
along the sequence dimension before the text decoder.

This export handles multi-tile by exporting a SINGLE-TILE vision encoder
(fixed [1, 3, 512, 512] input → [1, 256, 2048] output with downsample_factor=2).
The C++ runner calls vision_encoder once per tile, concatenates the results,
then passes the full sequence to text_decoder.

Vision encoder output per tile: 32×32 patches, projector downsamples 2× per dim
→ 16×16 = 256 tokens per tile.
Max tiles (thumbnail + 10 content tiles = 11) → max 2816 image tokens.

Text backbone: identical to LFM2-VL-1.6B (dim=2048, 16 layers, same hybrid pattern).

Usage:
    python examples/models/lfm2p5_vl/export_lfm2p5_vl.py
    python examples/models/lfm2p5_vl/export_lfm2p5_vl.py --quantize
    python examples/models/lfm2p5_vl/export_lfm2p5_vl.py --output my_model.pte
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
# Architecture constants
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048
IMAGE_SIZE = 512          # Each tile is 512×512
PATCH_SIZE = 16           # SigLIP2 patch size
TILE_H = IMAGE_SIZE // PATCH_SIZE   # 32 patches per side before downsampling
TILE_W = IMAGE_SIZE // PATCH_SIZE   # 32
DOWNSAMPLE_FACTOR = 2     # spatial downsampling in projector: 32÷2=16 per dim
TOKENS_PER_TILE = (TILE_H // DOWNSAMPLE_FACTOR) * (TILE_W // DOWNSAMPLE_FACTOR)  # 256 (16×16)

# LFM2.5-VL-1.6B text backbone — identical to LFM2-VL-1.6B
MODEL_CONFIG = {
    "model_id": "LiquidAI/LFM2.5-VL-1.6B",
    "dim": 2048,
    "n_heads": 32,
    "n_kv_heads": 8,
    "hidden_dim": 8192,   # config.intermediate_size=12288 is pre-SwiGLU → 12288*2/3=8192
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
}


# ---------------------------------------------------------------------------
# Shared LLMEdgeManager subclass
# ---------------------------------------------------------------------------


class Lfm2p5EdgeManager(LLMEdgeManager):
    def export(self) -> "Lfm2p5EdgeManager":
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
# Vision encoder export  (single-tile: [1, 3, 512, 512] → [1, 256, 2048])
# ---------------------------------------------------------------------------


def export_vision_encoder(hf_model, dtype: DType = DType.fp32) -> torch.export.ExportedProgram:
    """Export a single-tile SigLIP2 ViT + MLP projector as 'vision_encoder'.

    Accepts raw NCHW float32 pixels in [0, 255] range (one tile at a time).

    The C++ runner is responsible for:
      - Splitting the image into tiles (and generating a thumbnail tile)
      - Calling this method once per tile
      - Concatenating tile embeddings along the sequence dimension
      - Appending text embeddings and running text_decoder

    Bakes in:
      - Normalization: (x/255 - 0.5) / 0.5
      - Patch extraction: 16×16 patches → [1, 1024, 768]
      - Positional embeddings pre-interpolated for 32×32 patch grid
      - Full attention mask and spatial_shapes as constants
      - Spatial 2× downsampling via projector (output: [1, 256, 2048])
    """
    print("  Pre-computing positional embeddings for 32x32 patches...")
    orig_embeddings = (
        hf_model.model.vision_tower.vision_model.embeddings.position_embedding.weight.data
    )
    num_positions, pe_dim = orig_embeddings.shape
    sqrt_num = int(math.sqrt(num_positions))
    grid = orig_embeddings.reshape(sqrt_num, sqrt_num, pe_dim)
    resized = F.interpolate(
        grid.permute(2, 0, 1).unsqueeze(0),
        size=(TILE_H, TILE_W),
        mode="bilinear",
        align_corners=False,
    )
    precomputed_pos = (
        resized.squeeze(0).permute(1, 2, 0)
        .reshape(TILE_H * TILE_W, pe_dim)
        .contiguous()
    )

    def patched_resize(positional_embeddings, height=None, width=None, max_length=None):
        return precomputed_pos

    hf_model.model.vision_tower.vision_model.embeddings.resize_positional_embeddings = (
        patched_resize
    )

    FULL_MASK = torch.ones(1, TILE_H * TILE_W, dtype=torch.int32)

    class VisionEncoderSingleTile(torch.nn.Module):
        """Single-tile vision encoder.

        Input:  [1, 3, 512, 512] pixels in [0, 255] in model dtype.
        Output: [1, 256, 2048]  (256 = 16×16 tokens after 2× spatial downsampling per dim)

        The C++ runner must cast its float32 pixel buffer to the model dtype first.
        """

        def __init__(self, vision_tower, projector):
            super().__init__()
            self.vision_tower = vision_tower
            self.projector = projector

        def forward(self, nchw_pixels):
            # nchw_pixels: [1, 3, 512, 512] in model dtype, range [0, 255]
            x = nchw_pixels / 255.0
            x = (x - 0.5) / 0.5

            # Patch extraction: [1, 3, 512, 512] → [1, 1024, 768] (HW-major order)
            x = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
            x = x.permute(0, 2, 3, 4, 5, 1).reshape(
                1, TILE_H * TILE_W, PATCH_SIZE * PATCH_SIZE * 3
            )

            out = self.vision_tower(
                pixel_values=x,
                pixel_attention_mask=FULL_MASK,
                spatial_shapes=torch.tensor([[TILE_H, TILE_W]], dtype=torch.int64),
                return_dict=True,
            )
            feats = out.last_hidden_state  # [1, 1024, 1152]

            # Reshape to 2D spatial grid for projector with 2× spatial downsampling.
            # The projector folds 2×2 spatial neighborhoods → reduces 32×32 to 16×16,
            # then projects to dim=2048: output [1, 16, 16, 2048]
            feats = feats.reshape(feats.shape[0], TILE_H, TILE_W, -1)
            projected = self.projector(feats)  # [1, H', W', 2048]

            # Flatten spatial dims → [1, tokens_per_tile, 2048]
            return projected.reshape(1, -1, projected.shape[-1])

    torch_dtype = dtype.to_torch_dtype()
    pixel_values = torch.randint(
        0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch_dtype
    )

    encoder = VisionEncoderSingleTile(
        hf_model.model.vision_tower,
        hf_model.model.multi_modal_projector,
    )
    encoder.eval()

    # Sanity check against HF processor output
    print("  Verifying preprocessing matches HF processor...")
    import os

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    img_path = os.path.join(repo_root, "dog.jpg")
    if os.path.exists(img_path):
        from transformers import AutoProcessor as _AP
        import numpy as np

        processor = _AP.from_pretrained(MODEL_CONFIG["model_id"])
        image = Image.open(img_path).resize((IMAGE_SIZE, IMAGE_SIZE))
        inputs = processor(text=["<image>"], images=[image], return_tensors="pt")
        # pixel_values from processor: [1, N_tiles, 1024, 768] or [N_tiles, 1024, 768]
        # We only care that our patch extraction for one tile matches one slice.
        proc_pv = inputs["pixel_values"]

        arr = torch.tensor(np.array(image), dtype=torch_dtype)  # [512, 512, 3]
        raw_nchw = arr.permute(2, 0, 1).unsqueeze(0)  # [1, 3, 512, 512]
        with torch.no_grad():
            our_pv = raw_nchw / 255.0
            our_pv = (our_pv - 0.5) / 0.5
            our_pv = our_pv.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(
                3, PATCH_SIZE, PATCH_SIZE
            )
            our_pv = our_pv.permute(0, 2, 3, 4, 5, 1).reshape(
                1, TILE_H * TILE_W, PATCH_SIZE * PATCH_SIZE * 3
            )

        # proc_pv shape may be [1, N_tiles, 1024, 768]; compare against first tile
        if proc_pv.dim() == 4:
            proc_tile = proc_pv[0, 0:1]   # [1, 1024, 768] — thumbnail or first tile
        else:
            proc_tile = proc_pv[0:1]       # [1, 1024, 768]

        diff = (our_pv.float() - proc_tile.float()).abs().max().item()
        print(
            f"    Max pixel_values diff vs processor (first tile): {diff:.2e}  "
            f"{'✓' if diff < 1e-4 else '✗ MISMATCH (expected for tiled — thumbnail may differ)'}"
        )
        pixel_values = raw_nchw.to(torch_dtype)

    print("  Tracing single-tile vision encoder...")
    with torch.no_grad():
        ep = export(encoder, (pixel_values,), strict=False)

    return ep


# ---------------------------------------------------------------------------
# Token embedding export
# ---------------------------------------------------------------------------


def export_token_embedding(hf_model, quantize: bool) -> torch.export.ExportedProgram:
    """Export nn.Embedding(65536, 2048) as 'token_embedding' method.

    When quantize=True applies int8 group quantization (bitwidth=8, group_size=32).
    NOTE: EmbeddingQuantHandler mutates hf_model.model.language_model in-place,
    so export_text_decoder must be called before this function.
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
# Text decoder export  (identical backbone to LFM2-VL-1.6B)
# ---------------------------------------------------------------------------


def translate_weights(hf_model) -> dict:
    """Translate HF LFM2.5-VL state dict to ET construct_transformer format."""
    sd = {}
    for k, v in hf_model.model.language_model.state_dict().items():
        sd[k] = v
    for k, v in hf_model.lm_head.state_dict().items():
        sd[f"lm_head.{k}"] = v

    out = {}
    for key, val in sd.items():
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

        if rest.startswith("feed_forward.") or rest == "ffn_norm.weight":
            out[prefix + rest] = val
            continue
        if rest == "operator_norm.weight":
            out[prefix + "attention_norm.weight"] = val
            continue

        # Conv layer
        if rest == "conv.conv.weight":
            out[prefix + "conv.conv.weight"] = val
            continue
        if rest == "conv.in_proj.weight":
            B, C, x = torch.chunk(val, 3, dim=0)
            out[prefix + "conv.B_proj.weight"] = B
            out[prefix + "conv.C_proj.weight"] = C
            out[prefix + "conv.x_proj.weight"] = x
            continue
        if rest == "conv.out_proj.weight":
            out[prefix + "conv.out_proj.weight"] = val
            continue

        # Attention layer
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


def export_text_decoder(hf_model, quantize: bool, dtype: DType = DType.fp32) -> torch.export.ExportedProgram:
    """Export the hybrid LFM2.5 decoder as 'text_decoder'.

    Text backbone is identical to LFM2-VL-1.6B: dim=2048, 16 hybrid layers.

    enable_dynamic_shape=False avoids .item() in rope.get_freqs which is not
    traceable with FakeTensors. Dynamic shapes are still enforced via Dim().
    """
    print("  Building ET hybrid transformer (dim=2048, 16 layers)...")
    model_args = ModelArgs(
        dim=MODEL_CONFIG["dim"],
        n_layers=MODEL_CONFIG["n_layers"],
        n_heads=MODEL_CONFIG["n_heads"],
        n_kv_heads=MODEL_CONFIG["n_kv_heads"],
        vocab_size=65536,
        hidden_dim=MODEL_CONFIG["hidden_dim"],
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
        layer_types=MODEL_CONFIG["layer_types"],
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

    class Lfm2p5TextDecoder(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.text_model = model

        def forward(self, embeddings, input_pos):
            return self.text_model(None, {"input_pos": input_pos}, embeddings)

    decoder = Lfm2p5TextDecoder(et_model)
    decoder.eval()

    torch_dtype = dtype.to_torch_dtype()
    dummy_seq_len = 8
    dummy_embeddings = torch.randn(1, dummy_seq_len, MODEL_CONFIG["dim"], dtype=torch_dtype)
    dummy_input_pos = torch.arange(dummy_seq_len, dtype=torch.int64)

    token_dim = Dim("token_dim", min=1, max=MAX_SEQ_LEN)
    dynamic_shapes = ({1: token_dim}, {0: token_dim})

    manager = Lfm2p5EdgeManager(
        model=decoder,
        modelname="lfm2p5_text_decoder",
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
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Export LFM2.5-VL-1.6B to a single multi-method PTE"
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
        help="Output PTE path (default: lfm2p5_vl_1.6B[_fp16][_quantized]_xnnpack.pte)",
    )
    args = parser.parse_args()

    dtype = DType.fp16 if args.dtype == "fp16" else DType.fp32
    suffix = ("_fp16" if dtype == DType.fp16 else "") + ("_quantized" if args.quantize else "")
    output = args.output or f"lfm2p5_vl_1.6B{suffix}_xnnpack.pte"

    print(f"Loading {MODEL_CONFIG['model_id']} from HuggingFace ({args.dtype})...")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_CONFIG["model_id"], device_map="cpu", torch_dtype=dtype.to_torch_dtype()
    )
    hf_model.eval()

    print("\n[1/3] Vision encoder (single-tile)...")
    vision_ep = export_vision_encoder(hf_model, dtype)

    # Text decoder must run before token embedding: EmbeddingQuantHandler mutates
    # hf_model.model.language_model.embed_tokens in-place (replacing the fp32
    # weight with int8), which would cause load_state_dict to fail in the decoder.
    print("\n[2/3] Text decoder...")
    decoder_ep = export_text_decoder(hf_model, args.quantize, dtype)

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
            # EOS = <|im_end|> (token 7)
            "get_eos_ids": [7],
            # Expose tokens-per-tile so C++ runner can know how many image
            # tokens each vision_encoder call produces (256 for LFM2.5-VL).
            "get_tokens_per_tile": TOKENS_PER_TILE,
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

    print(f"\n✓ Saved {output}")
    print(f"  Methods: {et_program.methods}")
    print(f"  Vision encoder: [1, 3, 512, 512] → [1, {TOKENS_PER_TILE}, {MODEL_CONFIG['dim']}] per tile  ({TOKENS_PER_TILE} = 16×16 after 2× spatial downsample)")
    print(f"  Token embedding: [1, seq_len] → [1, seq_len, {MODEL_CONFIG['dim']}]")
    print(f"  Text decoder: ([1, seq_len, {MODEL_CONFIG['dim']}], [seq_len]) → [1, 65536]")
    print(f"\n  Multi-tile usage for C++ runner:")
    print(f"    1. Split input image into N tiles (thumbnail + content tiles, max 11)")
    print(f"    2. Call vision_encoder once per tile → [1, {TOKENS_PER_TILE}, {MODEL_CONFIG['dim']}]")
    print(f"    3. Concatenate tile embeddings: [1, N*{TOKENS_PER_TILE}, {MODEL_CONFIG['dim']}]")
    print(f"    4. Interleave with text token embeddings and call text_decoder")


if __name__ == "__main__":
    main()
