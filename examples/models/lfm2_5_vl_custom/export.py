"""
LFM2.5-VL-1.6B — clean multi-backend export script.

Produces a single multi-method PTE file with three named methods:
  "vision_encoder"  : [1, 3, 512, 512] f32 pixels → [1, 256, 2048] embeddings
  "token_embedding" : [1, seq_len] i64 token ids   → [1, seq_len, 2048] embeddings
  "text_decoder"    : ([1, seq_len, 2048], [seq_len]) → [1, 65536] logits

Constant methods (readable at runtime via module.get(...)):
  get_max_seq_len   : int = 2048
  get_eos_ids       : List[int] = [7]  (<|im_end|>)
  get_tokens_per_tile : int = 256     (16×16 tokens per image tile)

Usage:
  python -m examples.models.lfm2_5_vl_custom.export --backend xnnpack
  python -m examples.models.lfm2_5_vl_custom.export --backend coreml --dtype fp16
  python -m examples.models.lfm2_5_vl_custom.export --backend qnn --quantize
  python -m examples.models.lfm2_5_vl_custom.export --backend xnnpack --quantize --output my.pte

This file is intentionally thin: all model logic, source transforms, and backend
configuration lives in the other modules. main() is pure orchestration.
"""

import argparse
import torch
from transformers import AutoModelForImageTextToText

from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig
from executorch.exir import to_edge_transform_and_lower
from executorch.exir.passes.quant_fusion_pass import QuantFusionPass
from executorch.exir.passes.sym_shape_eval_pass import (
    ConstraintBasedSymShapeEvalPass,
    HintBasedSymShapeEvalPass,
)

from .config import HF_MODEL_ID, MAX_SEQ_LEN, TOKENS_PER_TILE, EOS_TOKEN_ID
from .vision_encoder import export_vision_encoder
from .token_embedding import export_token_embedding
from .text_decoder import export_text_decoder
from .backends import get_partitioners
from .quantization import get_quantizers, get_weight_quant_transform


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export LFM2.5-VL-1.6B to a multi-method PTE for ExecuTorch inference."
    )
    parser.add_argument(
        "--backend",
        choices=["xnnpack", "coreml", "qnn"],
        default="xnnpack",
        help="Target backend for delegation (default: xnnpack)",
    )
    parser.add_argument(
        "--dtype",
        choices=["fp32", "fp16"],
        default="fp32",
        help="Model compute dtype (default: fp32). Use fp16 for CoreML/ANE.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help=(
            "Quantize the text decoder (8da4w) and token embedding (int8). "
            "For QNN, uses QnnQuantizer instead of XNNPACKQuantizer."
        ),
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=MAX_SEQ_LEN,
        help=f"Maximum sequence length (default: {MAX_SEQ_LEN})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output PTE path (default: auto-generated from backend/dtype/quantize flags)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    # Auto-generate output filename if not specified
    if args.output is None:
        suffix = (
            f"_{'fp16' if dtype == torch.float16 else 'fp32'}"
            + ("_quantized" if args.quantize else "")
        )
        args.output = f"lfm2p5_vl_1.6B{suffix}_{args.backend}.pte"

    print(f"Backend:    {args.backend}")
    print(f"Dtype:      {args.dtype}")
    print(f"Quantize:   {args.quantize}")
    print(f"Max seq:    {args.max_seq_len}")
    print(f"Output:     {args.output}")
    print()

    # ── 1. Load HF model ──────────────────────────────────────────────────
    print(f"Loading {HF_MODEL_ID} from HuggingFace...")
    hf_model = AutoModelForImageTextToText.from_pretrained(
        HF_MODEL_ID,
        device_map="cpu",
        torch_dtype=dtype,
    )
    hf_model.eval()
    print()

    # ── 2. Prepare quantization artefacts (before any export) ─────────────
    quantizers = get_quantizers(args.backend) if args.quantize else []
    quant_transform = get_weight_quant_transform(args.quantize, dtype)

    # ── 3. Export each method ─────────────────────────────────────────────
    # Order matters: text decoder must run before token embedding because
    # EmbeddingQuantHandler (used in token_embedding.py) mutates
    # hf_model.model.language_model.embed_tokens in-place.
    print("[1/3] Exporting vision encoder...")
    vision_ep = export_vision_encoder(hf_model, dtype)
    print()

    print("[2/3] Exporting text decoder...")
    decoder_ep = export_text_decoder(
        hf_model,
        quantize=args.quantize,
        dtype=dtype,
        quantizers=quantizers,
        quant_transform=quant_transform,
    )
    print()

    print("[3/3] Exporting token embedding...")
    token_ep = export_token_embedding(hf_model, quantize=args.quantize)
    print()

    # ── 4. Get backend partitioners ───────────────────────────────────────
    print(f"Getting partitioners for backend: {args.backend}...")
    partitioners = get_partitioners(args.backend, args.quantize)
    print()

    # ── 5. Lower to Edge IR and delegate ──────────────────────────────────
    print("Lowering all methods to Edge IR and delegating to backend...")
    lowered = to_edge_transform_and_lower(
        {
            "vision_encoder":  vision_ep,
            "token_embedding": token_ep,
            "text_decoder":    decoder_ep,
        },
        partitioner=partitioners,
        constant_methods={
            "get_max_seq_len":    args.max_seq_len,
            "get_eos_ids":        [EOS_TOKEN_ID],
            "get_tokens_per_tile": TOKENS_PER_TILE,
        },
        compile_config=EdgeCompileConfig(_check_ir_validity=False),
    )
    print()

    # ── 6. Finalise and serialize ─────────────────────────────────────────
    print("Finalizing ExecuTorch program...")
    et_program = lowered.to_executorch(
        ExecutorchBackendConfig(
            extract_delegate_segments=True,
            # QuantFusionPass fuses quantize/dequantize op pairs that appear
            # adjacent in the graph after PT2E conversion.
            passes=[QuantFusionPass()] if args.quantize else [],
            sym_shape_eval_pass={
                # ConstraintBasedSymShapeEvalPass: evaluates SymInt expressions
                # using Dim() upper bounds — correct for dynamic-shape methods.
                "vision_encoder":  ConstraintBasedSymShapeEvalPass(),
                # HintBasedSymShapeEvalPass: uses concrete example shape as a hint
                # — works for token_embedding where shapes are fully dynamic.
                "token_embedding": HintBasedSymShapeEvalPass(),
                "text_decoder":    ConstraintBasedSymShapeEvalPass(),
            },
        )
    )

    print(f"Saving to {args.output}...")
    with open(args.output, "wb") as f:
        et_program.write_to_file(f)

    print(f"\nDone. Saved: {args.output}")
    print(f"  Methods: {list(et_program.methods)}")
    print(f"  Vision encoder  : [1, 3, 512, 512] → [1, {TOKENS_PER_TILE}, 2048] per tile")
    print(f"  Token embedding : [1, seq_len] → [1, seq_len, 2048]")
    print(f"  Text decoder    : ([1, seq_len, 2048], [seq_len]) → [1, 65536]")
    print()
    print("Multi-tile inference (C++ runner):")
    print("  1. Split image into N tiles (thumbnail + up to 10 content tiles)")
    print(f"  2. Call vision_encoder once per tile → [1, {TOKENS_PER_TILE}, 2048]")
    print(f"  3. Concatenate → [1, N×{TOKENS_PER_TILE}, 2048]")
    print("  4. Interleave with text embeddings, call text_decoder")


if __name__ == "__main__":
    main()
