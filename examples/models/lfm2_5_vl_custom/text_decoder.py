"""
Text decoder export for LFM2.5-VL-1.6B.

Exports the hybrid LFM2.5 text backbone as the 'text_decoder' method.
The backbone is identical to LFM2-VL-1.6B: dim=2048, 16 hybrid layers
(10 ShortConv/SSM + 6 full-attention).

Input:  embeddings [1, seq_len, 2048] float  +  input_pos [seq_len] int64
Output: logits     [1, 65536]          float

Pipeline (no LLMEdgeManager):
  1. build_et_model()       — construct_transformer + load_state_dict + dtype cast
  2. apply_source_transforms() — KV cache custom op + SDPA custom op (+ quant transform)
  3. export_text_decoder()  — wrap in TextDecoderWrapper, call torch.export, optional PT2E quant

Why enable_dynamic_shape=False in ModelArgs:
  True would use .item() inside rope.get_freqs to cast a SymInt → Python int.
  .item() on a FakeTensor raises "Cannot cast FakeTensor to number" during
  to_edge_transform_and_lower. With False the model uses tensor indexing
  (self.freqs_cos[input_pos]) which is fully exportable. Dynamic shapes are
  still supported at runtime via Dim() constraints.

Why dtype cast BEFORE source transforms:
  ShortConv registers conv_state as a float32 buffer by default.
  replace_kv_cache_with_custom_kv_cache runs after construction, so if we cast
  after the transform the conv_state stays float32 while Conv1d weights are
  float16 → dtype mismatch in the forward pass. Casting the whole model first
  ensures all parameters AND buffers match.
"""

import torch
from torch.export import Dim, export
from torch.nn.attention import SDPBackend, sdpa_kernel

from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    replace_kv_cache_with_custom_kv_cache,
)
from executorch.examples.models.llama.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
from executorch.examples.models.llama.source_transformation.quantize import (
    get_quant_weight_transform,
)

from .config import MAX_SEQ_LEN, MODEL_DIM, build_model_args
from .weights import translate_weights


# ---------------------------------------------------------------------------
# TextDecoderWrapper
# ---------------------------------------------------------------------------

class TextDecoderWrapper(torch.nn.Module):
    """
    Thin wrapper that adapts the ET model's calling convention to the
    (embeddings, input_pos) signature expected by the C++ runner.

    The ET model's forward signature is:
        forward(tokens, kwargs_dict, embeddings)
    where tokens=None signals that pre-computed embeddings should be used
    and input_pos is passed via kwargs_dict.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.text_model = model

    def forward(self, embeddings: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        return self.text_model(None, {"input_pos": input_pos}, embeddings)


# ---------------------------------------------------------------------------
# Step 1: Build the ET model
# ---------------------------------------------------------------------------

def build_et_model(hf_model, dtype: torch.dtype = torch.float32) -> torch.nn.Module:
    """
    Construct the ET hybrid transformer, load HF weights, and cast to dtype.

    The dtype cast happens here — before source transforms — so that all
    buffers (conv_state, kv_cache) are in the correct dtype from the start.
    """
    print("  Building ET hybrid transformer (dim=2048, 16 layers)...")
    model_args = build_model_args()
    et_model = construct_transformer(model_args)

    print("  Translating and loading weights...")
    state_dict = translate_weights(hf_model)
    missing, unexpected = et_model.load_state_dict(state_dict, strict=False, assign=True)
    print(
        f"  Loaded {len(state_dict)} tensors — "
        f"{len(missing)} missing (expected: KV/conv buffers), "
        f"{len(unexpected)} unexpected"
    )

    if dtype != torch.float32:
        print(f"  Casting model to {dtype}...")
        et_model = et_model.to(dtype)

    return et_model


# ---------------------------------------------------------------------------
# Step 2: Apply source transforms
# ---------------------------------------------------------------------------

def apply_source_transforms(
    et_model: torch.nn.Module,
    quantize: bool = False,
    quant_transform=None,
) -> torch.nn.Module:
    """
    Apply the two mandatory source transforms for ET export:

    1. replace_kv_cache_with_custom_kv_cache
       Replaces the nn.Module KV cache with torch.ops.llama.update_cache —
       a custom C++ op. This gives backends direct visibility into the KV
       update pattern and avoids exporting the full cache update as generic
       tensor ops.

    2. replace_sdpa_with_custom_op
       Replaces F.scaled_dot_product_attention with torch.ops.llama.sdpa_with_kv_cache —
       another custom op that fuses the KV cache read + attention computation.
       Required when use_sdpa_with_kv_cache_op=True in ModelArgs.

    If quantize=True and quant_transform is provided, the weight quantization
    transform (e.g. 8da4w) is applied after the above two.
    """
    print("  Applying source transform: KV cache custom op...")
    et_model = replace_kv_cache_with_custom_kv_cache(et_model)

    print("  Applying source transform: SDPA custom op...")
    et_model = replace_sdpa_with_custom_op(et_model)

    if quantize and quant_transform is not None:
        print("  Applying source transform: 8da4w weight quantization...")
        et_model = quant_transform(et_model)

    return et_model


# ---------------------------------------------------------------------------
# Step 3: Export
# ---------------------------------------------------------------------------

def export_text_decoder(
    hf_model,
    quantize: bool = False,
    dtype: torch.dtype = torch.float32,
    quantizers=None,
    quant_transform=None,
):
    """
    Build → source transforms → torch.export → optional PT2E quantization.

    Dynamic shapes:
      Both the embedding sequence length and input_pos length are dynamic,
      constrained to [1, MAX_SEQ_LEN] via Dim(). This covers both:
        - Prefill: seq_len = full prompt length
        - Decode:  seq_len = 1

    strict=False is required because the hybrid model has data-dependent
    control flow in the SSM conv blocks.

    Returns an ExportedProgram.
    """
    et_model = build_et_model(hf_model, dtype)
    et_model = apply_source_transforms(et_model, quantize, quant_transform)

    decoder = TextDecoderWrapper(et_model).eval()

    # Example inputs (small seq_len for fast tracing)
    dummy_seq_len = 8
    dummy_embeddings = torch.randn(1, dummy_seq_len, MODEL_DIM, dtype=dtype)
    dummy_input_pos = torch.arange(dummy_seq_len, dtype=torch.int64)

    token_dim = Dim("token_dim", min=1, max=MAX_SEQ_LEN)
    dynamic_shapes = ({1: token_dim}, {0: token_dim})

    print("  Tracing text decoder (strict=False)...")
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH]):
        ep = export(
            decoder,
            (dummy_embeddings, dummy_input_pos),
            dynamic_shapes=dynamic_shapes,
            strict=False,
        )

    # PT2E quantization: prepare → calibrate → convert → re-export
    if quantize and quantizers:
        from torchao.quantization.pt2e import prepare_pt2e, convert_pt2e

        print("  Applying PT2E quantization (prepare → convert)...")
        ep_module = ep.module()
        ep_module = prepare_pt2e(ep_module, quantizers[0])
        # NOTE: add calibration data here for better accuracy
        ep_module = convert_pt2e(ep_module)

        print("  Re-exporting quantized decoder (strict=True)...")
        with torch.no_grad():
            ep = export(
                ep_module,
                (dummy_embeddings, dummy_input_pos),
                dynamic_shapes=dynamic_shapes,
                strict=True,
            )

    print(
        f"  Text decoder exported: "
        f"([1, seq_len, {MODEL_DIM}], [seq_len]) → [1, 65536]"
    )
    return ep
