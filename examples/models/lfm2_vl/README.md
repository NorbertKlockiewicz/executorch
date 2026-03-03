# LFM2-VL ExecuTorch Export

Export and run [LiquidAI/LFM2-VL-450M](https://huggingface.co/LiquidAI/LFM2-VL-450M) entirely in ExecuTorch.

LFM2-VL is a **hybrid SSM+attention vision-language model** — 16 decoder layers alternating between short convolution blocks and full attention blocks. This makes it more complex to export than a pure-attention model like LLaVA, requiring careful weight translation and architecture-aware handling.

---

## Architecture Overview

The model is split into three independent ExecuTorch PTE files:

```
Image (512x512)
    │
    ▼
┌─────────────────────────────────────────┐
│  Vision Encoder  (lfm2_vision_xnnpack.pte) │
│  SigLIP ViT + MLP projector             │
│  in:  pixel_values [1, 1024, 768] f32   │
│  out: image_embeds [256, 1024]   f32    │
└─────────────────────────────────────────┘
    │
    │   Text tokens (before + after image)
    │       │
    │       ▼
    │  ┌──────────────────────────────────────────────┐
    │  │  Token Embedding  (lfm2_token_embedding_xnnpack.pte) │
    │  │  nn.Embedding(65536, 1024)                   │
    │  │  in:  token_ids [1, 2048]        i64         │
    │  │  out: token_embeds [1, 2048, 1024] f32       │
    │  └──────────────────────────────────────────────┘
    │       │
    └───────┤  cat([before_embeds, image_embeds, after_embeds])
            │
            ▼
┌──────────────────────────────────────────────────────────┐
│  Text Decoder  (lfm2_text_decoder_xnnpack_fp32.pte)      │
│  16-layer hybrid transformer (10 conv + 6 attention)     │
│  Internal KV cache + conv state (mutable buffers)        │
│  in:  embeddings [1, seq_len, 1024] f32                  │
│       input_pos  [seq_len]          i64                  │
│  out: logits     [1, 65536]         f32                  │
└──────────────────────────────────────────────────────────┘
            │
            ▼
      Generated tokens (autoregressive decode loop)
```

---

## Quick Start

### Prerequisites

```bash
# Install HuggingFace transformers and Pillow
pip install transformers pillow requests
```

ExecuTorch must be built with XNNPACK and LLM custom ops. The dylibs loaded at runtime are:
- `extension/pybindings/_portable_lib.cpython-310-darwin.so`
- `extension/llm/custom_ops/libcustom_ops_aot_lib.dylib`
- `kernels/quantized/libquantized_ops_aot_lib.dylib`

### Export all three components

```bash
# From repo root
python examples/models/lfm2_vl/export_vision_encoder.py
python examples/models/lfm2_vl/export_token_embedding.py
python examples/models/lfm2_vl/export_text_model.py
```

This produces in the repo root:
- `lfm2_vision_xnnpack.pte`
- `lfm2_token_embedding_xnnpack.pte`
- `lfm2_text_decoder_xnnpack_fp32.pte`

### Run inference

```bash
python run_lfm2.py
```

Requires `dog.jpg` (or any JPEG) in the repo root. Edit `IMAGE_PATH` and `prompt` in `run_lfm2.py` to change inputs.

### Validate with HF decoder (hybrid test)

```bash
python examples/models/lfm2_vl/test_et_vision_embed_hf_decoder.py
```

Runs ET vision encoder + ET token embedding, but decodes with the HF PyTorch model. Useful for isolating decoder issues.

---

## Export Details

### 1. Vision Encoder — `export_vision_encoder.py`

**What it exports:** SigLIP ViT backbone + MLP projector (patch size 16, image size 512).

**Interface:**
```
forward(pixel_values: [1, 1024, 768] f32) -> image_embeds: [256, 1024] f32
```

**Key decisions:**

**Fixed 512×512 input.** The HF vision tower supports variable-resolution images using dynamic positional embedding interpolation. This is not exportable — it involves conditional logic and runtime-computed shapes. The fix is to pre-compute positional embeddings for exactly 32×32 patches (512px / 16px-per-patch) and patch the `resize_positional_embeddings` method to return them as a constant.

**Full attention mask baked in.** At 512×512, all 1024 patches are valid, so `pixel_attention_mask` is always all-ones. Baking it as a constant avoids boolean/conditional ops that XNNPACK cannot handle.

**`_skip_dim_order=True`.** The vision tower has residual additions between tensors with mismatched memory layouts. Setting this flag prevents `dim_order_ops` insertions that would cause `aten::add.out` dim-order mismatches in the portable kernel.

**`.contiguous()` on positional embeddings.** After bilinear interpolation and reshape, the tensor may be non-contiguous. This causes dim-order errors during export. Calling `.contiguous()` before patching it in avoids this.

**Output shape:** The MLP projector downsamples 1024 patches → 256 patch tokens (2×2 pooling), each with 1024 dimensions.

---

### 2. Token Embedding — `export_token_embedding.py`

**What it exports:** `nn.Embedding(65536, 1024)` — the token lookup table from the language model.

**Interface:**
```
forward(token_ids: [1, 2048] i64) -> token_embeds: [1, 2048, 1024] f32
```

**Key decisions:**

**Fixed-length buffer (2048).** The embedding layer is exported with a fixed `MAX_SEQ_LEN=2048` input. At runtime, shorter sequences are padded with zeros before the call, and the output is sliced back to the real length. This avoids dynamic shapes entirely for a component that doesn't need them.

**`strict=True`.** Unlike the vision encoder and decoder, the embedding layer has no data-dependent control flow or custom ops. Standard strict tracing works fine.

**Why separate from the decoder?** The decoder's `tok_embeddings` weight is inside the text decoder PTE, but having a standalone embedding PTE allows the runner to embed arbitrary token sequences independently — needed for both the initial prompt and each decode step without running the full decoder forward pass just to look up an embedding.

---

### 3. Text Decoder — `export_text_model.py`

**What it exports:** The 16-layer hybrid LFM2 transformer with KV cache and conv state.

**Interface:**
```
forward(embeddings: [1, seq_len, 1024] f32,
        input_pos:  [seq_len] i64)
     -> logits: [1, 65536] f32
```
`seq_len` is dynamic (1 to 2048) — any length works at runtime without recompilation.

**Key decisions:**

**Do not wrap the HF model directly.** The HF `Lfm2Model` uses `DynamicCache` which grows dynamically and is not exportable. Instead, we use the existing ET LFM2 infrastructure: `construct_transformer` from `examples/models/llama/` with `layer_types` to build a statically-structured hybrid model with pre-allocated KV cache buffers.

**LFM2 hybrid layer layout.** LFM2-VL-450M has 16 layers with this pattern:
```python
LAYER_TYPES = [
    "conv", "conv", "full_attention",   # layers 0-2
    "conv", "conv", "full_attention",   # layers 3-5
    "conv", "conv", "full_attention",   # layers 6-8
    "conv", "full_attention",           # layers 9-10
    "conv", "full_attention",           # layers 11-12
    "conv", "full_attention",           # layers 13-14
    "conv",                             # layer 15
]
```
`"conv"` layers use `ShortConvBlock` (from `examples/models/lfm2/short_conv.py`), `"full_attention"` layers use the standard `TransformerBlock`.

**Weight translation.** The HF and ET weight naming conventions differ significantly. `translate_weights()` handles:

| HF key | ET key | Notes |
|--------|--------|-------|
| `embed_tokens.weight` | `tok_embeddings.weight` | rename |
| `embedding_norm.weight` | `norm.weight` | rename |
| `lm_head.weight` | `output.weight` | from `hf_model.lm_head` (separate module) |
| `layers.N.operator_norm.weight` | `layers.N.attention_norm.weight` | rename (both layer types) |
| `layers.N.conv.in_proj.weight` [3072, 1024] | `layers.N.conv.{B,C,x}_proj.weight` [1024, 1024] each | **split into 3** |
| `layers.N.self_attn.{q,k,v,out}_proj.weight` | `layers.N.attention.w{q,k,v,o}.weight` | rename |
| `layers.N.self_attn.{q,k}_layernorm.weight` | `layers.N.attention.{q,k}_norm_fn.weight` | rename |

The `in_proj` split is the most critical: HF stores a single `[3*dim, dim]` matrix that ET's `ShortConvBlock` expects as three separate `[dim, dim]` projections (`B_proj`, `C_proj`, `x_proj`).

**`enable_dynamic_shape=False`.** This controls a branch in `rope.get_freqs`:
- `True` → uses `input_pos[-1].item()` + `torch.narrow()` — the `.item()` call converts a `FakeTensor` to a Python int during `run_decompositions`, which crashes with `Cannot cast FakeTensor to number`
- `False` → uses `self.freqs_cos[input_pos]` (direct tensor indexing) — fully traceable

With `False`, dynamic shapes still work via `Dim()` constraints on the input. No runtime padding is needed.

**Do not pad prefill inputs.** With `enable_dynamic_shape=False`, the model uses `h[:, -1, :]` to extract last-token logits. If you pad the prefill sequence to `MAX_SEQ_LEN`, position `-1` indexes a zero-padded slot, producing garbage output. Pass the unpadded `[1, actual_seq_len, 1024]` tensor directly.

**Source transforms.** Applied before tracing:
- `replace_kv_cache_with_custom_kv_cache` — swaps the standard KV cache for the ET custom cache that uses `llama::update_cache` ops (write to pre-allocated buffers in-place)
- `replace_sdpa_with_custom_op` — swaps `F.scaled_dot_product_attention` for `llama::custom_sdpa` (mask-aware, works with the custom KV cache)

**`strict=False`.** The hybrid model has buffer mutations (`conv_state.copy_()`) and custom ops that the strict tracer can't handle. Non-strict tracing with `torch.export` works correctly.

**Mutable state (KV cache + conv state).** Both the attention KV caches and the convolution states are `register_buffer` buffers inside the model, mutated in-place via `copy_()` each forward pass. They persist between `.forward()` calls on the same loaded PTE instance — this is how the decoder maintains state across the prefill and decode steps. The export warning `Mutation on a buffer detected` is expected and intentional.

**ModelArgs hyperparameters:**
```python
ModelArgs(
    dim=1024, n_layers=16, n_heads=16, n_kv_heads=8,
    vocab_size=65536, hidden_dim=4608, ffn_dim_multiplier=1,
    norm_eps=1e-5, max_batch_size=1, max_seq_len=2048, max_context_len=2048,
    use_kv_cache=True, use_sdpa_with_kv_cache_op=True,
    use_hf_rope=True, enable_dynamic_shape=False,
    rope_theta=1000000.0, use_qk_norm=True, qk_norm_before_rope=True,
    layer_types=LAYER_TYPES,
)
```

---

## Inference Flow — `run_lfm2.py`

```python
# 1. Vision: pixel_values → image_embeds [1, 256, 1024]
image_embeds = vision_module.forward([pixel_values])[0].unsqueeze(0)

# 2. Embed text tokens on each side of the image placeholder
before_embeds = embed_tokens(embed_module, before_ids)   # [1, N, 1024]
after_embeds  = embed_tokens(embed_module, after_ids)    # [1, M, 1024]

# 3. Stitch into one sequence
combined = torch.cat([before_embeds, image_embeds, after_embeds], dim=1)

# 4. Prefill — pass unpadded combined embeddings directly
prefill_pos = torch.arange(seq_len, dtype=torch.int64)
logits = run_decoder(decoder_module, combined, prefill_pos)
next_token_id = logits.argmax(dim=-1).item()

# 5. Decode loop — one token at a time, KV cache accumulates internally
for step in range(MAX_NEW_TOKENS):
    next_embed = embed_tokens(embed_module, [[next_token_id]])  # [1, 1, 1024]
    next_pos   = torch.tensor([cur_pos])
    logits     = run_decoder(decoder_module, next_embed, next_pos)
    next_token_id = logits.argmax(dim=-1).item()
    cur_pos += 1
```

**`embed_tokens` helper** pads input to `MAX_SEQ_LEN=2048` (the fixed PTE buffer size), calls the embedding PTE, then slices the output back to the real length.

**`run_decoder` helper** calls the decoder PTE directly with unpadded inputs. No padding logic needed — dynamic shapes handle variable `seq_len` at runtime.

---

## Files

| File | Purpose |
|------|---------|
| `export_vision_encoder.py` | Export SigLIP ViT + projector |
| `export_token_embedding.py` | Export token embedding table |
| `export_text_model.py` | Export hybrid LFM2 decoder |
| `test_et_vision_embed_hf_decoder.py` | Hybrid validation (ET vision+embed, HF decoder) |
| `run_lfm2.py` (repo root) | Full ET inference runner |

---

## Troubleshooting

**`Cannot cast FakeTensor to number` during export**
Set `enable_dynamic_shape=False` in `ModelArgs`. The `True` path uses `.item()` which is not traceable with FakeTensors.

**Korean/garbage tokens at start of generation**
Do not pad prefill inputs to `MAX_SEQ_LEN`. Pass the unpadded combined embeddings directly. Padding causes `h[:, -1, :]` to slice from a zero-padded position.

**`strict=True` export fails for decoder**
Use `strict=False`. The hybrid model has buffer mutations and custom ops that the strict tracer cannot handle.

**Missing keys in `load_state_dict`**
The 22 missing keys are expected: KV cache buffers (`k_cache`, `v_cache`) and conv states (`conv_state`) are initialized as zeros by `construct_transformer` and are not loaded from HF weights — they are runtime state, not model parameters.

**`AttributeError: 'Lfm2VlForConditionalGeneration' has no attribute 'language_model'`**
Use `hf_model.model.language_model` (not `hf_model.language_model`) and `hf_model.lm_head` separately.
