"""
Architecture constants and ModelArgs factory for LFM2.5-VL-1.6B.

This file has no ExecuTorch pipeline imports — just constants and the
ModelArgs config struct. Everything else imports from here.
"""

from executorch.examples.models.llama.model_args import ModelArgs

# ---------------------------------------------------------------------------
# Image / vision constants
# ---------------------------------------------------------------------------

IMAGE_SIZE = 512        # Each tile is 512×512 pixels
PATCH_SIZE = 16         # SigLIP2 patch size (16×16 pixels per patch)
TILE_H = IMAGE_SIZE // PATCH_SIZE   # 32 patches per tile side before downsampling
TILE_W = IMAGE_SIZE // PATCH_SIZE   # 32
DOWNSAMPLE_FACTOR = 2               # Projector halves spatial dims: 32 → 16 per side
TOKENS_PER_TILE = (TILE_H // DOWNSAMPLE_FACTOR) * (TILE_W // DOWNSAMPLE_FACTOR)  # 256

# ---------------------------------------------------------------------------
# Sequence / runtime constants
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048
EOS_TOKEN_ID = 7        # <|im_end|>
BOS_TOKEN_ID = 1        # <|startoftext|>
VOCAB_SIZE = 65536

# ---------------------------------------------------------------------------
# Text backbone — identical to LFM2-VL-1.6B
# ---------------------------------------------------------------------------

# Layer pattern: 10 conv (SSM/ShortConv) + 6 full_attention across 16 layers.
# Indices of attention layers: 2, 5, 8, 10, 11, 13
LAYER_TYPES = [
    "conv", "conv", "full_attention",
    "conv", "conv", "full_attention",
    "conv", "conv", "full_attention",
    "conv", "full_attention",
    "conv", "full_attention",
    "conv", "full_attention",
    "conv",
]

MODEL_DIM = 2048
N_HEADS = 32
N_KV_HEADS = 8
HIDDEN_DIM = 8192       # config.intermediate_size=12288 is pre-SwiGLU → 12288*2/3=8192
N_LAYERS = 16
ROPE_THETA = 1_000_000.0

HF_MODEL_ID = "LiquidAI/LFM2.5-VL-1.6B"


# ---------------------------------------------------------------------------
# ModelArgs factory
# ---------------------------------------------------------------------------

def build_model_args(max_seq_len: int = MAX_SEQ_LEN) -> ModelArgs:
    """
    Returns the ModelArgs config for the LFM2.5-VL-1.6B text backbone.

    Key flags:
      - use_kv_cache=True          : stateful KV cache (required by C++ runner)
      - use_sdpa_with_kv_cache_op  : replaces SDPA with custom ET op (via source transform)
      - enable_dynamic_shape=False : avoids .item() in rope.get_freqs which is not
                                     traceable with FakeTensors; Dim() constraints
                                     still provide dynamic shape support at runtime
      - use_hf_rope=True           : HuggingFace RoPE convention
      - use_qk_norm=True           : QK normalisation before RoPE
    """
    return ModelArgs(
        dim=MODEL_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_kv_heads=N_KV_HEADS,
        vocab_size=VOCAB_SIZE,
        hidden_dim=HIDDEN_DIM,
        ffn_dim_multiplier=1,
        norm_eps=1e-5,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        max_context_len=max_seq_len,
        use_kv_cache=True,
        use_sdpa_with_kv_cache_op=True,
        use_hf_rope=True,
        enable_dynamic_shape=False,
        rope_theta=ROPE_THETA,
        use_qk_norm=True,
        qk_norm_before_rope=True,
        layer_types=LAYER_TYPES,
    )
