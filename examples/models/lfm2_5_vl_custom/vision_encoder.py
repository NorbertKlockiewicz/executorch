"""
Vision encoder export for LFM2.5-VL-1.6B.

Exports a single-tile SigLIP2 ViT + MLP projector as the 'vision_encoder' method.

Input:  [1, 3, 512, 512] float32 pixels in [0, 255] range (NCHW)
Output: [1, 256, 2048]   float32 image embeddings (256 = 16×16 tokens)

Baked into the exported graph:
  - Pixel normalisation:  x = (x / 255 - 0.5) / 0.5
  - Patch extraction:     16×16 patches → [1, 1024, 768] sequence
  - Positional embeddings pre-interpolated for the 32×32 patch grid
  - Vision tower forward (SigLIP2 ViT)
  - MLP projector with 2× spatial downsampling: 32×32 → 16×16 tokens

Multi-tile usage (C++ runner responsibility):
  1. Split the input image into N tiles (thumbnail + up to 10 content tiles)
  2. Call vision_encoder once per tile → [1, 256, 2048]
  3. Concatenate results along dim=1 → [1, N*256, 2048]
  4. Interleave with text token embeddings and pass to text_decoder
"""

import math
import torch
import torch.nn.functional as F
from torch.export import export

from .config import IMAGE_SIZE, PATCH_SIZE, TILE_H, TILE_W, TOKENS_PER_TILE


class VisionEncoderSingleTile(torch.nn.Module):
    """
    Wraps the HF vision tower + projector into a single exportable module.

    Accepts raw NCHW float32 pixels in [0, 255] — same as what the C++ runner
    produces from a decoded image buffer — so the runner needs no preprocessing.

    The FULL_MASK (all patches attended) and spatial_shapes are baked in as
    constants so the exported graph has no variable-length attention mask inputs.
    """

    def __init__(self, vision_tower, projector):
        super().__init__()
        self.vision_tower = vision_tower
        self.projector = projector
        # Constant: all 1024 patches (32×32) are attended
        self.register_buffer(
            "full_mask",
            torch.ones(1, TILE_H * TILE_W, dtype=torch.int32),
        )
        self.register_buffer(
            "spatial_shapes",
            torch.tensor([[TILE_H, TILE_W]], dtype=torch.int64),
        )

    def forward(self, nchw_pixels: torch.Tensor) -> torch.Tensor:
        # nchw_pixels: [1, 3, 512, 512], dtype matches model dtype, range [0, 255]

        # 1. Normalise
        x = nchw_pixels / 255.0
        x = (x - 0.5) / 0.5

        # 2. Extract 16×16 patches → [1, 1024, 768]  (HW-major order)
        x = x.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        x = x.permute(0, 2, 3, 4, 5, 1).reshape(
            1, TILE_H * TILE_W, PATCH_SIZE * PATCH_SIZE * 3
        )

        # 3. Vision tower (SigLIP2 ViT) → [1, 1024, 1152]
        out = self.vision_tower(
            pixel_values=x,
            pixel_attention_mask=self.full_mask,
            spatial_shapes=self.spatial_shapes,
            return_dict=True,
        )
        feats = out.last_hidden_state  # [1, 1024, 1152]

        # 4. Reshape to 2D spatial grid for projector → [1, 32, 32, 1152]
        feats = feats.reshape(feats.shape[0], TILE_H, TILE_W, -1)

        # 5. MLP projector with 2× spatial downsampling → [1, 16, 16, 2048]
        projected = self.projector(feats)

        # 6. Flatten spatial dims → [1, 256, 2048]
        return projected.reshape(1, TOKENS_PER_TILE, projected.shape[-1])


def export_vision_encoder(hf_model, dtype: torch.dtype = torch.float32):
    """
    Export the single-tile vision encoder.

    Steps:
      1. Pre-interpolate positional embeddings for the 32×32 patch grid and
         monkey-patch resize_positional_embeddings so the graph has no dynamic
         interpolation (a fixed constant instead).
      2. Build VisionEncoderSingleTile and call torch.export.export().

    strict=False is required because the HF vision tower has data-dependent
    control flow that strict tracing cannot handle.

    Returns an ExportedProgram ready to be passed to to_edge_transform_and_lower().
    """
    print("  Pre-computing positional embeddings for 32×32 patch grid...")
    orig_pe = (
        hf_model.model.vision_tower
        .vision_model.embeddings
        .position_embedding.weight.data
    )
    n_pos, pe_dim = orig_pe.shape
    sqrt_n = int(math.sqrt(n_pos))
    grid = orig_pe.reshape(sqrt_n, sqrt_n, pe_dim)  # [sqrt_n, sqrt_n, pe_dim]

    resized = F.interpolate(
        grid.permute(2, 0, 1).unsqueeze(0).float(),  # [1, pe_dim, sqrt_n, sqrt_n]
        size=(TILE_H, TILE_W),
        mode="bilinear",
        align_corners=False,
    )
    precomputed_pe = (
        resized.squeeze(0)
        .permute(1, 2, 0)
        .reshape(TILE_H * TILE_W, pe_dim)
        .contiguous()
        .to(dtype)
    )

    # Replace the dynamic resize call with a constant return
    def _patched_resize(positional_embeddings, height=None, width=None, max_length=None):
        return precomputed_pe

    hf_model.model.vision_tower.vision_model.embeddings.resize_positional_embeddings = (
        _patched_resize
    )

    # Build the wrapper module
    encoder = VisionEncoderSingleTile(
        vision_tower=hf_model.model.vision_tower,
        projector=hf_model.model.multi_modal_projector,
    ).eval()

    if dtype != torch.float32:
        encoder = encoder.to(dtype)

    # Example input: random pixels in [0, 255]
    example_pixels = torch.randint(
        0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=dtype
    )

    print("  Tracing single-tile vision encoder (strict=False)...")
    with torch.no_grad():
        ep = export(encoder, (example_pixels,), strict=False)

    print(f"  Vision encoder exported: [1, 3, {IMAGE_SIZE}, {IMAGE_SIZE}] → [1, {TOKENS_PER_TILE}, 2048]")
    return ep
