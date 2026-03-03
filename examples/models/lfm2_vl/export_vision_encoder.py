#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from torch.export import export
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
import requests
import math
import torch.nn.functional as F

# --- XNNPACK / ExecuTorch Imports ---
from executorch.exir import to_edge_transform_and_lower, EdgeCompileConfig, ExecutorchBackendConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


def main():
    print("Loading LFM2-VL model...")
    model = AutoModelForImageTextToText.from_pretrained(
        "LiquidAI/LFM2-VL-450M",
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    # --- Step 1: Pre-compute fixed positional embeddings for 32x32 patches ---
    # Use 512x512 input so all 1024 patches are valid (no masking needed).
    # FIXED_H/W=32 because 512px / 16px-per-patch = 32.
    print("Pre-computing positional embeddings for 32x32 patches...")
    FIXED_H, FIXED_W = 32, 32

    orig_embeddings = (
        model.model.vision_tower.vision_model.embeddings.position_embedding.weight.data
    )
    num_positions, dim = orig_embeddings.shape
    sqrt_num = int(math.sqrt(num_positions))
    embeddings_grid = orig_embeddings.reshape(sqrt_num, sqrt_num, dim)

    embeddings_4d = embeddings_grid.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(
        embeddings_4d, size=(FIXED_H, FIXED_W), mode="bilinear", align_corners=False
    )
    # .contiguous() is critical: prevents non-contiguous strides that cause
    # dim_order mismatches in the ExecuTorch portable kernel for aten::add.out
    precomputed_embeddings = resized.squeeze(0).permute(1, 2, 0).reshape(FIXED_H * FIXED_W, dim).contiguous()

    def patched_resize(positional_embeddings, height=None, width=None, max_length=None):
        return precomputed_embeddings

    model.model.vision_tower.vision_model.embeddings.resize_positional_embeddings = (
        patched_resize
    )

    # --- Step 2: Prepare Input ---
    processor = AutoProcessor.from_pretrained("LiquidAI/LFM2-VL-450M")

    # 512x512 gives spatial_shapes=[32,32] matching FIXED_H/W, and all 1024 patches valid.
    target_res = 512
    print(f"Loading example image (resized to {target_res}x{target_res})...")
    image_url = "https://llava-vl.github.io/static/images/view.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw).resize(
        (target_res, target_res)
    )

    print("Processing image...")
    inputs = processor(text=["<image>"], images=[image], return_tensors="pt")
    print(f"  pixel_values: {inputs['pixel_values'].shape}")
    print(f"  spatial_shapes: {inputs['spatial_shapes'].tolist()} (should be [[32,32]])")

    # Full attention mask (all 1024 patches valid for 512x512).
    # Using a constant mask avoids boolean/conditional ops that XNNPACK cannot handle.
    FULL_MASK = torch.ones(1, FIXED_H * FIXED_W, dtype=torch.int32)

    # --- Step 3: Vision encoder wrapper with fixed mask and spatial shapes ---
    class VisionEncoder(torch.nn.Module):
        """Takes only pixel_values; mask and spatial_shapes are baked in as constants."""

        def __init__(self, vision_tower, projector):
            super().__init__()
            self.vision_tower = vision_tower
            self.projector = projector

        def forward(self, pixel_values):
            vision_outputs = self.vision_tower(
                pixel_values=pixel_values,
                pixel_attention_mask=FULL_MASK,
                spatial_shapes=torch.tensor([[FIXED_H, FIXED_W]], dtype=torch.int64),
                return_dict=True,
            )
            image_features = vision_outputs.last_hidden_state  # [1, 1024, 768]
            x = image_features.reshape(image_features.shape[0], FIXED_H, FIXED_W, -1)
            projected = self.projector(x)  # [1, 16, 16, 1024]
            return projected.reshape(-1, projected.shape[-1])  # [256, 1024]

    vision_encoder = VisionEncoder(
        model.model.vision_tower, model.model.multi_modal_projector
    )
    vision_encoder.eval()

    pixel_values = inputs["pixel_values"]

    # Verify eager mode works
    with torch.no_grad():
        eager_out = vision_encoder(pixel_values)
    print(f"  Eager output shape: {eager_out.shape} (expected [256, 1024])")

    # --- Step 4: Export and Lower to XNNPACK ---
    print("\nExporting...")
    try:
        exported_program = export(vision_encoder, (pixel_values,), strict=False)

        print("Lowering to Edge IR + XNNPACK...")
        # _skip_dim_order=True: prevents dim_order_ops insertions that cause
        # tensor dim-order mismatches in the portable aten::add.out kernel
        lowered = to_edge_transform_and_lower(
            exported_program,
            partitioner=[XnnpackPartitioner()],
            compile_config=EdgeCompileConfig(_check_ir_validity=False, _skip_dim_order=True),
        )

        executorch_program = lowered.to_executorch(
            ExecutorchBackendConfig(extract_delegate_segments=True)
        )

        output_file = "lfm2_vision_xnnpack.pte"
        with open(output_file, "wb") as f:
            executorch_program.write_to_file(f)

        print(f"✓ Saved {output_file}")
        print(f"  Program methods: {executorch_program.methods}")
        print(f"  Input: pixel_values [1, 1024, 768] float32")
        print(f"  Output: image_embeddings [256, 1024] float32")

    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
