#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Run LFM2-VL divided model using ExecuTorch Python runtime.

Model interfaces (determined from method_meta):
  vision_encoder  (lfm2_vision_xnnpack.pte):
      forward(pixel_values [1, 1024, 768] f32) -> image_embeds [256, 1024] f32
      (mask and spatial_shapes are baked-in constants; use 512x512 input images)

  token_embedding (lfm2_token_embedding_xnnpack.pte):
      forward(token_ids [1, 2048] i64) -> token_embeds [1, 2048, 1024] f32
      (fixed-length buffer padded to max_seq_len=2048)

  text_decoder    (lfm2_text_decoder_xnnpack_fp32.pte):
      forward(embeddings [1, 2048, 1024] f32, input_pos [2048] i64)
          -> logits [1, 65536] f32  (flat: last-token logits for vocab_size=65536)
      (maintains internal KV cache; input_pos carries position indices)

Stitching:
  prefill_embeds = embed(before_image_tokens) + vision_embeds + embed(after_image_tokens)
  Pad to [1, 2048, 1024] with zeros; set pos IDs for valid tokens only.
"""

import ctypes
import os
import sys

import torch
from PIL import Image
from transformers import AutoProcessor

# --- Configuration ---
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
VISION_PTE = os.path.join(REPO_ROOT, "lfm2_vision_xnnpack.pte")
EMBED_PTE = os.path.join(REPO_ROOT, "lfm2_token_embedding_xnnpack.pte")
DECODER_PTE = os.path.join(REPO_ROOT, "lfm2_text_decoder_xnnpack_fp32.pte")
IMAGE_PATH = os.path.join(REPO_ROOT, "dog.jpg")
MODEL_ID = "LiquidAI/LFM2-VL-450M"
MAX_NEW_TOKENS = 50
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 65536

# 512x512 -> 32x32 patches baked into vision encoder
IMAGE_SIZE = 512

# Token IDs for LFM2-VL
IMAGE_START_TOKEN = 498  # <image>
IMAGE_END_TOKEN = 499    # </image>


def load_dylibs():
    """Load portable_lib and custom/quantized op dylibs needed for the PTE files."""
    dylibs = [
        os.path.join(REPO_ROOT, "extension", "pybindings", "_portable_lib.cpython-310-darwin.so"),
        os.path.join(REPO_ROOT, "extension", "llm", "custom_ops", "libcustom_ops_aot_lib.dylib"),
        os.path.join(REPO_ROOT, "kernels", "quantized", "libquantized_ops_aot_lib.dylib"),
    ]
    for path in dylibs:
        if os.path.exists(path):
            try:
                ctypes.cdll.LoadLibrary(path)
                print(f"  Loaded: {os.path.basename(path)}")
            except Exception as e:
                print(f"  Warning: {os.path.basename(path)}: {e}")


def load_pte(path):
    from executorch.extension.pybindings.portable_lib import _load_for_executorch
    print(f"  {os.path.basename(path)} ...", end=" ", flush=True)
    m = _load_for_executorch(path)
    meta = m.method_meta("forward")
    in_shapes = [meta.input_tensor_meta(i).sizes() for i in range(meta.num_inputs())]
    out_shapes = [meta.output_tensor_meta(i).sizes() for i in range(meta.num_outputs())]
    print(f"in={in_shapes} out={out_shapes}")
    return m


def embed_tokens(embed_module, token_ids):
    """Embed a variable-length token sequence using the fixed-buffer embedding model.

    token_ids: [1, seq_len] int64, seq_len <= MAX_SEQ_LEN
    Returns: [1, seq_len, 1024] float32
    """
    seq_len = token_ids.shape[1]
    padded = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int64)
    padded[:, :seq_len] = token_ids
    out = embed_module.forward([padded])[0]  # [1, MAX_SEQ_LEN, 1024]
    return out[:, :seq_len, :]


def run_decoder(decoder_module, embeddings, input_pos):
    """Run decoder for both prefill and decode steps.

    With enable_dynamic_shape=False the model accepts any seq_len directly —
    no padding needed. Position -1 (last token) is always the one we want.

    embeddings: [1, seq_len, 1024] float32
    input_pos:  [seq_len] int64
    Returns: logits [1, vocab_size] float32
    """
    flat_logits = decoder_module.forward([embeddings, input_pos])[0]
    return flat_logits.reshape(1, VOCAB_SIZE)


def main():
    print("=== LFM2-VL ExecuTorch Python Runner ===\n")

    print("--- Loading dylibs ---")
    load_dylibs()

    print("\n--- Loading ExecuTorch models ---")
    vision_module = load_pte(VISION_PTE)
    embed_module = load_pte(EMBED_PTE)
    decoder_module = load_pte(DECODER_PTE)

    print(f"\n--- Loading processor ---")
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    print(f"\n--- Preparing inputs ({IMAGE_SIZE}x{IMAGE_SIZE} image) ---")
    image = Image.open(IMAGE_PATH).resize((IMAGE_SIZE, IMAGE_SIZE))
    prompt = "What is in this image?"

    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    input_ids = inputs["input_ids"]  # [1, total_seq_len]
    pixel_values = inputs["pixel_values"]  # [1, 1024, 768]

    ids = input_ids[0].tolist()
    img_start_pos = ids.index(IMAGE_START_TOKEN)
    img_end_pos = ids.index(IMAGE_END_TOKEN)
    print(f"  input_ids: {input_ids.shape}")
    print(f"  <image> at {img_start_pos}, </image> at {img_end_pos}")
    print(f"  image placeholder tokens: {img_end_pos - img_start_pos - 1}")

    # --- Vision encoder ---
    print("\n--- Vision encoder ---")
    image_embeds = vision_module.forward([pixel_values])[0]  # [256, 1024]
    image_embeds = image_embeds.unsqueeze(0)  # [1, 256, 1024]
    print(f"  image_embeds: {image_embeds.shape}")

    # --- Text embeddings ---
    print("\n--- Embedding text tokens ---")
    before_ids = input_ids[:, :img_start_pos + 1]   # [..., <image>]
    after_ids = input_ids[:, img_end_pos:]           # [</image>, ...]

    before_embeds = embed_tokens(embed_module, before_ids)
    after_embeds = embed_tokens(embed_module, after_ids)
    print(f"  before: {before_embeds.shape}, image: {image_embeds.shape}, after: {after_embeds.shape}")

    # --- Stitch ---
    combined = torch.cat([before_embeds, image_embeds, after_embeds], dim=1)
    seq_len = combined.shape[1]
    print(f"  combined: {combined.shape}")

    if seq_len > MAX_SEQ_LEN:
        print(f"  WARNING: truncating from {seq_len} to {MAX_SEQ_LEN}")
        combined = combined[:, :MAX_SEQ_LEN, :]
        seq_len = MAX_SEQ_LEN

    # --- Prefill ---
    print("\n--- Prefill ---")
    prefill_pos = torch.arange(seq_len, dtype=torch.int64)
    logits = run_decoder(decoder_module, combined, prefill_pos)
    next_token_id = torch.argmax(logits, dim=-1).item()
    generated_ids = [next_token_id]
    print("Generating: ", end="", flush=True)
    print(processor.tokenizer.decode([next_token_id]), end="", flush=True)

    # --- Decode loop ---
    cur_pos = seq_len
    for _ in range(MAX_NEW_TOKENS - 1):
        if next_token_id == processor.tokenizer.eos_token_id:
            break

        next_embed = embed_tokens(
            embed_module, torch.tensor([[next_token_id]], dtype=torch.int64)
        )  # [1, 1, 1024]
        next_pos = torch.tensor([cur_pos], dtype=torch.int64)
        logits = run_decoder(decoder_module, next_embed, next_pos)
        next_token_id = torch.argmax(logits, dim=-1).item()

        if next_token_id == processor.tokenizer.eos_token_id:
            break

        generated_ids.append(next_token_id)
        print(processor.tokenizer.decode([next_token_id]), end="", flush=True)
        cur_pos += 1

    print()
    print("\n=== Generated text ===")
    print(processor.tokenizer.decode(generated_ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
