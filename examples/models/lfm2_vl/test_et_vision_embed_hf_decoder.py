#!/usr/bin/env python3
"""
Hybrid validation: ET vision encoder + ET token embedding -> HF PyTorch decoder.

Confirms that ET vision and embedding components produce correct outputs
before debugging the ET text decoder.
"""

import ctypes
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VISION_PTE  = os.path.join(REPO_ROOT, "lfm2_vision_xnnpack.pte")
EMBED_PTE   = os.path.join(REPO_ROOT, "lfm2_token_embedding_xnnpack.pte")
IMAGE_PATH  = os.path.join(REPO_ROOT, "dog.jpg")
MODEL_ID    = "LiquidAI/LFM2-VL-450M"

MAX_SEQ_LEN       = 2048
IMAGE_SIZE        = 512
IMAGE_START_TOKEN = 498  # <image>
IMAGE_END_TOKEN   = 499  # </image>
MAX_NEW_TOKENS    = 50


def load_dylibs():
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
    in_shapes  = [meta.input_tensor_meta(i).sizes() for i in range(meta.num_inputs())]
    out_shapes = [meta.output_tensor_meta(i).sizes() for i in range(meta.num_outputs())]
    print(f"in={in_shapes} out={out_shapes}")
    return m


def embed_tokens(embed_module, token_ids):
    seq_len = token_ids.shape[1]
    padded = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int64)
    padded[:, :seq_len] = token_ids
    out = embed_module.forward([padded])[0]
    return out[:, :seq_len, :]


def main():
    print("=== Hybrid: ET vision+embed -> HF decoder ===\n")

    print("--- Loading dylibs ---")
    load_dylibs()

    print("\n--- Loading ET models ---")
    vision_module = load_pte(VISION_PTE)
    embed_module  = load_pte(EMBED_PTE)

    print("\n--- Loading HF model and processor ---")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    hf_model  = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="cpu", torch_dtype=torch.float32
    )
    hf_model.eval()
    lm = hf_model.model.language_model  # Lfm2Model
    lm_head = hf_model.lm_head          # Linear(1024 -> vocab)

    print(f"\n--- Preparing inputs ({IMAGE_SIZE}x{IMAGE_SIZE}) ---")
    image  = Image.open(IMAGE_PATH).resize((IMAGE_SIZE, IMAGE_SIZE))
    prompt = "What is in this image?"
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}
    ]
    text   = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    input_ids    = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]

    ids = input_ids[0].tolist()
    img_start_pos = ids.index(IMAGE_START_TOKEN)
    img_end_pos   = ids.index(IMAGE_END_TOKEN)
    print(f"  input_ids: {input_ids.shape}, <image>@{img_start_pos}, </image>@{img_end_pos}")

    # --- ET vision encoder ---
    print("\n--- ET vision encoder ---")
    image_embeds = vision_module.forward([pixel_values])[0].unsqueeze(0)  # [1, 256, 1024]
    print(f"  image_embeds: {image_embeds.shape}")

    # --- ET token embedding ---
    print("\n--- ET token embedding ---")
    before_ids    = input_ids[:, :img_start_pos + 1]
    after_ids     = input_ids[:, img_end_pos:]
    before_embeds = embed_tokens(embed_module, before_ids)
    after_embeds  = embed_tokens(embed_module, after_ids)
    print(f"  before: {before_embeds.shape}, image: {image_embeds.shape}, after: {after_embeds.shape}")

    combined = torch.cat([before_embeds, image_embeds, after_embeds], dim=1)
    print(f"  combined: {combined.shape}")

    # --- HF decoder (prefill) ---
    print("\n--- HF decoder prefill ---")
    with torch.no_grad():
        out = lm(
            inputs_embeds=combined,
            use_cache=True,
            return_dict=True,
        )
    logits = lm_head(out.last_hidden_state[:, -1:, :]).squeeze(1)  # [1, vocab]
    past_key_values = out.past_key_values
    next_token_id = torch.argmax(logits, dim=-1).item()
    generated = [next_token_id]
    print("Generating: ", end="", flush=True)
    print(processor.tokenizer.decode([next_token_id]), end="", flush=True)

    # --- HF decode loop ---
    for _ in range(MAX_NEW_TOKENS - 1):
        if next_token_id == processor.tokenizer.eos_token_id:
            break
        with torch.no_grad():
            out = lm(
                input_ids=torch.tensor([[next_token_id]], dtype=torch.long),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        logits = lm_head(out.last_hidden_state[:, -1:, :]).squeeze(1)
        past_key_values = out.past_key_values
        next_token_id = torch.argmax(logits, dim=-1).item()
        if next_token_id == processor.tokenizer.eos_token_id:
            break
        generated.append(next_token_id)
        print(processor.tokenizer.decode([next_token_id]), end="", flush=True)

    print()
    print("\n=== Result ===")
    print(processor.tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()
