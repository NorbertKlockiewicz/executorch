#!/usr/bin/env python3
"""
Debug script: Compare ET text decoder logits vs HF PyTorch decoder logits.

Both get identical combined embeddings (from ET vision + ET embed) as input.
We compare:
  1. Top-5 token predictions at prefill (position -1 / last token)
  2. KL divergence / max-diff on logits
  3. First 10 decode steps side-by-side
"""

import ctypes
import os
import sys

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
VISION_PTE   = os.path.join(REPO_ROOT, "lfm2_vision_xnnpack.pte")
EMBED_PTE    = os.path.join(REPO_ROOT, "lfm2_token_embedding_xnnpack.pte")
DECODER_PTE  = os.path.join(REPO_ROOT, "lfm2_text_decoder_xnnpack_fp32.pte")
IMAGE_PATH   = os.path.join(REPO_ROOT, "dog.jpg")
MODEL_ID     = "LiquidAI/LFM2-VL-450M"

MAX_SEQ_LEN  = 2048
VOCAB_SIZE   = 65536
IMAGE_SIZE   = 512
IMAGE_START_TOKEN = 498
IMAGE_END_TOKEN   = 499
MAX_DECODE_STEPS  = 15


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
    in_shapes  = [meta.input_tensor_meta(i).sizes()  for i in range(meta.num_inputs())]
    out_shapes = [meta.output_tensor_meta(i).sizes() for i in range(meta.num_outputs())]
    print(f"in={in_shapes} out={out_shapes}")
    return m


def embed_tokens(embed_module, token_ids):
    seq_len = token_ids.shape[1]
    padded = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int64)
    padded[:, :seq_len] = token_ids
    out = embed_module.forward([padded])[0]
    return out[:, :seq_len, :]


def et_run_decoder(decoder_module, embeddings, input_pos):
    seq_len = embeddings.shape[1]
    if seq_len == 1:
        flat_logits = decoder_module.forward([embeddings, input_pos])[0]
    else:
        padded_emb = torch.zeros(1, MAX_SEQ_LEN, 1024, dtype=torch.float32)
        padded_emb[:, :seq_len, :] = embeddings
        padded_pos = torch.zeros(MAX_SEQ_LEN, dtype=torch.int64)
        padded_pos[:seq_len] = input_pos
        flat_logits = decoder_module.forward([padded_emb, padded_pos])[0]
    return flat_logits.reshape(1, VOCAB_SIZE)


def hf_run_prefill(hf_model, combined_embeds):
    """Run HF decoder prefill with inputs_embeds, return logits + past_key_values."""
    seq_len = combined_embeds.shape[1]
    with torch.no_grad():
        out = hf_model.language_model(
            inputs_embeds=combined_embeds,
            use_cache=True,
            return_dict=True,
        )
    return out.logits[:, -1, :], out.past_key_values


def hf_run_decode(hf_model, token_id, past_key_values):
    """Single-token HF decode step."""
    with torch.no_grad():
        out = hf_model.language_model(
            input_ids=torch.tensor([[token_id]], dtype=torch.long),
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
    return out.logits[:, -1, :], out.past_key_values


def top5(logits, tokenizer):
    probs = torch.softmax(logits.squeeze(0), dim=-1)
    vals, idxs = probs.topk(5)
    return [(tokenizer.decode([i.item()]), v.item()) for i, v in zip(idxs, vals)]


def main():
    print("=== ET Decoder Debug: Comparing ET vs HF Decoder ===\n")

    print("--- Loading dylibs ---")
    load_dylibs()

    print("\n--- Loading ET models ---")
    vision_module  = load_pte(VISION_PTE)
    embed_module   = load_pte(EMBED_PTE)
    decoder_module = load_pte(DECODER_PTE)

    print(f"\n--- Loading HF model and processor ---")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    hf_model  = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, device_map="cpu", torch_dtype=torch.float32
    )
    hf_model.eval()

    print(f"\n--- Preparing inputs ---")
    image = Image.open(IMAGE_PATH).resize((IMAGE_SIZE, IMAGE_SIZE))
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
    print(f"  input_ids: {input_ids.shape}, <image> at {img_start_pos}, </image> at {img_end_pos}")

    # --- Build combined embeddings (shared between ET and HF decoder) ---
    print("\n--- Building combined embeddings (ET pipeline) ---")
    image_embeds = vision_module.forward([pixel_values])[0].unsqueeze(0)  # [1,256,1024]
    before_ids   = input_ids[:, :img_start_pos + 1]
    after_ids    = input_ids[:, img_end_pos:]
    before_embeds = embed_tokens(embed_module, before_ids)
    after_embeds  = embed_tokens(embed_module, after_ids)
    combined = torch.cat([before_embeds, image_embeds, after_embeds], dim=1)
    seq_len  = combined.shape[1]
    print(f"  combined: {combined.shape}")

    # =====================================================================
    # PREFILL COMPARISON
    # =====================================================================
    print("\n" + "="*60)
    print("PREFILL COMPARISON")
    print("="*60)

    # ET prefill
    print("\n[ET] Running prefill...")
    prefill_pos  = torch.arange(seq_len, dtype=torch.int64)
    et_logits    = et_run_decoder(decoder_module, combined, prefill_pos)
    et_next_tok  = torch.argmax(et_logits, dim=-1).item()
    print(f"  ET next token: {et_next_tok!r} = {processor.tokenizer.decode([et_next_tok])!r}")
    print(f"  ET top-5: {top5(et_logits, processor.tokenizer)}")

    # HF prefill
    print("\n[HF] Running prefill...")
    hf_logits, hf_pkv = hf_run_prefill(hf_model, combined)
    hf_next_tok = torch.argmax(hf_logits, dim=-1).item()
    print(f"  HF next token: {hf_next_tok!r} = {processor.tokenizer.decode([hf_next_tok])!r}")
    print(f"  HF top-5: {top5(hf_logits, processor.tokenizer)}")

    # Compare logit distributions
    diff = (et_logits.squeeze(0) - hf_logits.squeeze(0)).abs()
    print(f"\n  Logit diff — max: {diff.max():.4f}, mean: {diff.mean():.4f}, "
          f"allclose(1e-1): {torch.allclose(et_logits, hf_logits, atol=1e-1)}")

    # KL divergence: KL(HF || ET)
    hf_log_p = torch.log_softmax(hf_logits.squeeze(0), dim=-1)
    et_log_q = torch.log_softmax(et_logits.squeeze(0), dim=-1)
    kl = torch.nn.functional.kl_div(et_log_q, hf_log_p.exp(), reduction="sum").item()
    print(f"  KL(HF‖ET): {kl:.4f}  (0=identical, <0.01=very close)")

    prefill_match = (et_next_tok == hf_next_tok)
    print(f"\n  Prefill first token match: {'✅ YES' if prefill_match else '❌ NO'}")

    # =====================================================================
    # DECODE LOOP COMPARISON
    # =====================================================================
    print("\n" + "="*60)
    print(f"DECODE LOOP COMPARISON (first {MAX_DECODE_STEPS} steps)")
    print("="*60)

    et_token  = et_next_tok
    hf_token  = hf_next_tok
    cur_pos   = seq_len

    et_tokens  = [et_token]
    hf_tokens  = [hf_token]
    mismatches = 0

    print(f"\n{'Step':>4}  {'ET token':>12}  {'HF token':>12}  {'Match':>6}")
    print("-" * 45)
    print(f"{'0':>4}  {processor.tokenizer.decode([et_token]):>12}  "
          f"{processor.tokenizer.decode([hf_token]):>12}  "
          f"{'✅' if et_token == hf_token else '❌':>6}")

    for step in range(1, MAX_DECODE_STEPS):
        # ET decode
        et_emb = embed_tokens(embed_module, torch.tensor([[et_token]], dtype=torch.int64))
        et_pos = torch.tensor([cur_pos], dtype=torch.int64)
        et_logits  = et_run_decoder(decoder_module, et_emb, et_pos)
        et_token   = torch.argmax(et_logits, dim=-1).item()
        et_tokens.append(et_token)

        # HF decode
        hf_logits, hf_pkv = hf_run_decode(hf_model, hf_token, hf_pkv)
        hf_token = torch.argmax(hf_logits, dim=-1).item()
        hf_tokens.append(hf_token)

        match = (et_token == hf_token)
        if not match:
            mismatches += 1
        print(f"{step:>4}  {processor.tokenizer.decode([et_token]):>12}  "
              f"{processor.tokenizer.decode([hf_token]):>12}  "
              f"{'✅' if match else '❌':>6}")

        cur_pos += 1
        if et_token == processor.tokenizer.eos_token_id:
            print("  [ET EOS]")
            break

    print(f"\n  Mismatches: {mismatches}/{MAX_DECODE_STEPS}")
    print(f"\n  ET output: {processor.tokenizer.decode(et_tokens, skip_special_tokens=True)!r}")
    print(f"  HF output: {processor.tokenizer.decode(hf_tokens, skip_special_tokens=True)!r}")

    # =====================================================================
    # EXTRA: Check if ET KV cache is getting corrupted
    # (Run a fresh ET prefill after decoder has already been used)
    # =====================================================================
    print("\n" + "="*60)
    print("EXTRA: ET logit stats at prefill")
    print("="*60)
    # Re-run just the prefill fresh (ET decoder has internal KV cache, so it
    # accumulated state from all the decode steps; let's check logit magnitude)
    et_log_soft = torch.log_softmax(et_logits.squeeze(0), dim=-1)
    print(f"  ET last-step logits — min: {et_logits.min():.4f}, max: {et_logits.max():.4f}")
    print(f"  HF last-step logits — min: {hf_logits.min():.4f}, max: {hf_logits.max():.4f}")


if __name__ == "__main__":
    main()
