"""
Token embedding export for LFM2.5-VL-1.6B.

Exports nn.Embedding(65536, 2048) as the 'token_embedding' method.

Input:  [1, seq_len] int64 token IDs   (seq_len is dynamic: 1 … MAX_SEQ_LEN)
Output: [1, seq_len, 2048] float32 embeddings

Optional: int8 group quantization via EmbeddingQuantHandler (bitwidth=8, group_size=32).

IMPORTANT — call order in export.py:
  export_text_decoder() must be called BEFORE this function.
  EmbeddingQuantHandler mutates hf_model.model.language_model.embed_tokens in-place
  (replaces the fp32 weight with int8 + scale tensors). If token_embedding runs first,
  the mutated weights break load_state_dict in the text decoder.
"""

import torch
from torch.export import Dim, export

from executorch.examples.models.llama.source_transformation.quantize import (
    EmbeddingQuantHandler,
)

from .config import MAX_SEQ_LEN, VOCAB_SIZE, MODEL_DIM


def export_token_embedding(hf_model, quantize: bool = False):
    """
    Export the token embedding table with a dynamic sequence length dimension.

    When quantize=True, applies int8 symmetric group quantization
    (bitwidth=8, group_size=32, packed=False) before tracing. The quantized
    embedding kernel is part of the quantized_decomposed:: op set and will be
    recognised by the XNNPACK partitioner.

    Returns an ExportedProgram.
    """
    language_model = hf_model.model.language_model

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

    embed_module = embed_module.eval()

    # Dynamic sequence length: any value from 1 to MAX_SEQ_LEN
    token_dim = Dim("token_dim", min=1, max=MAX_SEQ_LEN)
    example_ids = torch.zeros(1, MAX_SEQ_LEN, dtype=torch.int64)
    dynamic_shapes = [{1: token_dim}]

    print("  Tracing token embedding...")
    with torch.no_grad():
        ep = export(
            embed_module,
            (example_ids,),
            dynamic_shapes=dynamic_shapes,
            strict=True,
        )

    print(f"  Token embedding exported: [1, seq_len] → [1, seq_len, {MODEL_DIM}]")
    return ep
