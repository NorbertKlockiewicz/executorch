#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
from torch.export import export, Dim
from transformers import AutoModelForImageTextToText, AutoProcessor

# --- XNNPACK / ExecuTorch Imports ---
from executorch.exir import to_edge, EdgeCompileConfig
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner


def main():
    print("Loading LFM2-VL model...")
    # Load directly to CPU
    model = AutoModelForImageTextToText.from_pretrained(
        "LiquidAI/LFM2-VL-450M",
        device_map="cpu",
        torch_dtype=torch.float32,
    )

    processor = AutoProcessor.from_pretrained("LiquidAI/LFM2-VL-450M")

    print("\nExtracting token embedding layer...")
    token_embedding = model.model.language_model.get_input_embeddings()
    token_embedding.eval()

    # --- Step 1: Create Example Input ---
    print("\nCreating example input...")
    example_text = "Hello world"
    tokens = processor.tokenizer(example_text, return_tensors="pt")
    input_ids = tokens["input_ids"]

    print(f"Input tokens shape: {input_ids.shape}")

    # --- Step 2: Define Dynamic Shapes ---
    # We must allow variable sequence length (Dim 1)
    # Batch size (Dim 0) is fixed to 1 for simplicity
    seq_len = Dim("seq_len", min=1, max=2048)

    # We use the tuple format to avoid the "arg name mismatch" error you saw earlier
    # (Arg 0 matches input_ids)
    dynamic_shapes = ({0: 1, 1: seq_len},)

    # --- Step 3: Export and Lower to XNNPACK ---
    print("\nAttempting export and lowering...")

    try:
        # 1. Base Export
        print("1. Tracing graph...")

        # We wrap the embedding layer to give it a clean forward signature
        class EmbeddingWrapper(torch.nn.Module):
            def __init__(self, embedding_layer):
                super().__init__()
                self.emb = embedding_layer

            def forward(self, input_ids):
                return self.emb(input_ids)

        wrapper = EmbeddingWrapper(token_embedding)

        exported_program = export(
            wrapper,
            (input_ids,),
            dynamic_shapes=dynamic_shapes,
            strict=True,  # Embeddings are simple, so strict=True usually works fine
        )

        # 2. To Edge
        print("2. Converting to Edge IR...")
        edge_prog = to_edge(
            exported_program, compile_config=EdgeCompileConfig(_check_ir_validity=False)
        )

        # 3. Partition for XNNPACK
        print("3. Partitioning for XNNPACK...")
        edge_prog = edge_prog.to_backend(XnnpackPartitioner())

        # 4. Finalize
        print("4. Finalizing ExecuTorch program...")
        executorch_program = edge_prog.to_executorch()

        # Save
        output_file = "lfm2_token_embedding_xnnpack.pte"
        with open(output_file, "wb") as f:
            executorch_program.write_to_file(f)

        print(f"✓ Saved XNNPACK optimized model to {output_file}")
        print(f"  Program methods: {executorch_program.methods}")

    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
