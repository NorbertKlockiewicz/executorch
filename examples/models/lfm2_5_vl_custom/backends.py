"""
Backend partitioner factory for LFM2.5-VL-1.6B.

This is the thesis core: the same model, the same export pipeline, one flag
change → a different backend. get_partitioners() returns the right set of
partitioners for each named method based on the chosen backend.

All backend-specific imports are inside the if-branches so that missing
SDKs (e.g. coremltools not installed on a Linux machine) don't cause import
errors when running with a different backend.

Supported backends:
  xnnpack  — XNNPACK (CPU, all platforms)
  coreml   — CoreML / Apple Neural Engine (macOS / iOS only)
  qnn      — Qualcomm Neural Network (Snapdragon devices only)

Usage:
  partitioners = get_partitioners("xnnpack", quantize=True)
  # Returns {"vision_encoder": [...], "token_embedding": [...], "text_decoder": [...]}

Known limitations documented as thesis findings:
  CoreML:
    - SSM Conv1d layers are not supported by CoreML; they fall back to
      portable C++ kernels. Only standard attention + linear layers delegate.
    - Dynamic sequence length unsupported below iOS 18.
    - KV cache state management (mutable buffers) disabled in OSS build.

  QNN:
    - Static shapes ONLY — sequence length must be fixed or padded to a constant.
    - Conv1d is auto-converted to Conv2d (unsqueeze → conv2d → squeeze) by the
      CanonicalizeConv pass; this adds overhead.
    - 4-bit embedding quantization is NOT supported by QNN.
"""

from typing import Dict, List


def get_partitioners(backend: str, quantize: bool) -> Dict[str, List]:
    """
    Returns a dict mapping method name → list of Partitioner instances.

    Partitioners are applied in order within each method: the first
    partitioner claims nodes it can handle, the second gets the remainder.
    This is how XNNPACK handles quantized models: first pass grabs only
    DynamicQuantLinear nodes (via DYNAMIC_QUANT precision), second pass
    grabs all remaining compatible ops.
    """
    if backend == "xnnpack":
        return _xnnpack_partitioners(quantize)
    elif backend == "coreml":
        return _coreml_partitioners(quantize)
    elif backend == "qnn":
        return _qnn_partitioners(quantize)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose from: xnnpack, coreml, qnn"
        )


# ---------------------------------------------------------------------------
# XNNPACK
# ---------------------------------------------------------------------------

def _xnnpack_partitioners(quantize: bool) -> Dict[str, List]:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.backends.xnnpack.partition.config.xnnpack_config import ConfigPrecisionType

    if quantize:
        # Two-pass strategy for quantized text decoder:
        #   Pass 1: partition only DynamicQuantLinear ops (per_op_mode=True prevents
        #           merging, so pass 2 can still see remaining ops in the graph)
        #   Pass 2: partition all remaining XNNPACK-compatible ops (conv, linear, etc.)
        decoder_partitioners = [
            XnnpackPartitioner(
                config_precisions=ConfigPrecisionType.DYNAMIC_QUANT,
                per_op_mode=True,
            ),
            XnnpackPartitioner(),
        ]
    else:
        decoder_partitioners = [XnnpackPartitioner()]

    return {
        "vision_encoder":  [XnnpackPartitioner()],
        "token_embedding": [XnnpackPartitioner()],
        "text_decoder":    decoder_partitioners,
    }


# ---------------------------------------------------------------------------
# CoreML
# ---------------------------------------------------------------------------

def _coreml_partitioners(quantize: bool) -> Dict[str, List]:
    import coremltools as ct
    from executorch.backends.apple.coreml.partition.coreml_partitioner import CoreMLPartitioner
    from executorch.backends.apple.coreml.compiler import CoreMLBackend

    # iOS 17 minimum: required for int8 activation quantization.
    # Raise to iOS 18 if you need per-block int4 weight quantization or
    # dynamic (enumerated) input shapes.
    min_target = ct.target.iOS17

    compile_specs = CoreMLBackend.generate_compile_specs(
        # CPU_AND_NE: allow both CPU and Apple Neural Engine.
        # Use CPU_ONLY for debugging (deterministic, no ANE dispatch).
        compute_unit=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=min_target,
        # FLOAT16 is CoreML's preferred compute precision on ANE.
        # Use FLOAT32 only if you observe numerical issues.
        compute_precision=ct.precision.FLOAT16,
    )

    partitioner = CoreMLPartitioner(
        compile_specs=compile_specs,
        # take_over_mutable_buffer=False: KV cache state management via
        # CoreML stateful models requires iOS 18+ and is disabled in the
        # OSS ExecuTorch build. The KV cache custom op path handles state.
        take_over_mutable_buffer=False,
    )

    return {
        "vision_encoder":  [partitioner],
        "token_embedding": [partitioner],
        "text_decoder":    [partitioner],
    }


# ---------------------------------------------------------------------------
# QNN (Qualcomm Neural Network)
# ---------------------------------------------------------------------------

def _qnn_partitioners(quantize: bool) -> Dict[str, List]:
    from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
    from executorch.backends.qualcomm.utils.utils import (
        generate_htp_compiler_spec,
        generate_qnn_executorch_compiler_spec,
    )
    from executorch.backends.qualcomm.serialization.qnn_compile_spec_schema import QcomChipset

    # HTP (Hexagon Tensor Processor) backend options.
    # use_fp16=True: run non-quantized ops in fp16 on the HTP DSP.
    # use_fp16=False: force int8 everywhere (requires PT2E quantization).
    backend_options = generate_htp_compiler_spec(use_fp16=not quantize)

    # SM8650 = Snapdragon 8 Gen 3. Change to match your target SoC.
    compiler_specs = generate_qnn_executorch_compiler_spec(
        soc_model=QcomChipset.SM8650,
        backend_options=backend_options,
        debug=False,
    )

    partitioner = QnnPartitioner(compiler_specs)

    # NOTE: QNN requires static shapes (use_kv_cache=True, no dynamic seq_len).
    # The text decoder must be exported with a fixed sequence length, or the
    # graph must be padded to the max length at runtime.
    return {
        "vision_encoder":  [partitioner],
        "token_embedding": [partitioner],
        "text_decoder":    [partitioner],
    }
