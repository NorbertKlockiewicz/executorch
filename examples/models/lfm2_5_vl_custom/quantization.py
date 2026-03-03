"""
Quantizer factories and PT2E quantization helpers for LFM2.5-VL-1.6B.

Provides two things:
  1. get_quantizers(backend) — returns a list of PT2E Quantizer instances
     for the chosen backend (used by text_decoder.py for the re-export step).
  2. get_quant_weight_transform(quantize, dtype) — returns the source-transform
     function that rewrites weights in-place before torch.export (8da4w).

Supported modes:
  XNNPACK: 8da4w — 8-bit dynamic activation, 4-bit grouped weight (group_size=128)
  QNN:     8da4w via QnnQuantizer (different annotation pass, same concept)
  CoreML:  quantization handled post-export by CoreML compiler (b4w / c4w compile spec),
           so no PT2E quantizer is needed here.
"""

from typing import List, Optional
import torch


# ---------------------------------------------------------------------------
# PT2E Quantizers
# ---------------------------------------------------------------------------

def get_xnnpack_quantizers() -> List:
    """
    Returns an XNNPACKQuantizer configured for 8-bit dynamic activation,
    4-bit per-group weight quantization (group_size=128).

    This matches the 8da4w mode used in export_lfm2p5_vl.py but is called
    directly without going through get_quantizer_and_quant_params().
    """
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )

    quantizer = XNNPACKQuantizer()
    quantizer.set_global(
        get_symmetric_quantization_config(
            is_per_channel=True,
            is_dynamic=True,   # dynamic activation quantization (8da)
            is_qat=False,
        )
    )
    return [quantizer]


def get_qnn_quantizers() -> List:
    """
    Returns a QnnQuantizer for int8 activation, int4 weight quantization
    (W8A8 or W4A8 depending on the QNN SDK version and target SoC).

    Note: QNN does NOT support 4-bit embedding quantization — use int8
    embedding for QNN targets (handled in token_embedding.py via the
    quantize flag and EmbeddingQuantHandler).
    """
    from executorch.backends.qualcomm.quantizer.quantizer import (
        QnnQuantizer,
        QuantDtype,
    )

    quantizer = QnnQuantizer()
    quantizer.set_global_quantization_config(
        QuantDtype.use_8a8w,  # Start with W8A8; try use_8a4w if SoC supports it
    )
    return [quantizer]


def get_quantizers(backend: str) -> List:
    """
    Returns the appropriate PT2E quantizer list for the given backend.
    Returns an empty list for CoreML (quantization is done post-export
    via compile specs in backends.py).
    """
    if backend == "xnnpack":
        return get_xnnpack_quantizers()
    elif backend == "qnn":
        return get_qnn_quantizers()
    elif backend == "coreml":
        # CoreML applies quantization during preprocess() via coremltools,
        # not via PT2E. No PT2E quantizer needed.
        return []
    else:
        return []


# ---------------------------------------------------------------------------
# Weight quantization source transform (8da4w)
# ---------------------------------------------------------------------------

def get_weight_quant_transform(
    quantize: bool,
    dtype: torch.dtype = torch.float32,
    qmode: str = "8da4w",
    group_size: int = 128,
):
    """
    Returns the weight quantization source transform function for 8da4w,
    or None if quantize=False.

    This transform is applied to the ET model (nn.Module) BEFORE torch.export,
    rewriting Linear weight tensors to quantized (int4 packed) format in-place.
    The resulting model uses quantized_decomposed:: ops for weight dequantize
    + matmul that the XNNPACK partitioner recognises as DynamicQuantLinear nodes.

    Returns:
      A callable f(model) -> model, or None.
    """
    if not quantize:
        return None

    from executorch.examples.models.llama.source_transformation.quantize import (
        get_quant_weight_transform,
    )
    from executorch.extension.llm.export.config.llm_config import LlmConfig

    llm_config = LlmConfig()
    llm_config.quantization.qmode = qmode
    llm_config.quantization.group_size = group_size

    return get_quant_weight_transform(
        quantization_mode=llm_config.quantization.qmode,
        group_size=llm_config.quantization.group_size,
        computation_dtype=dtype,
        checkpoint_path=None,
        tokenizer_path=None,
        calibration_tasks=None,
        calibration_limit=None,
        calibration_seq_length=None,
    )
