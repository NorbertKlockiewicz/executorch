# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# An ExecuTorch friendly implementation of LFM2-VL.

import re
from typing import Any, Dict, Optional, Tuple

import requests
import torch
import torchvision
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    replace_kv_cache_with_custom_kv_cache,
)
from executorch.examples.models.llama.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
from executorch.examples.models.model_base import EagerModelBase
from PIL import Image
from torch.export import Dim
from torchvision.transforms.v2 import functional as F
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)


def prepare_image(image: Image, target_h: int, target_w: int) -> torch.Tensor:
    """Read image into a tensor and resize the image so that it fits in
    a target_h x target_w canvas.

    Args:
        image (Image): An Image object.
        target_h (int): Target height.
        target_w (int): Target width.

    Returns:
        torch.Tensor: resized image tensor.
    """
    img = torchvision.transforms.functional.pil_to_tensor(image)
    # height ratio
    ratio_h = img.shape[1] / target_h
    # width ratio
    ratio_w = img.shape[2] / target_w
    # resize the image so that it fits in a target_h x target_w canvas
    ratio = max(ratio_h, ratio_w)
    output_size = (int(img.shape[1] / ratio), int(img.shape[2] / ratio))
    img = torchvision.transforms.Resize(size=output_size)(img)
    return img


class Lfm2Vl(torch.nn.Module):
    def __init__(
        self,
        lfm2_vl_model: AutoModelForImageTextToText,
        image_processor,
        use_sdpa_with_kv_cache_op: bool = True,
        max_context_len: int = 2048,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.use_sdpa_with_kv_cache_op = use_sdpa_with_kv_cache_op
        self.model_ = lfm2_vl_model
        self.image_processor = image_processor

        # LFM2-VL specific config
        self.image_token_index = self.model_.config.image_token_index
        self.projector_hidden_size = self.model_.config.text_config.hidden_size

        # IMPORTANT: Disable dynamic positional embedding resizing for static export
        # This prevents F.interpolate from being called with dynamic shapes
        vision_model = self.model_.model.vision_tower.vision_model
        vision_embeddings = vision_model.embeddings

        # Store original method and replace with no-op
        self._original_resize_pos_emb = vision_embeddings.resize_positional_embeddings
        vision_embeddings.resize_positional_embeddings = self._static_resize_positional_embeddings

        # Text model args for LFM2-VL (450M variant)
        self.text_model_args = ModelArgs(
            use_kv_cache=True,
            n_layers=self.model_.config.text_config.num_hidden_layers,
            vocab_size=self.model_.config.text_config.vocab_size,
            hidden_dim=self.model_.config.text_config.intermediate_size,
            max_batch_size=1,
            ffn_dim_multiplier=1,
            enable_dynamic_shape=True,
            use_sdpa_with_kv_cache_op=use_sdpa_with_kv_cache_op,
            use_hf_rope=True,
            max_context_len=max_context_len,
            max_seq_len=max_seq_len,
        )

        self.text_model = construct_transformer(self.text_model_args)

        # use custom op for SDPA
        if use_sdpa_with_kv_cache_op:
            self.text_model = replace_kv_cache_with_custom_kv_cache(self.text_model)
            self.text_model = replace_sdpa_with_custom_op(self.text_model)

        # load state dict
        self.text_model.load_state_dict(
            state_dict=self._translate_state_dict_for_text_model(),
            strict=False,
            assign=True,
        )

    def _static_resize_positional_embeddings(self, embeddings, height=None, width=None, max_length=None):
        """Static version of resize_positional_embeddings using repeat/slice instead of interpolate.

        Args:
            embeddings: The positional embeddings tensor [H, W, D]
            height: Target height (optional)
            width: Target width (optional)
            max_length: Optional max length

        Returns:
            Resized embeddings [height, width, D] or original if height/width not provided
        """
        # If height or width not provided, return unchanged
        if height is None or width is None:
            return embeddings

        # Get current dimensions
        curr_h, curr_w, hidden_dim = embeddings.shape

        # If already the right size, return as-is
        if curr_h == height and curr_w == width:
            return embeddings

        # Use simple repeat and slice instead of interpolate to avoid dynamic shapes
        # This is less accurate but works for export

        # Handle height dimension
        if height >= curr_h:
            # Repeat to get at least the target height
            repeat_factor = (height + curr_h - 1) // curr_h
            embeddings_h = embeddings.repeat_interleave(repeat_factor, dim=0)[:height]
        else:
            # Slice evenly
            indices = torch.linspace(0, curr_h - 1, height).long()
            embeddings_h = embeddings[indices]

        # Handle width dimension
        if width >= curr_w:
            # Repeat to get at least the target width
            repeat_factor = (width + curr_w - 1) // curr_w
            embeddings_hw = embeddings_h.repeat_interleave(repeat_factor, dim=1)[:, :width]
        else:
            # Slice evenly
            indices = torch.linspace(0, curr_w - 1, width).long()
            embeddings_hw = embeddings_h[:, indices]

        return embeddings_hw

    def _translate_state_dict_for_text_model(self) -> Dict[str, Any]:
        """Translate HuggingFace LFM2-VL state dict to Llama transformer format."""
        state_dict = self.model_.state_dict()
        key_map = {
            # Language model mappings for LFM2-VL
            r"language_model.model.layers.([0-9]+).self_attn.q_proj.": r"layers.\1.attention.wq.",
            r"language_model.model.layers.([0-9]+).self_attn.k_proj.": r"layers.\1.attention.wk.",
            r"language_model.model.layers.([0-9]+).self_attn.v_proj.": r"layers.\1.attention.wv.",
            r"language_model.model.layers.([0-9]+).self_attn.o_proj.": r"layers.\1.attention.wo.",
            r"language_model.model.layers.([0-9]+).input_layernorm.": r"layers.\1.attention_norm.",
            r"language_model.model.layers.([0-9]+).mlp.gate_proj.": r"layers.\1.feed_forward.w1.",
            r"language_model.model.layers.([0-9]+).mlp.down_proj.": r"layers.\1.feed_forward.w2.",
            r"language_model.model.layers.([0-9]+).mlp.up_proj.": r"layers.\1.feed_forward.w3.",
            r"language_model.model.layers.([0-9]+).post_attention_layernorm.": r"layers.\1.ffn_norm.",
            r"language_model.model.norm.": r"norm.",
            r"language_model.lm_head.": r"output.",
        }

        new_state_dict = {}

        def get_new_key(old_key: str) -> str:
            for old_pattern, replacement in key_map.items():
                if (new_key := re.sub(old_pattern, replacement, old_key)) != old_key:
                    return new_key
            return old_key

        # Convert module keys from HF transformer to Llama transformer
        for old_key in state_dict.keys():
            new_key = get_new_key(old_key)
            new_state_dict[new_key] = state_dict[old_key]

        return new_state_dict

    def get_model(self):
        return self.model_

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed tokens using the language model's embedding layer."""
        return self.model_.model.language_model.get_input_embeddings()(tokens)

    def encode_images(self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor,
                      spatial_shapes: torch.Tensor) -> torch.Tensor:
        """Encode images through vision tower and projector.

        The vision tower always outputs max_patches (32x32=1024) padded tokens.
        We unpad using pixel_attention_mask.sum(), reshape to spatial_shapes,
        then apply the projector — mirroring what Lfm2VlModel.get_image_features() does.
        """
        pixel_values = pixel_values.to(dtype=self.model_.dtype)

        # Pass through vision encoder
        vision_outputs = self.model_.model.vision_tower(
            pixel_values=pixel_values.to(device=self.model_.device, dtype=self.model_.dtype),
            pixel_attention_mask=pixel_attention_mask.to(device=self.model_.device),
            spatial_shapes=spatial_shapes.to(device=self.model_.device),
            return_dict=True,
        )

        last_hidden_state = vision_outputs.last_hidden_state  # [1, 1024, 768]

        # Unpad: keep only the valid (non-padded) patches, then reshape to spatial_shapes
        valid_len = pixel_attention_mask[0].sum()
        sh = spatial_shapes[0, 0]
        sw = spatial_shapes[0, 1]
        feature = last_hidden_state[0, :valid_len, :].unsqueeze(0)  # [1, sh*sw, 768]
        feature = feature.reshape(1, sh, sw, last_hidden_state.shape[-1])  # [1, sh, sw, 768]

        # Project to text embedding space
        image_features = self.model_.model.multi_modal_projector(feature)  # [1, sh/2, sw/2, 1024]

        # Flatten spatial dims → [1, num_tokens, 1024]
        image_features = image_features.reshape(1, -1, image_features.shape[-1])

        return image_features

    def image_preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """Preprocess image tensor for LFM2-VL."""
        target_h = self.image_processor.size["height"]
        target_w = self.image_processor.size["width"]

        # Pad the image to make it square
        l_pad = (target_w - img.shape[2]) // 2
        t_pad = (target_h - img.shape[1]) // 2
        r_pad = -((target_w - img.shape[2]) // -2)
        b_pad = -((target_h - img.shape[1]) // -2)

        torch._check(l_pad >= 0)
        torch._check(t_pad >= 0)
        torch._check(r_pad >= 0)
        torch._check(b_pad >= 0)

        # Pad the image
        resized = torch.nn.functional.pad(
            img,
            (l_pad, r_pad, t_pad, b_pad),
        )

        # Rescale
        scaled = resized * self.image_processor.rescale_factor

        # Normalize
        normed = F.normalize(
            scaled,
            self.image_processor.image_mean,
            self.image_processor.image_std,
        )

        return normed.unsqueeze(0)

    def step(
        self, token: torch.Tensor, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Input is one token. Return logits for next token."""
        token_embeds = self.embed_tokens(token).unsqueeze(0)
        return self.text_model.forward(None, {"input_pos": input_pos}, token_embeds)

    def image_embedding(self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor,
                        spatial_shapes: torch.Tensor) -> torch.Tensor:
        """Get image embeddings from preprocessed images."""
        return self.encode_images(pixel_values, pixel_attention_mask, spatial_shapes)

    def prefill_embedding(
        self,
        prompt_before_image: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
        prompt_after_image: torch.Tensor,
    ) -> torch.Tensor:
        """Create combined embeddings for prefill phase."""
        image_embeds = self.image_embedding(pixel_values, pixel_attention_mask, spatial_shapes)
        embeds_before_img = self.embed_tokens(prompt_before_image)
        embeds_after_img = self.embed_tokens(prompt_after_image)
        result = torch.cat((embeds_before_img, image_embeds, embeds_after_img), dim=1)
        return result

    def prefill(
        self,
        prompt_before_image: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
        prompt_after_image: torch.Tensor,
    ) -> Tuple[int, torch.Tensor]:
        """Prefill phase: process prompt and image together."""
        embeds = self.prefill_embedding(prompt_before_image, pixel_values, pixel_attention_mask,
                                       spatial_shapes, prompt_after_image)
        return embeds.shape[1], self.text_model.forward(
            None, {"input_pos": torch.tensor([0])}, embeds
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        return self.image_embedding(pixel_values, pixel_attention_mask, spatial_shapes)


class Lfm2VlModel(EagerModelBase):
    def __init__(
        self,
        use_sdpa_with_kv_cache_op=True,
        max_seq_len=2048,
        max_context_len=2048
    ):
        self.use_sdpa_with_kv_cache_op = use_sdpa_with_kv_cache_op
        self.max_context_len = max_context_len
        self.max_seq_len = max_seq_len

        # Load LFM2-VL model from HuggingFace
        self.model = AutoModelForImageTextToText.from_pretrained(
            "LiquidAI/LFM2-VL-450M",
            device_map="cpu",
            torch_dtype=torch.float32,
        )

        self.processor = AutoProcessor.from_pretrained(
            "LiquidAI/LFM2-VL-450M",
        )

        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor

        # Example image and prompt
        self.image_url = "https://llava-vl.github.io/static/images/view.jpg"
        self.image = Image.open(requests.get(self.image_url, stream=True).raw)

        self.model_name = "LFM2-VL-450M"

        self.conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": "What are the things I should be cautious about when I visit here?",
                    },
                ],
            },
        ]

        # Initialize lazily
        self.input = None
        self.resized_image = None

    def get_eager_model(self):
        model = Lfm2Vl(
            self.model,
            self.image_processor,
            self.use_sdpa_with_kv_cache_op,
            self.max_context_len,
            self.max_seq_len,
        )
        model.to(dtype=torch.float32)
        return model

    def get_example_inputs(self):
        """Returns processed image inputs for model.forward()."""
        if self.resized_image:
            return self.resized_image

        # Use processor to get proper pixel_values, masks, and spatial shapes
        inputs = self.processor(
            text=[""],  # Empty text, just processing image
            images=[self.image],
            return_tensors="pt",
        )

        self.resized_image = (
            inputs["pixel_values"],
            inputs["pixel_attention_mask"],
            inputs["spatial_shapes"],
        )
        return self.resized_image

    def get_inputs_for_prefill(self):
        """Returns prompts as well as image."""
        if self.input:
            return self.input

        # Apply chat template
        text = self.processor.apply_chat_template(
            self.conversation,
            add_generation_prompt=True,
        )

        # Tokenize
        inputs = self.processor(
            text=[text],
            images=[self.image],
            return_tensors="pt",
        )

        self.input_ids = inputs["input_ids"]

        # Find image token positions
        index = torch.where(self.input_ids == self.model.config.image_token_index)[1]
        self.prompt_before_image = self.input_ids[:, : index[0]]
        self.prompt_after_image = self.input_ids[:, index[-1] + 1 :]

        self.input = (
            self.prompt_before_image,
            inputs["pixel_values"],
            inputs["pixel_attention_mask"],
            inputs["spatial_shapes"],
            self.prompt_after_image,
        )
        return self.input

    def get_dynamic_shapes(self):
        return self._get_image_dynamic_shapes()

    def _get_image_dynamic_shapes(self):
        """Define dynamic shapes for image inputs (pixel_values, pixel_attention_mask, spatial_shapes)."""
        # For now, use None to disable dynamic shapes for image encoder
        # LFM2-VL's vision model has complex interpolation that doesn't work well with dynamic shapes
        return None

    def _get_prompt_dynamic_shapes(self):
        """Define dynamic shapes for prompt tokens and image inputs."""
        dim = torch.export.Dim("token_dim", min=2, max=self.max_seq_len)

        # For embeddings input, only token dimension is dynamic
        # Image inputs are already processed into fixed-size embeddings
        text_model_dynamic_shapes = ({1: dim}, {0: 1})
        return text_model_dynamic_shapes
