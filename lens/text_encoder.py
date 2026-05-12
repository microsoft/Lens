"""GPT-OSS text encoder for Lens.

We subclass ``transformers.GptOssForCausalLM`` so we can:

1. Return hidden states *only* at a configured layer subset (default
   ``[5, 11, 17, 23]``), avoiding the memory cost of HF's stock
   ``output_hidden_states=True`` which materializes every layer.
2. Early-exit after the last selected layer, since we don't need the
   downstream LM head at all when extracting features.

Standard ``generate(...)`` is inherited unchanged and is used by the optional
prompt reasoner.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM


class LensGptOssEncoder(GptOssForCausalLM):
    """``GptOssForCausalLM`` subclass that exposes selected hidden states."""

    def set_selected_layers(self, layer_indices: Sequence[int]) -> None:
        layers = [int(i) for i in layer_indices]
        if not layers:
            raise ValueError("layer_indices must be non-empty")
        if len(set(layers)) != len(layers):
            raise ValueError(f"layer_indices must be unique; got {layers}")
        if min(layers) < 0 or max(layers) >= len(self.model.layers):
            raise ValueError(
                f"layer_indices out of range; got {layers}, "
                f"model has {len(self.model.layers)} layers"
            )
        self._lens_selected_layers = layers
        self._lens_max_layer = max(layers)

    @torch.no_grad()
    def encode_layers(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Run the decoder up to the last selected layer and return the
        hidden states at each requested layer (in ``selected_layers`` order).
        """
        if not hasattr(self, "_lens_selected_layers"):
            raise RuntimeError("Call set_selected_layers(...) before encode_layers().")

        model = self.model
        inputs_embeds = model.embed_tokens(input_ids)
        position_ids = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
        ).unsqueeze(0).expand_as(input_ids)

        mask_kwargs = {
            "config": model.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

        hidden_states = inputs_embeds
        position_embeddings = model.rotary_emb(hidden_states, position_ids)

        captured: List[torch.Tensor] = [None] * len(self._lens_selected_layers)
        index_lookup = {idx: pos for pos, idx in enumerate(self._lens_selected_layers)}

        for i, decoder_layer in enumerate(model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[model.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
            if i in index_lookup:
                captured[index_lookup[i]] = hidden_states
            if i == self._lens_max_layer:
                break

        for pos, layer_idx in enumerate(self._lens_selected_layers):
            if captured[pos] is None:
                raise RuntimeError(
                    f"Failed to capture hidden state for layer {layer_idx}"
                )
        return captured
