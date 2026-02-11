from functools import partial
from typing import Optional
import sys
import os

import torch
import torch.nn as nn

from flash_attn.ops.triton.layer_norm import RMSNorm
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party"))
)

from mamba_ssm.modules.mamba2 import Mamba2

class Mamba2Wrapper(Mamba2):
    """
    Mamba2 wrapper class that matches the block inference interface.
    """

    def step(self, hidden_states, inference_params):
        # Don't use _get_states_from_cache because we want to assert that they exist
        conv_state, ssm_state = inference_params.key_value_memory_dict[
            self.layer_idx
        ] # init class of Mamba2 accepts layer_idx
        result, conv_state, ssm_state = super().step(
            hidden_states, conv_state, ssm_state
        )

        # Update the state cache in-place
        inference_params.key_value_memory_dict[self.layer_idx][0].copy_(conv_state)
        inference_params.key_value_memory_dict[self.layer_idx][1].copy_(ssm_state)
        return result

class Block(nn.Module):
    def __init__(
        self,
        d_model,
        mixer_cls=None,
        residual_in_fp32=True,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm1 = RMSNorm(d_model, eps=1e-5)
        self.mixer = mixer_cls(d_model)
        self.mlp = None

    def forward(self,
                hidden_states: torch.Tensor, 
                residual: Optional[torch.Tensor], 
                seq_idx: Optional[torch.Tensor], 
                cu_seqlens: Optional[torch.Tensor], 
                max_seqlen: Optional[torch.Tensor],
                inference_params=None):
        hidden_states, residual = self.norm1(
            hidden_states,
            residual=residual,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
        )
        
        if isinstance(self.mixer, Mamba2Wrapper):
            hidden_states = self.mixer(hidden_states, inference_params=inference_params, seq_idx=seq_idx)
        else:
            hidden_states = self.mixer(hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        return hidden_states, residual

    @classmethod
    def create(cls, arch: str, d_model: int, ssm_cfg: dict, layer_idx: int):
        if arch.lower() != "m":
            raise ValueError(f"Unsupported arch '{arch}'. Only 'm'/'M' is supported.")
        mixer_cls = partial(Mamba2Wrapper, **ssm_cfg, layer_idx=layer_idx)

        return cls(d_model, mixer_cls)
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )
    
    def step(self, hidden_states, inference_params, residual=None):
        hidden_states, residual = self.norm1(
            hidden_states,
            residual=residual,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
        )
        hidden_states = self.mixer.step(hidden_states, inference_params)

        return hidden_states, residual
