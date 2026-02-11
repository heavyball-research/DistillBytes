import re
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

from typing import Optional

import torch
import torch.nn as nn

from flash_attn.ops.triton.layer_norm import RMSNorm

from hnet.modules.block import Block
from hnet.modules.mlp import SwiGLU

from hnet.modules.utils import get_seq_idx, get_stage_cfg

from hnet.models.config_hnet import HNetConfig

@dataclass
class IsotropicInferenceParams:
    """Inference parameters that are passed to the main model in order
    to efficienly calculate and store the context during inference."""
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_value_memory_dict: dict = field(default_factory=dict)
    lengths_per_sample: Optional[torch.Tensor] = None

class Isotropic(nn.Module):
    def __init__(
        self,
        config: HNetConfig,
        arch: str,
        stage_idx: int,
        d_model: int = None,
        out_dim: int = None,
    ):
        super().__init__()

        self.stage_idx = stage_idx
        if d_model is None:
            self.d_model = config.d_model[self.stage_idx]
        else:
            self.d_model = d_model
        self.d_intermediate = config.d_intermediate[self.stage_idx]
        self.ssm_cfg = get_stage_cfg(config.ssm_cfg, stage_idx)
        layout_parse = re.findall(r"([mM])(\d+)", arch)

        layers = []
        layer_idx = -1
        self.arch_full = []
        for arch, n_layer in layout_parse:
            assert arch in ("m", "M")
            assert n_layer.isdigit()
            layers += [
                Block.create(
                    arch,
                    self.d_model,
                    ssm_cfg=self.ssm_cfg,
                    layer_idx=(layer_idx + i),
                )
                for i in range(int(n_layer))
            ]
            self.arch_full.extend([arch for _ in range(int(n_layer))])
            layer_idx += int(n_layer)

        self.layers = nn.ModuleList(layers)

        self.rmsnorm = RMSNorm(self.d_model, eps=1e-5)
        
        if out_dim is not None:
            self.mlp = SwiGLU(d_model=self.d_model, d_intermediate=self.d_intermediate, d_out=out_dim)
        else:
            self.mlp = None
            
    @contextmanager
    def unsafe_reduce_optimizedmodule_overhead(self):
        # extremely unsafe strategy to reduce overhead of executing torch.compiled blocks
        ocs = torch._dynamo.utils.call_size
        torch._dynamo.utils.call_size = torch._dynamo.decorators.skip(lambda x, i: x.shape[i])
        with torch._dynamo.set_stance(skip_guard_eval_unsafe=True):
            yield
        torch._dynamo.utils.call_size = ocs
        
    @contextmanager
    def nonfirst_blockctx(self):
        callcount = getattr(self, "callcount", 0)
        with (
            self.unsafe_reduce_optimizedmodule_overhead()
            if callcount > 3
            else nullcontext()
        ):
            yield
        self.callcount = callcount + (torch._dynamo.eval_frame._stance.stance != "force_eager")
    
    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        mask=None,
        inference_params=None
    ):
        seq_idx = None
        
        if mask is not None:
            packed = False
            assert (
                hidden_states.dim() == 3
            ), "Hidden states must be (B, L, D) in unpacked mode"
        else:
            seq_idx = get_seq_idx(cu_seqlens, device=hidden_states.device)
            packed = True

        residual = None
        
        is_first_layer = True
        for layer, arch in zip(self.layers, self.arch_full):
            if arch in ("m", "M"):
                if hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(0)
                    residual = None if residual is None else residual.unsqueeze(0)
            elif arch in ("t", "T"):
                if hidden_states.dim() == 3 and packed:
                    hidden_states = hidden_states.squeeze(0)
                    residual = None if residual is None else residual.squeeze(0)  
            else:
                # Currently supporting only Mamba2 blocks.
                raise NotImplementedError
            
            if is_first_layer:
                hidden_states, residual = layer(
                    hidden_states,
                    residual,
                    seq_idx=seq_idx,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                    inference_params=inference_params
                )
                is_first_layer = False
            else:
                with self.nonfirst_blockctx():
                    hidden_states, residual = layer(
                        hidden_states,
                        residual,
                        seq_idx=seq_idx,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                        inference_params=inference_params
                    )

        # Setting prenorm=False ignores the residual
        hidden_states = self.rmsnorm(
            hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
        )

        if hidden_states.dim() == 3 and packed:
            hidden_states = hidden_states.squeeze(0)
            
        hidden_states = self.mlp(hidden_states) if self.mlp else hidden_states
        
        if inference_params is not None:
            inference_params.seqlen_offset += hidden_states.shape[1]

        return hidden_states
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        key_value_memory_dict = {}
        for i, layer in enumerate(self.layers):
            key_value_memory_dict[i] = layer.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype
            )
        return IsotropicInferenceParams(
            key_value_memory_dict=key_value_memory_dict,
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
        )

    def step(self, hidden_states, inference_params):
        """
        Assumes hidden_states is (B, 1, D). Steps each of the layers in order, and then steps the main model.
        """
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer.step(
                hidden_states, inference_params, residual=residual
            )

        hidden_states = self.rmsnorm(
            hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
        )
        hidden_states = self.mlp(hidden_states) if self.mlp else hidden_states
        inference_params.seqlen_offset += 1

        return hidden_states
    
    def block_compile(self, mode):
        for i, l in enumerate(self.layers):
            self.layers.register_module(
                str(i), torch.compile(l, fullgraph=True,mode=mode,backend="inductor")
            )
