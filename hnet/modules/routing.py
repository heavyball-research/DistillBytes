from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnet.modules.mlp import SwiGLU
from hnet.modules.utils import get_seq_idx

from einops import repeat, rearrange

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

@dataclass
class SequenceRouting:
    p: torch.Tensor
    b: torch.Tensor
    p_selected: torch.Tensor
    
@dataclass
class RoutingModuleState:
    """
    The state of the routing module.

    Contains
        - [has_seen_tokens] (batch_size,) bool tensor. Whether that batch element has processed any tokens yet.
        - [last_hidden_state] (batch_size, d_model) tensor. The last hidden state of the batch element (used for boundary prediction).
    """

    has_seen_tokens: torch.Tensor  # (batch_size,)
    last_hidden_state: torch.Tensor  # (batch_size, d_model)
    
@dataclass
class DeChunkState:
    """
    The state of the dechunk.

    Contains
        - [last_value] (batch_size, d_model) tensor. The last value of the batch element (used for the EMA).
    """

    last_value: torch.Tensor  # (batch_size, d_model)

class QProjPadded(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_flat, w, k_flat, cu):
        slen = x_flat.shape[0]
        # compute x@w.T, but padded left by 1seqlen
        q_padded = torch.empty(
            slen + 1, *x_flat.shape[1:], dtype=x_flat.dtype, device=x_flat.device
        )
        torch.mm(x_flat, w.T.type_as(x_flat), out=q_padded[1:])
        ctx.save_for_backward(x_flat, w, cu)
        return q_padded.index_copy_(0, cu[:-1], -k_flat[cu[:-1]])[:slen]

    @staticmethod
    def backward(ctx, dq_flat):
        x_flat, w, cu = ctx.saved_tensors
        zero_grad = torch.zeros(
            cu.shape[0] - 1,
            dq_flat.shape[-1],
            device=dq_flat.device,
            dtype=dq_flat.dtype,
        )
        dq_flat = dq_flat.index_copy(0, cu[:-1], zero_grad)

        dx_flat = torch.zeros_like(x_flat)
        torch.mm(dq_flat[1:], w.type_as(dq_flat), out=dx_flat[:-1])
        dw = dq_flat[1:].mT @ x_flat[:-1]

        return dx_flat, dw, None, None

class RoutingModule(nn.Module):
    def __init__(self, d, d_intermediate):
        super().__init__()
        self.d = d
        self.d_intermediate = d_intermediate
        self.q_proj_layer = SwiGLU(d_model=d, d_intermediate=d_intermediate)
        self.k_proj_layer = SwiGLU(d_model=d, d_intermediate=d_intermediate)

    def forward(self, hidden_states, cu_seqlens=None, mask=None, prefill_boundary=None, next_boundary=False):
        assert (mask is not None) or (
            cu_seqlens is not None
        ), "Either mask or cu_seqlens must be provided"

        hidden_states_flat = None
        if cu_seqlens is not None:
            # We are in packed mode, so hidden_states is (T, D). Make it (B, T, D)
            hidden_states_flat = hidden_states
            hidden_states = hidden_states.unsqueeze(0)
            
        if prefill_boundary is not None:
            prob = torch.tensor(1.0, dtype=hidden_states.dtype, device=hidden_states.device).unsqueeze(0)
            if next_boundary:
                boundary_prob = torch.stack(((1.0 - prob), prob), dim=-1)
            else:
                boundary_prob = torch.stack((prob, (1.0 - prob)), dim=-1)
            
            selected_idx = torch.argmax(boundary_prob, dim=-1)
            boundary_mask = (selected_idx == 1)
            
            prefill_boundary_boundary_mask = prefill_boundary.b
            prefill_boundary_boundary_prob = prefill_boundary.p
            
            boundary_prob = torch.cat([prefill_boundary_boundary_prob, boundary_prob], dim=0)
            boundary_mask = torch.cat([prefill_boundary_boundary_mask, boundary_mask], dim=0)
            
            selected_idx = torch.argmax(boundary_prob, dim=-1)
            
            if cu_seqlens is not None:
                selected_probs = F.pad(boundary_mask.cumsum(0), (1, 0))[cu_seqlens]
            else:
                selected_probs = boundary_prob.gather(
                    dim=-1, index=selected_idx.unsqueeze(-1)
                )  # (shape hidden_states.shape[:-1], 1)

            return SequenceRouting(
                p=boundary_prob,
                b=boundary_mask,
                p_selected=selected_probs,
            )

        if cu_seqlens is not None:
            k_flat = self.k_proj_layer(hidden_states_flat)
            q_flat_unshifted = self.q_proj_layer(hidden_states_flat)
            eye = torch.eye(
                self.d, device=hidden_states_flat.device, dtype=hidden_states_flat.dtype
            )
            q_flat = QProjPadded.apply(q_flat_unshifted, eye, k_flat, cu_seqlens)

            cos_sim = torch.einsum(
                "l d, l d -> l",
                F.normalize(q_flat[1:], dim=-1),
                F.normalize(k_flat[1:], dim=-1),
            )
        else:
            q = self.q_proj_layer(hidden_states[:, :-1])
            k = self.k_proj_layer(hidden_states[:, 1:])

            cos_sim = torch.einsum(
                "b l d, b l d -> b l",
                F.normalize(q, dim=-1),
                F.normalize(k, dim=-1),
            )
        # this clamp should no-op as long as no precision issues are encountered
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        
        # Force boundary probability of the first element to 1.0
        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (0, 1), "constant", PAD_PROB)

        if cu_seqlens is not None:
            if boundary_prob.dim() > 1:
                boundary_prob = boundary_prob.squeeze(0)
            boundary_prob[cu_seqlens[:1] - 1] = PAD_PROB

        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        # Forward: discrete selection
        selected_idx = torch.argmax(boundary_prob, dim=-1)
        boundary_mask = (selected_idx == 1)  # boolean mask
        
        if mask is not None:
            # No invalid tokens can be selected
            boundary_mask = boundary_mask & mask

        selected_probs = boundary_prob.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )  # (shape hidden_states.shape[:-1], 1)

        return SequenceRouting(
            p=boundary_prob,  # (shape hidden_states.shape[:-1], 2)
            b=boundary_mask,  # (shape hidden_states.shape[:-1])
            p_selected=selected_probs,  # (shape hidden_states.shape[:-1], 1)
        )
        
class ChunkLayer(nn.Module):
    
    def forward(self, hidden_states, boundary_mask, cu_seqlens=None, mask=None):
        assert (mask is not None) or (
            cu_seqlens is not None
        ), "Either mask or cu_seqlens must be provided"

        if cu_seqlens is not None:
            next_hidden_states = hidden_states[boundary_mask]
            next_cu_seqlens = F.pad(
                boundary_mask.cumsum(dim=0)[cu_seqlens[1:] - 1], (1, 0)
            )
            next_max_seqlen = int((next_cu_seqlens[1:] - next_cu_seqlens[:-1]).max())
            next_mask = None
        else:
            next_cu_seqlens = None
            num_tokens = boundary_mask.sum(dim=-1)
            next_max_seqlen = int(num_tokens.max())

            device = hidden_states.device
            L = hidden_states.shape[1]
            token_idx = (
                torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            next_hidden_states = torch.gather(
                hidden_states,
                dim=1,
                index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                    -1, -1, hidden_states.shape[-1]
                ),
            )

            next_mask = (
                torch.arange(next_max_seqlen, device=device)[None, :]
                < num_tokens[:, None]
            )
            next_max_seqlen = None

        return next_hidden_states, next_cu_seqlens, next_max_seqlen, next_mask
    
class DeChunkLayer(nn.Module):

    def __init__(
        self,
        d_model,
        dtype=torch.bfloat16,
        block_size=256,
        headdim=32,
    ):
        super().__init__()
        self.d_model = d_model

        self.dtype = dtype
        self.block_size = block_size
        self.headdim = headdim
        assert d_model % self.headdim == 0
        self.nheads = d_model // self.headdim
        
    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return DeChunkState(
            last_value=torch.zeros(
                batch_size, self.d_model, device=device, dtype=torch.bfloat16
            ),
        )

    def forward(
        self,
        hidden_states,
        boundary_mask,
        boundary_prob,
        null_group,
        cu_seqlens=None,
        mask=None,
        inference_params=None
    ):
        p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - (1e-4))

        if cu_seqlens is not None:
            p = p[boundary_mask].unsqueeze(0)
            seq_idx = get_seq_idx(cu_seqlens, device=hidden_states.device)
        else:
            B, L = boundary_mask.shape
            seq_idx = None

            token_idx = (
                torch.arange(L, device=hidden_states.device)[None, :]
                + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)

            p = torch.gather(
                p, dim=1, index=seq_sorted_indices[:, : hidden_states.shape[1]]
            )  # (B, M)

        original_dtype = hidden_states.dtype
        # Reuse Mamba2 kernel for EMA Deaggregator.
        dt = torch.log(1 / (1 - p)).to(self.dtype)
        x = (hidden_states / dt[..., None]).to(self.dtype)
        A = -torch.ones(
            (self.nheads,), device=hidden_states.device, dtype=torch.float32
        )
        b = p.to(self.dtype)
        c = torch.ones_like(b)

        out = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            repeat(dt, "b l -> b l h", h=self.nheads),
            A,
            rearrange(b, "b l -> b l 1 1"),
            rearrange(c, "b l -> b l 1 1"),
            chunk_size=self.block_size,
            seq_idx=seq_idx,
        )
        out = rearrange(out, "b l h p -> b l (h p)")
        if cu_seqlens is not None:
            # Flatten batch dimension that was introduced by unsqueeze(0) upstream.
            out = out.squeeze(0)
            plug_back_idx = boundary_mask.cumsum(dim=0)
            gather_idx = torch.clamp(plug_back_idx - 1, min=0)
            out = torch.gather(
                out, dim=0, index=gather_idx.unsqueeze(-1).expand(-1, self.d_model)
            )
            null_fill = null_group.reshape(1, self.d_model).expand(out.shape[0], -1)
            null_mask = plug_back_idx.unsqueeze(-1) == 0
            out = torch.where(
                null_mask,
                null_fill,
                out,
            )
        else:
            plug_back_idx = torch.cumsum(boundary_mask, dim=1)  # (B, L)
            gather_idx = torch.clamp(plug_back_idx - 1, min=0)
            out = torch.gather(
                out,
                dim=1,
                index=gather_idx.unsqueeze(-1).expand(-1, -1, self.d_model),
            )
            null_mask = plug_back_idx.unsqueeze(-1) == 0
            null_fill = null_group.expand_as(out)
            out = torch.where(
                null_mask,
                null_fill,
                out,
            )
            
        out = out.to(original_dtype)
            
        if inference_params is not None:
            inference_params.last_value.copy_(out[-1])

        return out
