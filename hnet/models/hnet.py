import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, DynamicCache

from dataclasses import dataclass

from typing import Optional

from hnet.modules.isotropic import Isotropic, IsotropicInferenceParams
from hnet.modules.routing import (
    RoutingModule,
    ChunkLayer,
    DeChunkLayer,
    RoutingModuleState,
    DeChunkState
)
from hnet.modules.mlp import SwiGLU
from hnet.modules.utils import packed_to_padded

from .config_hnet import HNetConfig

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = grad_output
        return grad_x

def ste_func(x):
    return STE.apply(x)

@dataclass
class HNetState:
    encoder_state: Optional[IsotropicInferenceParams] = None
    routing_module_state: Optional[RoutingModuleState] = None
    main_network_state: Optional[DynamicCache] = None
    dechunk_state: Optional[DeChunkState] = None
    decoder_state: Optional[IsotropicInferenceParams] = None
    
    def reset(self):
        self.encoder_state = None
        self.main_network_state = None
        self.dechunk_state = None
        self.decoder_state = None

torch.set_float32_matmul_precision('high')

class HNet(nn.Module):
    def __init__(self, config: HNetConfig, stage_idx: int, pretrain_transformer_name: str):
        super().__init__()
        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]
        
        self.n = 5

        arch_layout = config.arch_layout
        for _ in range(stage_idx): 
            arch_layout = arch_layout[1]

        assert len(arch_layout) in [3, 1]
        self.is_innermost = len(arch_layout) == 1

        if self.is_innermost:
            self.main_network = AutoModelForCausalLM.from_pretrained(
                pretrain_transformer_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_3"
            )
        else:
            emb_dim = config.d_model[stage_idx + 1]

            self.main_network = HNet(config, stage_idx+1, pretrain_transformer_name=pretrain_transformer_name)

            self.encoder = Isotropic(config, arch_layout[0], stage_idx=stage_idx, out_dim=emb_dim)

            self.routing_module = RoutingModule(emb_dim, config.routing_d_intermediate)
            self.chunk_layer = ChunkLayer()
                
            self.decoder = Isotropic(config, arch_layout[0], stage_idx=stage_idx)
            self.mlp_out = SwiGLU(d_model=emb_dim, d_intermediate=config.d_intermediate[stage_idx + 1], d_out=self.d_model)
            
            self.dechunk_layer = DeChunkLayer(emb_dim)
            self.null_group = nn.Parameter(torch.Tensor(1, 1, emb_dim).zero_())
            nn.init.normal_(self.null_group)
            
            self.residual_proj = nn.Linear(
                emb_dim, emb_dim, dtype=torch.float32
            )
            nn.init.zeros_(self.residual_proj.weight)
            self.residual_proj.weight._no_reinit = True
            self.residual_func = lambda out, residual, p: out * ste_func(p) + residual
    
    def encoder_alignment(self, x_values: torch.Tensor, x_offsets: torch.Tensor, max_seqlen: int):
        embedding_hnet = self.encoder(x_values, x_offsets, max_seqlen).type_as(x_values)
        return embedding_hnet
    
    def boundary_learning(self, x_values: torch.Tensor, x_offsets: torch.Tensor, max_seqlen: int):
        with torch.no_grad():
            r = self.encoder(x_values, x_offsets, max_seqlen).type_as(x_values)
        
        batch_size = len(x_offsets) - 1
        dim = r.shape[-1]
        padded_tensor = torch.zeros(batch_size, max_seqlen, dim, dtype=r.dtype, device=r.device)
        mask = torch.zeros(batch_size, max_seqlen, dtype=torch.bool, device=r.device)
        
        for i in range(batch_size):
            start_idx = x_offsets[i]
            end_idx = x_offsets[i + 1]
            seq_len = end_idx - start_idx
            padded_tensor[i, :seq_len] = r[start_idx:end_idx]
            mask[i, :seq_len] = True

        bpred = self.routing_module(padded_tensor, mask=mask)
        return bpred, mask
    
    def _select_token_ids(self, boundary_mask: torch.Tensor, token_input_ids: torch.Tensor, x_offsets: torch.Tensor, max_seqlen: int):
        """Select embeddings and next token ids at boundary positions with padding."""
        batch_size = len(x_offsets) - 1
        
        next_token_ids_list = []
        
        for i in range(batch_size):
            start_idx = x_offsets[i]
            end_idx = x_offsets[i + 1]
            
            seq_boundary_mask = boundary_mask[start_idx:end_idx]
            
            # Get corresponding pretrain input ids and next token ids
            seq_token_ids = token_input_ids[start_idx:end_idx]
            
            # Next token ground truth is the current byte's input_ids
            next_tokens = seq_token_ids[seq_boundary_mask]
            
            next_token_ids_list.append(next_tokens)
        
        padded_next_tokens = torch.full((batch_size, max_seqlen), 0, 
                                       device=x_offsets.device, dtype=torch.long)
        
        for i, tokens in enumerate(next_token_ids_list):
            actual_len = tokens.shape[0]
            padded_next_tokens[i, :actual_len] = tokens
            
        return padded_next_tokens
    
    def joint_distillation(self, x_values: torch.Tensor, x_offsets: torch.Tensor, max_seqlen: int, 
                          token_input_ids: torch.Tensor, teacher_model, boundary):
        hidden_states = self.encoder(x_values, x_offsets, max_seqlen).type_as(x_values)
        
        batch_size = len(x_offsets) - 1
        dim = hidden_states.shape[-1]
        padded_tensor = torch.zeros(batch_size, max_seqlen, dim, dtype=hidden_states.dtype, device=hidden_states.device)
        mask = torch.zeros(batch_size, max_seqlen, dtype=torch.bool, device=hidden_states.device)
        padded_boundary = torch.zeros(batch_size, max_seqlen, dtype=torch.bool, device=hidden_states.device)
        
        for i in range(batch_size):
            start_idx = x_offsets[i]
            end_idx = x_offsets[i + 1]
            seq_len = end_idx - start_idx
            padded_tensor[i, :seq_len] = hidden_states[start_idx:end_idx]
            padded_boundary[i, :seq_len] = boundary[start_idx:end_idx]
            mask[i, :seq_len] = True
            

        student_hidden, _, _, student_mask = self.chunk_layer(
            padded_tensor, padded_boundary, mask=mask
        )

        student_outputs = self.main_network.main_network(inputs_embeds=student_hidden, attention_mask=student_mask).logits
        
        # Teacher: use GT boundary
        num_gt_tokens = padded_boundary.sum(dim=-1)
        gt_max_seqlen = int(num_gt_tokens.max())
        teacher_mask = (
            torch.arange(gt_max_seqlen, device=hidden_states.device)[None, :]
            < num_gt_tokens[:, None]
        )
        token_input_ids = torch.tensor(token_input_ids, device=hidden_states.device, dtype=torch.long)
        gt_token_ids = self._select_token_ids(boundary, token_input_ids, x_offsets, gt_max_seqlen)

        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=gt_token_ids,
                attention_mask=teacher_mask
            )

        return (student_outputs, teacher_outputs.logits, student_mask, padded_boundary, padded_boundary, mask)
        
    def forward_dechunk(self, x_values: torch.Tensor, x_offsets: torch.Tensor, max_seqlen: int, boundary_embeddings: torch.Tensor, prefill_boundary: torch.Tensor, next_boundary: bool = False, inference_params=None):
        if inference_params is None:
            inference_params = HNetState(main_network_state=None)
            
        hidden_states = self.encoder(x_values, x_offsets, max_seqlen, inference_params=inference_params.encoder_state).type_as(x_values)
            
        hidden_states_for_residual = hidden_states.to(
            dtype=self.residual_proj.weight.dtype
        )
            
        residual = self.residual_proj(hidden_states_for_residual)

        bpred = self.routing_module(hidden_states, cu_seqlens=x_offsets, prefill_boundary=prefill_boundary, next_boundary=next_boundary)
    
        hidden_states, next_cu_seqlens, next_max_seqlen, _ = self.chunk_layer(
            hidden_states, bpred.b, cu_seqlens=x_offsets 
        )
        
        padded_tensor, mask = packed_to_padded(hidden_states, next_cu_seqlens, next_max_seqlen)
        
        pretrain_outputs = self.main_network.main_network.model(inputs_embeds=padded_tensor, attention_mask=mask, use_cache=True)
        inference_params.main_network_state = pretrain_outputs.past_key_values
        
        hidden_states = pretrain_outputs.last_hidden_state[mask]
        
        hidden_states = self.dechunk_layer(
            hidden_states,
            bpred.b,
            bpred.p,
            self.null_group,
            cu_seqlens=next_cu_seqlens,
            inference_params=inference_params.dechunk_state
        )

        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype), residual, bpred.p_selected
        ).to(hidden_states.dtype)
        
        hidden_states = self.mlp_out(hidden_states)
        
        if boundary_embeddings is not None:
            hidden_states = hidden_states + boundary_embeddings

        hidden_states = self.decoder(
            hidden_states,
            x_offsets,
            max_seqlen,
            inference_params=inference_params.decoder_state
        )

        return hidden_states, inference_params, bpred

    def stage_1_inference(self, x_values: torch.Tensor, x_offsets: torch.Tensor, max_seqlen: int, inference_params=None):
        if inference_params is None:
            inference_params = HNetState(main_network_state=None)
            
        hidden_states = self.encoder(x_values, x_offsets, max_seqlen, inference_params=inference_params.encoder_state).type_as(x_values)

        # Convert to padded format for routing module
        batch_size = len(x_offsets) - 1
        dim = hidden_states.shape[-1]
        padded_tensor = torch.zeros(batch_size, max_seqlen, dim, dtype=hidden_states.dtype, device=hidden_states.device)
        mask = torch.zeros(batch_size, max_seqlen, dtype=torch.bool, device=hidden_states.device)
        
        for i in range(batch_size):
            start_idx = x_offsets[i]
            end_idx = x_offsets[i + 1]
            seq_len = end_idx - start_idx
            padded_tensor[i, :seq_len] = hidden_states[start_idx:end_idx]
            mask[i, :seq_len] = True

        bpred_output = self.routing_module(padded_tensor, mask=mask).b

        hidden_states, _, _, next_mask = self.chunk_layer(
            padded_tensor, bpred_output, mask=mask,
        )
        
        llama_output = self.main_network.main_network(inputs_embeds=hidden_states, attention_mask=next_mask, use_cache=True)
        outputs = llama_output.logits
        inference_params.main_network_state = llama_output.past_key_values

        return outputs, bpred_output, next_mask, inference_params
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """
        Allocate the inference cache for the HNet.

        Arguments:
            batch_size: int. The number of sequences in the batch.
            max_seqlen: int. The maximum sequence length in the batch.
            dtype: torch.dtype. The dtype of the inference cache.
        """
        device = self.residual_proj.weight.device
        return HNetState(
            encoder_state=self.encoder.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype
            ),
            dechunk_state=self.dechunk_layer.allocate_inference_cache(
                batch_size, max_seqlen, device, dtype=dtype
            ),
            decoder_state=self.decoder.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype
            ),
        )

    def step(self, hidden_state, next_token_len, inference_params: HNetState):
        """Single-example pretrain transformer prefill; updates HF kv_cache in the provided cache."""
        assert not self.is_innermost, "step expects the outer HNet stage"
        
        assert hidden_state.shape[1] == next_token_len, f"{hidden_state.shape[1]} != {next_token_len}"
        
        for i in range(next_token_len):
            input_state = hidden_state[:, i].unsqueeze(1)
            hidden_states = self.encoder.step(input_state, inference_params.encoder_state)

        transformer_model = self.main_network.main_network if not self.is_innermost else self.main_network
        transformer_out = transformer_model(
            inputs_embeds=hidden_states,
            past_key_values=inference_params.main_network_state,
            use_cache=True,
        )
        inference_params.main_network_state = transformer_out.past_key_values
        return transformer_out.logits, inference_params
    
    def step_byte(self, hidden_states, boundary_embedding, is_next_boundary, inference_params: HNetState):
        """Single-example pretrain transformer prefill; updates HF kv_cache in the provided cache."""
        
        hidden_states = self.encoder.step(hidden_states, inference_params.encoder_state)
        
        hidden_states_for_residual = hidden_states.to(
            dtype=self.residual_proj.weight.dtype
        )
        residual = self.residual_proj(hidden_states_for_residual)
        
        if is_next_boundary:
            transformer_out = self.main_network.main_network.model(
                inputs_embeds=hidden_states,
                past_key_values=inference_params.main_network_state,
                use_cache=True,
            )
            hidden_states = transformer_out.last_hidden_state
            inference_params.main_network_state = transformer_out.past_key_values
            inference_params.dechunk_state.last_value = hidden_states
        else:
            hidden_states = inference_params.dechunk_state.last_value
        
        original_dtype = hidden_states.dtype
        hidden_states = (hidden_states.to(dtype=residual.dtype) + residual).to(original_dtype)
        
        hidden_states = self.mlp_out(hidden_states)
        
        if boundary_embedding is not None:
            hidden_states = hidden_states + boundary_embedding

        hidden_states = self.decoder.step(hidden_states, inference_params.decoder_state)

        return hidden_states, inference_params

    def freeze_for_routing(self):
        # Freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def block_compile(self, mode):
        if self.is_innermost:
            return

        self.main_network.block_compile(mode)
        self.encoder.block_compile(mode)
        self.decoder.block_compile(mode)
        # self.register_module(
        #     "routing_module",
        #     torch.compile(self.routing_module, backend="inductor", fullgraph=True),
        # )
        self.register_module(
            "residual_proj",
            torch.compile(self.residual_proj, backend="inductor", fullgraph=True),
        )
