import torch.nn as nn

from .hnet import HNet
from .config_hnet import HNetConfig

class HNetLM(nn.Module):
    def __init__(self, c: HNetConfig, pretrain_transformer_name: str):
        super().__init__()
        self.c, self.v, d = c, c.vocab_size, c.d_model[0]
        self.embeddings = nn.Embedding(self.v, d)
        self.boundary_embedding = nn.Embedding(self.v, d)
        self.backbone = HNet(c, stage_idx=0, pretrain_transformer_name=pretrain_transformer_name)
        self.lm_head = nn.Linear(d, self.v * 2, bias=False)
        
    def forward(self, forward_mode: str, **forward_kwargs):
        assert forward_mode in ["encoder_alignment", "joint_distillation", "boundary_learning", "byte_inference", "stage_1_inference"]
        
        input_ids = forward_kwargs["iids_values"]
        iids_values = self.embeddings(input_ids)
        boundary_embedding = self.boundary_embedding(input_ids)

        if forward_mode == "encoder_alignment":
            return self.backbone.encoder_alignment(
                iids_values,
                forward_kwargs["iids_offsets"],
                forward_kwargs["max_seqlen"]
            )
        elif forward_mode == "boundary_learning":
            return self.backbone.boundary_learning(
                iids_values,
                forward_kwargs["iids_offsets"],
                forward_kwargs["max_seqlen"]
            )
        elif forward_mode == "joint_distillation":
            student_outputs, teacher_outputs, student_mask, predicted_boundary, padded_boundary, mask = self.backbone.joint_distillation(
                iids_values,
                forward_kwargs["iids_offsets"],
                forward_kwargs["max_seqlen"],
                forward_kwargs["token_input_ids"],
                forward_kwargs["teacher_model"],
                forward_kwargs["boundary"]
            )
            
            return student_outputs, teacher_outputs, student_mask, predicted_boundary, padded_boundary, mask
        elif forward_mode == "byte_inference":
            hidden_states, inference_params, bpred = self.backbone.forward_dechunk(
                iids_values,
                forward_kwargs["iids_offsets"],
                forward_kwargs["max_seqlen"],
                boundary_embedding,
                prefill_boundary=forward_kwargs.get("prefill_boundary"),
                next_boundary=forward_kwargs.get("next_boundary"),
                inference_params=forward_kwargs.get("inference_params")
            )
            return self.lm_head(hidden_states), inference_params, bpred
        elif forward_mode == "stage_1_inference":
            return self.backbone.stage_1_inference(
                iids_values,
                forward_kwargs["iids_offsets"],
                forward_kwargs["max_seqlen"],
                inference_params=forward_kwargs.get("inference_params")
            )
    
    def stage1_step(self, iids_values, next_token_len, inference_params):
        B = iids_values.shape[0]
        assert (
            B == 1
        ), "HNetForCausalLM step currently only supports batch size 1 -- need to handle different-size lengths for each sample"

        hidden_states = self.embeddings(iids_values)

        return self.backbone.step(
            hidden_states, next_token_len, inference_params
        )
        
    def stage2_step(self, iids_values, is_next_boundary, inference_params):
        B = iids_values.shape[0]
        assert (
            B == 1
        ), "HNetForCausalLM step currently only supports batch size 1 -- need to handle different-size lengths for each sample"

        hidden_states = self.embeddings(iids_values)
        boundary_embedding = None
        if self.boundary_embedding:
            boundary_embedding = self.boundary_embedding(iids_values)
        

        hidden_states, inference_params = self.backbone.step_byte(
            hidden_states, boundary_embedding, is_next_boundary, inference_params
        )
        
        return self.lm_head(hidden_states),inference_params


    def freeze_for_routing(self):
        """Freeze parameters for routing training, keeping only routing modules trainable"""
        # Freeze embeddings
        for param in self.embeddings.parameters():
            param.requires_grad = False

        # Freeze backbone components for routing
        self.backbone.freeze_for_routing()
        
    def block_compile(self, mode="default"):
        self.backbone.block_compile(mode)
