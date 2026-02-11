"""Training strategies implementing the Strategy pattern for different training stages."""

import time
from abc import ABC, abstractmethod
from typing import List, Optional
import torch
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForCausalLM

from .data import (
    TrainingStage, BatchData, TrainingMetrics, TrainingStepResult,
    TrainingConfig, StrategyConfig, TrainingStep
)

def reverse_kl(logits, teacher_logits):
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.mean(x)
    return distil_loss

class TrainingStrategy(ABC):
    """Abstract base class for training strategies."""

    def __init__(self, config: TrainingConfig, accelerator=None):
        self._config = config
        self._accelerator = accelerator

    @abstractmethod
    def execute_step(self, batch: BatchData, step: TrainingStep, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler, accumulate_steps: int) -> TrainingStepResult:
        """Execute a single training step for this strategy."""
        pass

    @abstractmethod
    def get_stage_name(self) -> str:
        """Get the name of this training stage."""
        pass

    def cleanup(self) -> None:
        """Optional cleanup hook for releasing resources after a stage."""
        return None

    def _execute_training_base(self, forward_fn, batch: BatchData, step: TrainingStep,
                              model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                              scheduler, accumulate_steps: int, *forward_args) -> TrainingMetrics:
        """Base training step execution with gradient accumulation."""
        start_time = time.perf_counter()

        if self._accelerator is not None:
            with self._accelerator.accumulate(model):
                # Forward pass
                loss_tensors, metrics_values = forward_fn(
                    batch.iids_values, batch.iids_offsets, batch.max_seqlen,
                    batch.boundary_sequences, batch.embedding_sequences,
                    *forward_args, model=model
                )
                
                # Scale loss for gradient accumulation and backward pass
                total_loss = sum(loss_tensors) / accumulate_steps
                self._accelerator.backward(total_loss)

                # Update weights if needed
                if step.update_weights:
                    self._accelerator.clip_grad_norm_(model.parameters(), self._config.grad_clip_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
        else:
            model.require_backward_grad_sync = step.update_weights

            # Forward pass
            loss_tensors, metrics_values = forward_fn(
                batch.iids_values, batch.iids_offsets, batch.max_seqlen,
                batch.boundary_sequences, batch.embedding_sequences,
                *forward_args, model=model
            )
            
            # Scale loss for gradient accumulation and backward pass
            total_loss = sum(loss_tensors) / accumulate_steps
            total_loss.backward()

            # Update weights if needed
            if step.update_weights:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self._config.grad_clip_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        processing_time = time.perf_counter() - start_time
        return metrics_values, processing_time


class EmbeddingTrainingStrategy(TrainingStrategy):
    """Training strategy for embedding stage."""

    def execute_step(self, batch: BatchData, step: TrainingStep, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler, accumulate_steps: int) -> TrainingStepResult:
        """Execute embedding training step."""
        metrics_values, processing_time = self._execute_training_base(
            self._forward_embedding, batch, step, model, optimizer, scheduler, accumulate_steps
        )

        # Create structured metrics
        loss_embedding = metrics_values[0]
        metrics = TrainingMetrics()
        metrics.add_loss('loss_embedding', loss_embedding)

        return TrainingStepResult(metrics=metrics, processing_time=processing_time)

    def get_stage_name(self) -> str:
        return "embedding"

    @staticmethod
    def _forward_embedding(iids_values, iids_offsets, max_seqlen, boundary_sequences,
                          embedding_sequences, model):
        """Forward pass for embedding training."""
        embedding_hnet_values = model(
            "encoder_alignment", iids_values=iids_values, iids_offsets=iids_offsets, max_seqlen=max_seqlen
        )

        # Process boundary sequences
        boundary_tensors = [
            torch.tensor(b, dtype=torch.bool, device="cuda") for b in boundary_sequences
        ]
        true_boundaries = torch.cat(boundary_tensors, dim=0)

        # Process embedding sequences
        true_embeddings_tensors = [torch.stack(emb_seq) for emb_seq in embedding_sequences]
        true_embeddings = torch.cat(true_embeddings_tensors, dim=0)[true_boundaries]
        embedding_hnet = embedding_hnet_values[true_boundaries]

        # Calculate losses
        loss_mse = F.mse_loss(embedding_hnet, true_embeddings, reduction="mean")

        return [loss_mse], [loss_mse.item()]


class RoutingTrainingStrategy(TrainingStrategy):
    """Training strategy for routing stage."""

    def execute_step(self, batch: BatchData, step: TrainingStep, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler, accumulate_steps: int) -> TrainingStepResult:
        """Execute routing training step."""
        metrics_values, processing_time = self._execute_training_base(
            self._forward_routing, batch, step, model, optimizer, scheduler, accumulate_steps
        )

        # Create structured metrics
        loss_routing, accuracy, precision, recall, f1 = metrics_values
        metrics = TrainingMetrics()
        metrics.add_loss('routing', loss_routing)
        metrics.add_metric('accuracy', accuracy)
        metrics.add_metric('precision', precision)
        metrics.add_metric('recall', recall)
        metrics.add_metric('f1', f1)

        return TrainingStepResult(metrics=metrics, processing_time=processing_time)

    def get_stage_name(self) -> str:
        return "routing"

    @staticmethod
    def _forward_routing(iids_values, iids_offsets, max_seqlen, boundary_sequences,
                        embedding_sequences, model):
        """Forward pass for routing training."""
        bpred, mask = model("boundary_learning", iids_values=iids_values, iids_offsets=iids_offsets, max_seqlen=max_seqlen)

        # Get routing predictions
        pred_probs = bpred.p[mask][:, -1]
        boundaries = [b for boundary_seq in boundary_sequences for b in boundary_seq]
        true_boundaries = torch.tensor(boundaries, dtype=pred_probs.dtype, device=pred_probs.device)

        loss_routing = F.binary_cross_entropy(
            pred_probs, true_boundaries, reduction='mean'
        )
        
        loss = loss_routing

        # Calculate evaluation metrics
        pred_binary = bpred.b[mask].long()
        true_binary = true_boundaries.float().cpu().numpy()
        pred_binary_cpu = pred_binary.float().cpu().numpy()

        accuracy = accuracy_score(true_binary, pred_binary_cpu)
        precision = precision_score(true_binary, pred_binary_cpu)
        recall = recall_score(true_binary, pred_binary_cpu)
        f1 = f1_score(true_binary, pred_binary_cpu)

        return [loss], [loss_routing.item(), accuracy, precision, recall, f1]


class DechunkStep2TrainingStrategy(TrainingStrategy):
    """Training strategy for dechunk-step2 stage."""

    def execute_step(self, batch: BatchData, step: TrainingStep, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler, accumulate_steps: int) -> TrainingStepResult:
        """Execute dechunk-step2 training step."""
        if batch.byte_label_ids is None:
            raise ValueError("Dechunk-step2 training requires byte_label_ids")

        metrics_values, processing_time = self._execute_training_base(
            self._forward_dechunk_step2, batch, step, model, optimizer, scheduler, accumulate_steps,
            batch.byte_label_ids
        )

        # Create structured metrics
        loss_ce, bpb, boundary_loss = metrics_values
        metrics = TrainingMetrics()
        metrics.add_loss('loss_ce', loss_ce)
        metrics.add_metric('bpb1', bpb)
        metrics.add_metric('boundary_loss', boundary_loss)

        return TrainingStepResult(metrics=metrics, processing_time=processing_time)

    def get_stage_name(self) -> str:
        return "dechunk_step2"

    @staticmethod
    def _forward_dechunk_step2(iids_values, iids_offsets, max_seqlen, boundary_sequences,
                           embedding_sequences, byte_label_ids, model):
        """Forward pass for dechunk-step2 training."""
        
        logits, _, bpred = model(
            "byte_inference", iids_values=iids_values, iids_offsets=iids_offsets, max_seqlen=max_seqlen
        )
        
        ln2 = torch.tensor(2, device='cuda').log()
        numel = torch.tensor(byte_label_ids.numel(), dtype=torch.long).to('cuda', non_blocking=True)
        
        ce_sum = F.cross_entropy(logits.float(), byte_label_ids, reduction='sum')
        loss_ce = ce_sum / numel
        bpb = ce_sum / ln2 / numel
        
        pred_probs = bpred.p[:, -1]
        boundaries = [b for boundary_seq in boundary_sequences for b in boundary_seq[:-1]]
        true_boundaries = torch.Tensor(boundaries).to(dtype=torch.bfloat16, device=pred_probs.device)
        
        boundary_loss = F.binary_cross_entropy(
            pred_probs, true_boundaries, reduction='mean'
        )
        
        loss = loss_ce + boundary_loss

        return [loss], [loss_ce.item(), bpb.item(), boundary_loss.item()]

class DistillTrainingStrategy(TrainingStrategy):
    """Training strategy for knowledge distillation stage."""

    def __init__(self, config: TrainingConfig, pretrain_transformer_name: str, accelerator=None):
        super().__init__(config, accelerator=accelerator)
        self.pretrain_transformer_name = pretrain_transformer_name
        self._teacher_model = None

    def _load_teacher_model(self, device: torch.device):
        """Load teacher model for distributed training."""
        if self._teacher_model is None:
            self._teacher_model = AutoModelForCausalLM.from_pretrained(
                self.pretrain_transformer_name,
                torch_dtype=torch.float16,
                device_map={"": device},  # Force load on specific device
                attn_implementation="sdpa"
            ) 
            self._teacher_model.eval()

            # Ensure teacher model requires no gradients
            for param in self._teacher_model.parameters():
                param.requires_grad = False

            if dist.is_initialized() and dist.get_rank() == 0:
                teacher_params = sum(p.numel() for p in self._teacher_model.parameters())
                print(f"Teacher model loaded: {teacher_params / 1_000_000:.1f}M parameters")

    def execute_step(self, batch: BatchData, step: TrainingStep, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler, accumulate_steps: int) -> TrainingStepResult:
        """Execute distillation training step."""
        # Get device from the student model (handle DDP wrapper)
        if hasattr(model, 'module'):
            device = next(model.module.parameters()).device
        else:
            device = next(model.parameters()).device

        # Ensure teacher model is loaded on the correct device
        self._load_teacher_model(device)

        metrics_values, processing_time = self._execute_training_base(
            self._forward_distill, batch, step, model, optimizer, scheduler, accumulate_steps,
            self._teacher_model, batch.token_input_ids,
        )
        
        loss = metrics_values[0]
        metrics = TrainingMetrics()
        metrics.add_loss('loss_distill', loss)

        return TrainingStepResult(metrics=metrics, processing_time=processing_time)

    def get_stage_name(self) -> str:
        return "distill"

    def cleanup(self) -> None:
        if self._teacher_model is None:
            return
        if dist.is_initialized() and dist.get_rank() == 0:
            print("Releasing teacher model after distillation stage")
        self._teacher_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _forward_distill(
        iids_values,
        iids_offsets,
        max_seqlen,
        boundary_sequences,
        embedding_sequences,
        teacher_model,
        token_input_ids,
        model
    ):
        """Forward pass for knowledge distillation."""
        device = next(model.module.parameters()).device if hasattr(model, 'module') else next(model.parameters()).device

        # Process boundary sequences to get ground truth positions
        boundaries = [b for boundary_seq in boundary_sequences for b in boundary_seq]
        true_boundaries = torch.Tensor(boundaries).to(dtype=torch.bool, device=device)
        
        student_logits, teacher_logits, student_mask, _, _, _ = model(
            "joint_distillation", iids_values=iids_values, iids_offsets=iids_offsets, max_seqlen=max_seqlen,
            token_input_ids=token_input_ids, teacher_model=teacher_model, boundary=true_boundaries
        )
        
        student_logits_all = student_logits[student_mask]
        teacher_logits_all = teacher_logits[student_mask]

        distill_loss = reverse_kl(student_logits_all, teacher_logits_all)

        return [distill_loss], [distill_loss.item()]
    
class DechunkTrainingStrategy(TrainingStrategy):
    """Training strategy for dechunk stage."""

    def execute_step(self, batch: BatchData, step: TrainingStep, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler, accumulate_steps: int) -> TrainingStepResult:
        """Execute embedding training step."""
        metrics_values, processing_time = self._execute_training_base(
            self._forward_dechunk, batch, step, model, optimizer, scheduler, accumulate_steps, batch.byte_label_ids
        )

        # Create structured metrics
        loss_ce, bpb = metrics_values
        metrics = TrainingMetrics()
        metrics.add_loss('loss_ce', loss_ce)
        metrics.add_metric('bpb', bpb)

        return TrainingStepResult(metrics=metrics, processing_time=processing_time)

    def get_stage_name(self) -> str:
        return "dechunk"

    @staticmethod
    def _forward_dechunk(iids_values, iids_offsets, max_seqlen, boundary_sequences, embedding_sequences, byte_label_ids, model):
        """Forward pass for dechunk training."""

        logits, _, _ = model(
            "byte_inference", iids_values=iids_values, iids_offsets=iids_offsets, max_seqlen=max_seqlen
        )
        
        ln2 = torch.tensor(2, device='cuda').log()
        numel = torch.tensor(byte_label_ids.numel(), dtype=torch.long).to('cuda', non_blocking=True)
        
        ce_sum = F.cross_entropy(logits.float(), byte_label_ids, reduction='sum')
        loss_ce = ce_sum / numel
        bpb = ce_sum / ln2 / numel
        
        loss = loss_ce

        return [loss], [loss_ce.item(), bpb.item()]


class TrainingStrategyFactory:
    """Factory for creating training strategy instances."""

    @staticmethod
    def create(config: StrategyConfig, pretrain_transformer_name: Optional[str] = None,
               teacher_model_path: Optional[str] = None, accelerator=None) -> TrainingStrategy:
        """Create appropriate training strategy based on configuration."""
        if config.stage == TrainingStage.DISTILL:
            if teacher_model_path is None:
                assert pretrain_transformer_name
                teacher_model_path = pretrain_transformer_name
            return DistillTrainingStrategy(config.training_config, teacher_model_path, accelerator=accelerator)

        strategies = {
            TrainingStage.EMBEDDING: EmbeddingTrainingStrategy,
            TrainingStage.ROUTING: RoutingTrainingStrategy,
            TrainingStage.DECHUNK: DechunkTrainingStrategy,
            TrainingStage.DECHUNK_STEP2: DechunkStep2TrainingStrategy,
        }

        if config.stage not in strategies:
            raise ValueError(f"Unknown training stage: {config.stage}")

        return strategies[config.stage](config.training_config, accelerator=accelerator)

    @staticmethod
    def get_available_stages() -> List[TrainingStage]:
        """Get list of available training stages."""
        return [TrainingStage.EMBEDDING, TrainingStage.DISTILL, TrainingStage.ROUTING, TrainingStage.DECHUNK, TrainingStage.DECHUNK_STEP2]
