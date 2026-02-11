"""Core Trainer class that orchestrates the training process."""

import functools
import math
from collections import defaultdict
from typing import Callable, Tuple
import torch

from .data import (
    TrainerConfig, TrainingStage, TrainingStep, BatchData,
    DistributedState, LoggingConfig, CheckpointConfig, StrategyConfig
)
from .strategies import TrainingStrategyFactory, TrainingStrategy
from .metrics import MetricsCollector
from .checkpointing import CheckpointManager
from .distributed import DistributedSynchronizer
from .logger import LoggerFactory, LoggerConfig, LoggerType


class Trainer:
    """Main training orchestrator with clean separation of responsibilities."""

    def __init__(self, config: TrainerConfig, model: torch.nn.Module,
                 pretrain_tokenizer, distributed_state: DistributedState):
        self._config = config
        self._model = model
        self._pretrain_tokenizer = pretrain_tokenizer
        self._distributed_state = distributed_state
        self._accelerator = distributed_state.accelerator

        # Calculate dynamic accumulate steps
        self._actual_accumulate_steps = self._config.training_config.calculate_accumulate_steps(
            self._distributed_state.world_size, self._config.microbatch_size
        )

        if distributed_state.is_main_process:
            print(f"💡 Dynamic accumulate_steps: {self._actual_accumulate_steps} "
                  f"(target: {self._config.training_config.target_batch_tokens} tokens, "
                  f"msl: {self._config.microbatch_size})")

        if self._accelerator is not None:
            self._accelerator.gradient_accumulation_steps = self._actual_accumulate_steps

        # Initialize specialized components
        self._strategy = self._create_training_strategy()
        self._distributed_sync = self._create_distributed_synchronizer()
        self._metrics_collector = self._create_metrics_collector()
        self._checkpoint_manager = self._create_checkpoint_manager()

        # Training state
        self._actual_steps = 0

    def run_training_loop(self) -> torch.nn.Module:
        """Execute the main training loop."""
        optimizer, scheduler = self._create_optimizer_and_scheduler()

        if self._accelerator is not None:            
            self._model, optimizer, scheduler = self._accelerator.prepare(
                self._model, optimizer, scheduler
            )
            
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"
            self._model.module.block_compile(mode="reduce-overhead")

        dataloader = self._create_dataloader()

        for step, raw_data in enumerate(dataloader):
            if step >= self._config.total_steps:
                break

            batch_data = self._prepare_batch_data(raw_data)
            training_step = self._create_training_step(step)

            torch.compiler.cudagraph_mark_step_begin()
            self._execute_single_step(batch_data, training_step, optimizer, scheduler)
            self._handle_step_completion(training_step, optimizer, scheduler)

        # TODO: fix here
        return self._model.module

    def _execute_single_step(self, batch: BatchData, step: TrainingStep,
                           optimizer: torch.optim.Optimizer, scheduler) -> None:
        """Execute a single training step with all components coordinated."""
        # Execute training step using strategy
        result = self._strategy.execute_step(
            batch, step, self._model, optimizer, scheduler, self._actual_accumulate_steps
        )

        # Collect metrics
        self._metrics_collector.collect_step_metrics(step, result.metrics, result.processing_time, self._config.microbatch_size)

        # Log if needed
        self._metrics_collector.log_if_needed(step)

    def _handle_step_completion(self, step: TrainingStep,
                              optimizer: torch.optim.Optimizer, scheduler) -> None:
        """Handle post-step operations like checkpointing."""
        if step.update_weights:
            self._actual_steps += 1

        # Handle checkpointing
        self._checkpoint_manager.save_if_needed(
            step, self._actual_steps, self._model
        )

    def _create_training_step(self, step: int) -> TrainingStep:
        """Create training step configuration."""
        return TrainingStep(
            step=step,
            total_steps=self._config.total_steps,
            is_first_accumulation=(step % self._actual_accumulate_steps == 0),
            update_weights=((step + 1) % self._actual_accumulate_steps == 0)
        )

    def _prepare_batch_data(self, raw_data) -> BatchData:
        """Convert raw dataloader output to structured batch data."""
        str_sequences, boundary_sequences, embedding_sequences, token_input_ids = raw_data
    
        iids_values, iids_offsets, max_seqlen, byte_label_ids = self._seq2args(
            str_sequences, boundary_sequences, self._config.stage
        )

        return BatchData(
            iids_values=iids_values,
            iids_offsets=iids_offsets,
            max_seqlen=max_seqlen,
            boundary_sequences=boundary_sequences,
            embedding_sequences=embedding_sequences,
            token_input_ids=token_input_ids,
            byte_label_ids=byte_label_ids
        )

    def _create_training_strategy(self) -> TrainingStrategy:
        """Create the appropriate training strategy."""
        strategy_config = StrategyConfig(
            stage=self._config.stage,
            training_config=self._config.training_config
        )
        return TrainingStrategyFactory.create(
            strategy_config,
            accelerator=self._accelerator,
            pretrain_transformer_name=self._config.pretrain_transformer_name,
            teacher_model_path=self._config.teacher_model_path
        )

    def _create_distributed_synchronizer(self) -> DistributedSynchronizer:
        """Create distributed synchronization handler."""
        return DistributedSynchronizer(self._distributed_state)

    def _create_metrics_collector(self) -> MetricsCollector:
        """Create metrics collection and logging system."""
        logger_func = self._create_logger()
        logging_config = LoggingConfig(logger_func=logger_func)
        return MetricsCollector(logging_config, self._distributed_state, self._distributed_sync)

    def _create_checkpoint_manager(self) -> CheckpointManager:
        """Create checkpoint management system."""
        checkpoint_config = CheckpointConfig(
            save_dir=self._config.save_dir,
            save_interval_steps=self._calculate_save_interval()
        )
        return CheckpointManager(checkpoint_config, self._config.stage.name, self._distributed_state, self._distributed_sync)

    def _create_optimizer_and_scheduler(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """Create optimizer and learning rate scheduler."""
        training_cfg = self._config.training_config
        trainable_params = [p for p in self._model.parameters() if p.requires_grad]
        total_update_steps = self._config.total_steps // self._actual_accumulate_steps
        use_hierarchical_lrs = training_cfg.use_hierarchical_lrs

        def _make_optimizer(param_spec):
            if training_cfg.optim == 'adamw':
                return torch.optim.AdamW(
                    param_spec,
                    lr=training_cfg.lr,
                    betas=training_cfg.betas,
                    weight_decay=training_cfg.weight_decay
                )
            return torch.optim.SGD(param_spec, lr=training_cfg.lr)

        if not use_hierarchical_lrs:
            print("Using default LRs")
            optimizer = _make_optimizer(trainable_params)
            lr_lambda = self._get_lr_scheduler_function(total_update_steps)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return optimizer, scheduler

        base_model = self._model.module if hasattr(self._model, 'module') else self._model

        def _split_params_by_hierarchy(model):
            buckets = defaultdict(list)
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                depth = name.count('main_network')
                buckets[depth].append(param)

            if not buckets:
                return []

            max_depth = max(buckets.keys())
            
            assert 0 == len(buckets[max_depth - 1])

            buckets[max_depth - 1].extend(buckets.pop(max_depth))

            return [buckets[depth] for depth in range(len(buckets))]

        def _lr_modulation():
            return [2.0]

        param_splits = [group for group in _split_params_by_hierarchy(base_model) if group]

        lr_multipliers = _lr_modulation()
        lr_multipliers.append(1.0)
        
        assert len(lr_multipliers) == len(param_splits), f"{len(lr_multipliers)} != {len(param_splits)}"

        param_groups = []
        for group, mult in zip(param_splits, lr_multipliers):
            param_groups.append({
                'params': group,
                'lr': training_cfg.lr * mult
            })

        optimizer = _make_optimizer(param_groups)
        lr_lambda = self._get_lr_scheduler_function(total_update_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler

    def _create_dataloader(self):
        """Create data loader based on configuration."""
        from data import seqlen_8192_fineweb

        # Get BOS embedding for data processing
        bos_token_id = self._pretrain_tokenizer.bos_token_id if self._pretrain_tokenizer.bos_token_id else self._pretrain_tokenizer.eos_token_id
        with torch.no_grad():
            underlying_model = self._model.module if hasattr(self._model, 'module') else self._model
            bos_embedding = underlying_model.backbone.main_network.main_network.model.embed_tokens(
                torch.tensor([bos_token_id], dtype=torch.long, device='cuda')
            )[0]
            
        return seqlen_8192_fineweb(
            msl=self._config.microbatch_size,
            seq_len=8192,
            tokenizer=self._pretrain_tokenizer,
            return_ids=True,
            model=underlying_model,
            bos_emb=bos_embedding,
            return_emb=(self._config.stage == TrainingStage.EMBEDDING),
            output_dir=self._config.data_dir,
            rank=self._distributed_state.rank,
            world_size=self._distributed_state.world_size,
            use_precomputed=True,
            tokenizer_name=self._config.pretrain_transformer_name
        )

    def _calculate_save_interval(self) -> int:
        """Calculate save interval for 1B tokens."""
        tokens_per_update = (self._config.microbatch_size *
                           self._actual_accumulate_steps *
                           self._distributed_state.world_size)
        return 1_000_000_000 // tokens_per_update

    def _get_lr_scheduler_function(self, total_steps: int) -> Callable[[int], float]:
        """Get learning rate scheduler function based on training stage."""
        if self._config.stage == TrainingStage.EMBEDDING:
            return functools.partial(self._embedding_lr_scheduler, end=total_steps)
        elif self._config.stage == TrainingStage.ROUTING:
            return functools.partial(self._routing_lr_scheduler, end=total_steps)
        elif self._config.stage == TrainingStage.DISTILL:
            return functools.partial(self._distill_lr_scheduler, end=total_steps)
        elif self._config.stage == TrainingStage.DECHUNK:
            return functools.partial(self._dechunk_lr_scheduler, end=total_steps)
        else:
            return functools.partial(self._dechunk_step2_lr_scheduler, end=total_steps)

    def _create_logger(self) -> Callable:
        """Create logger function based on configuration."""
        # Get logger configuration from trainer config, default to console
        if self._config.logger_config is not None:
            logger_config = self._config.logger_config
        else:
            # Default fallback configuration
            logger_config = LoggerConfig(
                logger_type=LoggerType.CONSOLE,
                project_name="hnet"
            )

        # Create logger using LoggerFactory
        return LoggerFactory.create_callable(
            config=logger_config,
            distributed_state=self._distributed_state,
            model=self._model
        )

    @staticmethod
    def _seq2args(str_sequences, boundary_sequences, stage) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Convert string sequences to tensor arguments."""
        if stage in (TrainingStage.DECHUNK, TrainingStage.DECHUNK_STEP2):
            samples = [torch.tensor(bytearray(b), dtype=torch.long).pin_memory().to('cuda', non_blocking=True)
                  for b in str_sequences]
            input_samples = [s[:-1] for s in samples]
            values = torch.cat(input_samples, dim=0)

            lengths = [len(s) for s in input_samples]
            offsets = torch.zeros(len(lengths) + 1, dtype=torch.long, device='cuda')
            offsets[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.long, device='cuda'), dim=0)

            max_seqlen = max(lengths) if lengths else 0
            
            labels = [s[1:] for s in samples]
        
            boundaries = [b for boundary_seq in boundary_sequences for b in boundary_seq[1:]]
            true_boundaries = torch.Tensor(boundaries).to(dtype=torch.bool, device=offsets.device)
                           
            byte_label_ids = torch.cat(labels, dim=0)

            byte_label_ids[true_boundaries] += 256

            return values, offsets, max_seqlen, byte_label_ids
        else:
            samples = [torch.tensor(bytearray(b), dtype=torch.long).pin_memory().to('cuda', non_blocking=True)
                    for b in str_sequences]
            values = torch.cat(samples, dim=0)

            lengths = [len(s) for s in samples]
            offsets = torch.zeros(len(lengths) + 1, dtype=torch.long, device='cuda')
            offsets[1:] = torch.cumsum(torch.tensor(lengths, dtype=torch.long, device='cuda'), dim=0)

            max_seqlen = max(lengths) if lengths else 0
            return values, offsets, max_seqlen, None

    @staticmethod
    def _embedding_lr_scheduler(step: int, end: int) -> float:
        """Learning rate scheduler for embedding training."""
        pct = step / end
        warmup_end = 0.2
        if pct < warmup_end:
            return pct / warmup_end
        else:
            progress = (pct - warmup_end) / (1 - warmup_end)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    @staticmethod
    def _routing_lr_scheduler(step: int, end: int) -> float:
        """Learning rate scheduler for routing training."""
        pct = step / end
        warmup_end = 0.1
        if pct < warmup_end:
            return pct / warmup_end
        else:
            progress = (pct - warmup_end) / (1 - warmup_end)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    @staticmethod
    def _distill_lr_scheduler(step: int, end: int) -> float:
        """Learning rate scheduler for distillation training."""
        pct = step / end
        warmup_end = 0.01
        if pct < warmup_end:
            return pct / warmup_end
        else:
            progress = (pct - warmup_end) / (1 - warmup_end)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
        
    @staticmethod
    def _dechunk_lr_scheduler(step: int, end: int) -> float:
        """Learning rate scheduler for dechunk training."""
        pct = step / end
        warmup_end = 0.1
        if pct < warmup_end:
            return pct / warmup_end
        else:
            progress = (pct - warmup_end) / (1 - warmup_end)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    @staticmethod
    def _dechunk_step2_lr_scheduler(step: int, end: int) -> float:
        """Learning rate scheduler for dechunk-step2 training."""
        pct = step / end
        warmup_end = 0.01
        if pct < warmup_end:
            return pct / warmup_end
        else:
            progress = (pct - warmup_end) / (1 - warmup_end)
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
