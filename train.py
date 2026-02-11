import argparse
import os
from typing import Optional


import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DeepSpeedPlugin

from transformers import AutoTokenizer

from hnet.models.mixer_seq import HNetConfig, HNetLM
from training import (
    Trainer,
    TrainerConfig,
    TrainingStage,
    TrainingConfig,
    DistributedState,
    LoggerConfig,
)


def train_stage(stage: TrainingStage, args, model: torch.nn.Module,
                pretrain_tokenizer, distributed_state: DistributedState) -> torch.nn.Module:
    """Unified training stage entry point using new Trainer architecture."""
    # Stage-specific model preparation
    model = _prepare_model_for_stage(stage, model)

    # Create trainer configuration
    config = _create_trainer_config(stage, args)

    # Create and run trainer
    trainer = Trainer(config, model, pretrain_tokenizer, distributed_state)
    return trainer.run_training_loop()

def _prepare_model_for_stage(stage: TrainingStage, model: torch.nn.Module) -> torch.nn.Module:
    """Prepare model for specific training stage."""

    model = model.bfloat16()

    if stage == TrainingStage.ROUTING:
        model.freeze_for_routing()
    elif stage == TrainingStage.DISTILL:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
    elif stage == TrainingStage.DECHUNK:
        for name, param in model.named_parameters():
            if ("boundary_embedding" in name or
               "null_group" in name or
               "decoder" in name or
               "mlp_out" in name or
               "residual_proj" in name or
               "lm_head.weight" == name):
                param.requires_grad = True
            else:
                param.requires_grad = False
    elif stage == TrainingStage.DECHUNK_STEP2:
        # Unfreeze all parameters
        for name, param in model.named_parameters():
            param.requires_grad = True

    return model

def _create_trainer_config(stage: TrainingStage, args) -> TrainerConfig:
    """Create trainer configuration for the given stage."""
    # Get stage-specific parameters
    steps, mbs, lr = _get_stage_params(stage, args)

    # Create training configuration
    training_config = TrainingConfig(
        optim=args.optim,
        lr=lr,
        use_hierarchical_lrs=args.use_hierarchical_lrs
    )

    # Create logger configuration from args
    logger_config = LoggerConfig.from_string(
        logger_type_str=getattr(args, 'logger', 'console'),
        project_name="hnet"
    )

    # Create TrainerConfig with distill-specific parameters
    trainer_config = TrainerConfig(
        stage=stage,
        total_steps=steps,
        microbatch_size=mbs,
        training_config=training_config,
        save_dir=args.save_dir,
        logger_config=logger_config,
        data_dir=args.data_dir,
        pretrain_transformer_name=getattr(args, 'pretrain_transformer_name', None),
        teacher_model_path=getattr(args, 'teacher_model_path', None),
    )

    return trainer_config

def _get_stage_params(stage: TrainingStage, args) -> tuple:
    """Get stage-specific steps, microbatch size, and learning rate."""
    if stage == TrainingStage.EMBEDDING:
        return args.embedding_steps, args.embedding_mbs, args.embedding_lr
    elif stage == TrainingStage.ROUTING:
        return args.routing_steps, args.routing_mbs, args.routing_lr
    elif stage == TrainingStage.DISTILL:
        return args.distill_steps, args.distill_mbs, args.distill_lr
    elif stage == TrainingStage.DECHUNK:
        return args.dechunk_steps, args.dechunk_mbs, args.dechunk_lr
    else:  # DECHUNK_STEP2
        return args.dechunk_step2_steps, args.dechunk_step2_mbs, args.dechunk_step2_lr


def _select_microbatch_size(args) -> int:
    stage_options = [
        ("run_embedding", "embedding_mbs"),
        ("run_routing", "routing_mbs"),
        ("run_distill", "distill_mbs"),
        ("run_dechunk", "dechunk_mbs"),
        ("run_dechunk_step2", "dechunk_step2_mbs"),
    ]

    for flag_name, mbs_name in stage_options:
        if getattr(args, flag_name, False):
            return int(getattr(args, mbs_name, 1))

    return 1

def _create_accelerator(args) -> Accelerator:
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, gradient_as_bucket_view=True, bucket_cap_mb=100)
    
    micro_batch_size = _select_microbatch_size(args)
    accum_steps = TrainingConfig().calculate_accumulate_steps(int(os.environ['WORLD_SIZE']), micro_batch_size)

    deepspeed_plugin = DeepSpeedPlugin(
        zero_stage=2,
        gradient_accumulation_steps=accum_steps
    )
    
    deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = int(micro_batch_size)
    accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin, kwargs_handlers=[ddp_kwargs]) 

    return accelerator

def _create_distributed_state(accelerator: Accelerator) -> DistributedState:
    return DistributedState(
        is_distributed=accelerator.num_processes > 1,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        device=accelerator.device,
        accelerator=accelerator
    )


def train(args):    
    """Main training function using new modular architecture."""
    # Setup distributed training via Accelerate
    accelerator = _create_accelerator(args)
    distributed_state = _create_distributed_state(accelerator)

    if distributed_state.is_distributed and distributed_state.is_main_process:
        print(f"Initialized distributed training with {distributed_state.world_size} GPUs")

    # Setup model and tokenizer
    model, pretrain_tokenizer = _setup_model_and_tokenizer(args, distributed_state)

    # Run training stages based on command line arguments
    if args.run_embedding:
        if distributed_state.is_main_process:
            print('Running embedding training stage')
        model = train_stage(TrainingStage.EMBEDDING, args, model, pretrain_tokenizer, distributed_state)

    if args.run_routing:
        if distributed_state.is_main_process:
            print('Running routing training stage')
        model = train_stage(TrainingStage.ROUTING, args, model, pretrain_tokenizer, distributed_state)

    if args.run_distill:
        if distributed_state.is_main_process:
            print('Running distill training stage')
        model = train_stage(TrainingStage.DISTILL, args, model, pretrain_tokenizer, distributed_state)
        
    if args.run_dechunk:
        if distributed_state.is_main_process:
            print('Running dechunk training stage')
        model = train_stage(TrainingStage.DECHUNK, args, model, pretrain_tokenizer, distributed_state)
        
    if args.run_dechunk_step2:
        if distributed_state.is_main_process:
            print('Running dechunk-step2 training stage')
        model = train_stage(TrainingStage.DECHUNK_STEP2, args, model, pretrain_tokenizer, distributed_state)

    # Cleanup distributed training
    if distributed_state.accelerator is not None:
        distributed_state.accelerator.wait_for_everyone()

def _setup_model_and_tokenizer(args, distributed_state: DistributedState):
    """Setup model and tokenizer with proper initialization."""
    # Configuration
    config = HNetConfig.load_config(args.config)
    
    model = HNetLM(config, pretrain_transformer_name=args.pretrain_transformer_name)

    # Load checkpoint
    _load_checkpoint(model, distributed_state, args.pt_ckpt)

    # Initialize tokenizer
    pretrain_tokenizer = AutoTokenizer.from_pretrained(args.pretrain_transformer_name)

    return model, pretrain_tokenizer

def _load_checkpoint(model: torch.nn.Module, distributed_state: DistributedState, checkpoint_path: Optional[str] = None) -> None:
    """Load model checkpoint."""
    if checkpoint_path is None:
        if distributed_state.is_main_process:
            print("No checkpoint specified, skipping checkpoint loading")
        return

    model_ckpt = torch.load(checkpoint_path, map_location="cpu")
            
    from collections import OrderedDict
    new_state_dict = OrderedDict()
        
    for k, v in model_ckpt.items():
        new_key = k.replace("._orig_mod", "")
            
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict, strict=True)

    print(f"Checkpoint loaded from: {checkpoint_path}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', type=str)
    ap.add_argument('-p', '--pt-ckpt', type=str, default=None,
                   help='Path to checkpoint file (e.g. ./checkpoints/model.pt). If not provided, training starts from scratch.')
    ap.add_argument('--pretrain-transformer-name', type=str, default=None, help='Pretrained transformer name or path')
    ap.add_argument('--teacher-model-path', type=str, default=None, help='Teacher model weight path')
    ap.add_argument('-l', '--logger', default='local', choices=['wandb', 'local', 'swanlab'])
    ap.add_argument('-o', '--optim', default='adamw', choices=['adamw', 'sgd'])
    ap.add_argument('--use-hierarchical-lrs', action='store_true', help='enable hierarchical learning rates')
    ap.add_argument('--embedding-lr', type=float, default=6e-4, help='adamw learning rate for embedding training')
    ap.add_argument('--embedding-mbs', type=int, default=1<<17, help='maximum microbatchsize (tokens per gpu) for embedding training')
    ap.add_argument('--embedding-steps', type=int, default=10000, help='embedding training steps')
    ap.add_argument('--routing-lr', type=float, default=6e-4, help='adamw learning rate for routing training')
    ap.add_argument('--routing-mbs', type=int, default=1<<16, help='maximum microbatchsize (tokens per gpu) for routing training')
    ap.add_argument('--routing-steps', type=int, default=20000, help='routing training steps')
    ap.add_argument('--distill-lr', type=float, default=4e-5, help='adamw learning rate for distill training')
    ap.add_argument('--distill-mbs', type=int, default=1<<13, help='maximum microbatchsize (tokens per gpu) for distill training')
    ap.add_argument('--distill-steps', type=int, default=160_000, help='distill training steps')
    ap.add_argument('--dechunk-lr', type=float, default=2e-5, help='adamw learning rate for dechunk training')
    ap.add_argument('--dechunk-mbs', type=int, default=1<<17, help='maximum microbatchsize (tokens per gpu) for dechunk training')
    ap.add_argument('--dechunk-steps', type=int, default=60000, help='dechunk training steps')
    ap.add_argument('--dechunk-step2-lr', type=float, default=2e-4, help='adamw learning rate for dechunk-step2 training')
    ap.add_argument('--dechunk-step2-mbs', type=int, default=1<<16, help='maximum microbatchsize (tokens per gpu) for dechunk-step2 training')
    ap.add_argument('--dechunk-step2-steps', type=int, default=20000, help='dechunk-step2 training steps')
    ap.add_argument('--save-dir', type=str, help='overwriting output path to save checkpoints')
    ap.add_argument('--data-dir', type=str, default=None,
                   help='override dataset directory used by the dataloader (default: data/dataloader.py OUTPUT_DIR)')

    # Training stage control
    ap.add_argument('--run-embedding', action='store_true', help='run embedding training stage')
    ap.add_argument('--run-routing', action='store_true', help='run routing training stage')
    ap.add_argument('--run-distill', action='store_true', help='run distill training stage')
    ap.add_argument('--run-dechunk', action='store_true', help='run dechunk training stage')
    ap.add_argument('--run-dechunk-step2', action='store_true', help='run dechunk-step2 training stage')

    args = ap.parse_args()

    assert (args.run_embedding or args.run_routing or args.run_distill or args.run_dechunk_step2 or args.run_dechunk)

    train(args)

if __name__ == '__main__':
    main()
