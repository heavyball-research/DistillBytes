"""Checkpoint management with clean separation of concerns."""

from pathlib import Path
import torch

from .data import CheckpointConfig, TrainingStep, DistributedState
from .distributed import DistributedSynchronizer


class CheckpointManager:
    """Manages checkpoint saving and loading operations."""

    def __init__(self, config: CheckpointConfig, stage_name: str, distributed_state: DistributedState,
                 distributed_sync: DistributedSynchronizer):
        self._config = config
        self.stage_name = stage_name
        self._distributed_state = distributed_state
        self._distributed_sync = distributed_sync
        self._accelerator = distributed_state.accelerator

    def save_if_needed(self, step: TrainingStep, actual_steps: int,
                      model: torch.nn.Module) -> None:
        """Save checkpoint if conditions are met."""
        if not self._config.should_save_step(step, actual_steps):
            return

        if self._distributed_sync and self._config.distributed_save:
            self._distributed_sync.barrier_if_needed()

        if self._should_save_on_this_process():
            self._save_checkpoint(step.step, model)

    def save_final_checkpoint(self, step: int, model: torch.nn.Module) -> None:
        """Save final checkpoint at end of training."""
        if self._distributed_sync:
            self._distributed_sync.barrier_if_needed()

        if self._should_save_on_this_process():
            self._save_checkpoint(step, model)

    def _should_save_on_this_process(self) -> bool:
        """Determine if this process should handle checkpoint saving."""
        # Always use regular checkpoint saving
        # In distributed environments, only main process saves
        return not self._distributed_state.is_distributed or self._distributed_state.is_main_process

    def _save_checkpoint(self, step: int, model: torch.nn.Module) -> None:
        """Execute the checkpoint saving."""
        save_dir = Path(self._config.save_dir) / self.stage_name / f'{step}'

        # Always use regular checkpoint saving
        # In distributed environments, only main process saves
        if not self._distributed_state.is_distributed or self._distributed_state.is_main_process:
            self._save_regular_checkpoint(save_dir, model)

        if self._distributed_state.is_main_process:
            print(f'Checkpoint saved to {save_dir}')

    def _save_regular_checkpoint(self, save_dir: Path, model: torch.nn.Module) -> None:
        """Save regular (non-distributed) checkpoint."""
        save_dir.mkdir(exist_ok=True, parents=True)

        # Get clean model state dict
        model_dict = self._get_clean_model_state_dict(model)

        if self._accelerator is not None:
            self._accelerator.save(model_dict, save_dir / 'checkpoint.pt')
        else:
            torch.save(model_dict, save_dir / 'checkpoint.pt')

    def _get_clean_model_state_dict(self, model: torch.nn.Module) -> dict:
        """Get clean model state dict, handling DDP wrapper."""
        if self._accelerator is not None:
            # model = self._accelerator.unwrap_model(model, keep_torch_compile=True)
            # TODO: fix here
            model_dict = model.module.state_dict()
        elif hasattr(model, 'module'):
            model_dict = model.module.state_dict()
        else:
            model_dict = model.state_dict()

        # Filter out pretrain transformer parameters if needed
        return {k: v for k, v in model_dict.items()}
