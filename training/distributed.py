"""Distributed training utilities with clean separation of concerns."""

import os
from datetime import timedelta
from typing import Dict
import torch
import torch.distributed as dist

from .data import DistributedState


class DistributedStateFactory:
    """Factory for creating distributed state configuration."""

    @staticmethod
    def create() -> DistributedState:
        """Create distributed state from environment variables."""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            return DistributedStateFactory._create_distributed()
        else:
            return DistributedStateFactory._create_single_process()

    @staticmethod
    def _create_distributed() -> DistributedState:
        """Create state for distributed training."""
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        dist.init_process_group(backend='nccl', timeout=timedelta(days=365))

        return DistributedState(
            is_distributed=True,
            rank=rank,
            world_size=world_size,
            device=device
        )

    @staticmethod
    def _create_single_process() -> DistributedState:
        """Create state for single process training."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return DistributedState(
            is_distributed=False,
            rank=0,
            world_size=1,
            device=device
        )

    @staticmethod
    def cleanup():
        """Clean up distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()


class DistributedSynchronizer:
    """Handles distributed synchronization operations."""

    def __init__(self, distributed_state: DistributedState):
        self._state = distributed_state
        self._accelerator = distributed_state.accelerator

    def barrier_if_needed(self) -> None:
        """Execute barrier synchronization if distributed."""
        if self._accelerator is not None:
            self._accelerator.wait_for_everyone()
        elif self._state.is_distributed:
            dist.barrier()

    def sync_tensor_across_processes(self, tensor: torch.Tensor) -> torch.Tensor:
        """Synchronize tensor values across all processes."""
        if not self._state.is_distributed:
            return tensor

        if self._accelerator is not None:
            if tensor.device != self._state.device:
                tensor = tensor.to(self._state.device)
            if tensor.ndim == 0:
                tensor = tensor.unsqueeze(0)
            gathered = self._accelerator.gather(tensor)
            return gathered.mean()

        # Ensure tensor is on correct device
        if tensor.device != self._state.device:
            tensor = tensor.to(self._state.device)

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor / self._state.world_size

    def sync_metrics_dict(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Synchronize metrics dictionary across processes."""
        if not self._state.is_distributed:
            return metrics

        synced_metrics = {}
        for key, value in metrics.items():
            tensor = torch.tensor(value, device=self._state.device)
            synced_tensor = self.sync_tensor_across_processes(tensor)
            synced_metrics[key] = synced_tensor.item()

        return synced_metrics
