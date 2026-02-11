"""Metrics collection and logging with single responsibility."""

from typing import Dict, List, Optional

from .data import TrainingMetrics, TrainingStep, LoggingConfig, DistributedState
from .distributed import DistributedSynchronizer


class MetricsAccumulator:
    """Accumulates metrics across multiple training steps."""

    def __init__(self):
        self._accumulated: Dict[str, List[float]] = {}
        self._count = 0

    def add_metrics(self, metrics: TrainingMetrics) -> None:
        """Add metrics to the accumulator."""
        all_metrics = {**metrics.loss_values, **metrics.additional_metrics}

        for key, value in all_metrics.items():
            if key not in self._accumulated:
                self._accumulated[key] = []
            self._accumulated[key].append(value)

        self._count += 1

    def get_averaged_metrics(self) -> Dict[str, float]:
        """Get averaged metrics and reset accumulator."""
        if self._count == 0:
            return {}

        averaged = {}
        for key, values in self._accumulated.items():
            averaged[key] = sum(values) / len(values)

        self.reset()
        return averaged

    def reset(self) -> None:
        """Reset the accumulator."""
        self._accumulated.clear()
        self._count = 0

    @property
    def has_metrics(self) -> bool:
        """Check if accumulator has any metrics."""
        return self._count > 0


class MetricsCollector:
    """Collects, processes and logs training metrics."""

    def __init__(self, config: LoggingConfig, distributed_state: DistributedState,
                 distributed_sync: Optional[DistributedSynchronizer] = None):
        self._config = config
        self._distributed_state = distributed_state
        self._distributed_sync = distributed_sync
        self._accumulator = MetricsAccumulator()
        self._actual_steps = 0

    def collect_step_metrics(self, step: TrainingStep, metrics: TrainingMetrics,
                           processing_time: float, microbatch_size: int) -> None:
        """Collect metrics from a training step."""
        self._accumulator.add_metrics(metrics)

        if self._config.console_output and self._distributed_state.is_main_process:
            self._print_step_info(step, metrics, processing_time, microbatch_size)

    def log_if_needed(self, step: TrainingStep) -> None:
        """Log accumulated metrics if conditions are met."""
        if not step.should_log or not self._accumulator.has_metrics:
            return

        averaged_metrics = self._accumulator.get_averaged_metrics()

        # Sync metrics across processes if distributed
        if self._distributed_sync:
            averaged_metrics = self._distributed_sync.sync_metrics_dict(averaged_metrics)

        if self._distributed_state.is_main_process:
            log_dict = self._prepare_log_dict(averaged_metrics)
            self._config.logger_func(log_dict)

        self._actual_steps += 1

    def _prepare_log_dict(self, metrics: Dict[str, float]) -> Dict[str, any]:
        """Prepare metrics dictionary for logging."""
        return {
            'step': self._actual_steps,
            **metrics
        }

    def _print_step_info(self, step: TrainingStep, metrics: TrainingMetrics,
                        processing_time: float, microbatch_size: int) -> None:
        """Print step information to console."""
        primary_loss = metrics.get_primary_loss()
        tokens_per_sec = self._calculate_throughput(processing_time, microbatch_size)

        print(f'Step {step.step} | Time: {processing_time:.2f}s | '
              f'Throughput: {tokens_per_sec:.2f} tokens/s | '
              f'Update: {step.update_weights} | Loss: {primary_loss:.4f}')

    def _calculate_throughput(self, processing_time: float, microbatch_size: int) -> float:
        """Calculate approximate tokens per second."""
        if processing_time <= 0:
            return 0.0
        return microbatch_size / processing_time

    @property
    def actual_steps(self) -> int:
        """Get the number of actual training steps (after accumulation)."""
        return self._actual_steps
