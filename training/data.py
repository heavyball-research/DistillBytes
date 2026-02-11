"""Data types and structures for training system using dataclasses."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING
from enum import Enum
import torch

if TYPE_CHECKING:
    from .logger import LoggerConfig


class TrainingStage(Enum):
    """Training stages enumeration."""
    EMBEDDING = "embedding"
    ROUTING = "routing"
    DISTILL = "distill"
    DECHUNK = "dechunk"
    DECHUNK_STEP2 = "dechunk_step2"


@dataclass(frozen=True)
class TrainingStep:
    """Immutable representation of a training step state."""
    step: int
    total_steps: int
    is_first_accumulation: bool
    update_weights: bool

    @property
    def should_log(self) -> bool:
        """Whether this step should trigger logging."""
        return self.update_weights

    @property
    def should_save(self) -> bool:
        """Whether this step should trigger checkpoint saving."""
        return self.update_weights

    @property
    def progress_fraction(self) -> float:
        """Training progress as fraction (0.0 to 1.0)."""
        return self.step / max(1, self.total_steps)


@dataclass
class TrainingMetrics:
    """Container for training metrics with flexible structure."""
    loss_values: Dict[str, float] = field(default_factory=dict)
    additional_metrics: Dict[str, float] = field(default_factory=dict)

    def add_loss(self, name: str, value: float) -> None:
        """Add a loss value to the metrics."""
        self.loss_values[name] = value

    def add_metric(self, name: str, value: float) -> None:
        """Add an additional metric."""
        self.additional_metrics[name] = value

    def to_log_dict(self, step: int) -> Dict[str, Any]:
        """Convert to dictionary suitable for logging."""
        return {
            'step': step,
            **self.loss_values,
            **self.additional_metrics
        }

    def get_primary_loss(self) -> float:
        """Get the primary loss value (first loss if multiple)."""
        if not self.loss_values:
            return 0.0
        return next(iter(self.loss_values.values()))


@dataclass
class DistributedState:
    """Distributed training configuration and state."""
    is_distributed: bool
    rank: int
    world_size: int
    device: torch.device
    accelerator: Optional[Any] = None

    @property
    def is_main_process(self) -> bool:
        """Whether this is the main process (rank 0 or non-distributed)."""
        return not self.is_distributed or self.rank == 0


@dataclass
class BatchData:
    """Training batch data container."""
    iids_values: torch.Tensor
    iids_offsets: torch.Tensor
    max_seqlen: int
    boundary_sequences: List[List[bool]]
    embedding_sequences: List[List[torch.Tensor]]
    token_input_ids: Optional[torch.Tensor] = None
    byte_label_ids: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        """Number of sequences in the batch."""
        return len(self.iids_offsets) - 1

    @property
    def total_tokens(self) -> int:
        """Total number of tokens in the batch."""
        return len(self.iids_values)


@dataclass
class TrainingStepResult:
    """Result from executing a training step."""
    metrics: TrainingMetrics
    processing_time: float


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    optim: str = 'adamw'
    lr: float = 6e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    accumulate_steps: Optional[int] = None  # Will be calculated dynamically if None
    grad_clip_norm: float = 1.0
    use_hierarchical_lrs: bool = False

    # Dynamic accumulate steps calculation parameters
    target_batch_tokens: int = 8192 * 256  # Target effective batch size in tokens

    def calculate_accumulate_steps(self, world_size: int, microbatch_size: int) -> int:
        """Calculate accumulate steps dynamically based on microbatch size.

        Formula: accumulate_steps = target_batch_tokens / microbatch_size
        Default target: 8192 * 256 = 2,097,152 tokens
        """
        if self.accumulate_steps is not None:
            # Use explicitly set accumulate_steps
            return self.accumulate_steps

        # Calculate dynamically
        calculated_steps = self.target_batch_tokens // (world_size * microbatch_size)

        # Ensure minimum of 1
        return max(1, calculated_steps)

    def __post_init__(self):
        """Validation after initialization."""
        if self.lr <= 0:
            raise ValueError("Learning rate must be positive")
        if self.accumulate_steps is not None and self.accumulate_steps <= 0:
            raise ValueError("Accumulate steps must be positive if specified")
        if self.target_batch_tokens <= 0:
            raise ValueError("Target batch tokens must be positive")


@dataclass
class LoggingConfig:
    """Logging configuration."""
    logger_func: Callable[[Dict], None]
    console_output: bool = True


@dataclass
class CheckpointConfig:
    """Checkpoint saving configuration."""
    save_dir: str
    save_interval_steps: int
    distributed_save: bool = False

    def should_save_step(self, step: TrainingStep, actual_steps: int) -> bool:
        """Determine if this step should trigger checkpoint saving."""
        if not step.should_save:
            return False
        return actual_steps % self.save_interval_steps == 0


@dataclass
class StrategyConfig:
    """Configuration for training strategy selection."""
    stage: TrainingStage
    training_config: TrainingConfig


@dataclass
class TrainerConfig:
    """Main trainer configuration."""
    stage: TrainingStage
    total_steps: int
    microbatch_size: int
    training_config: TrainingConfig
    save_dir: str
    logger_config: Optional['LoggerConfig'] = None  # Forward reference to avoid circular import
    data_dir: Optional[str] = None

    # Distill stage specific parameters
    pretrain_transformer_name: Optional[str] = None
    teacher_model_path: Optional[str] = None
