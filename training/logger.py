"""Logger system for H-Net training with support for wandb, swanlab, and console logging."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Callable
import warnings

from .data import DistributedState


class LoggerType(Enum):
    """Available logger types."""
    WANDB = "wandb"
    SWANLAB = "swanlab"
    CONSOLE = "console"
    LOCAL = "local"  # Alias for console


@dataclass
class LoggerConfig:
    """Configuration for logger initialization."""
    logger_type: LoggerType
    project_name: str = "hnet"
    experiment_name: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=lambda: {"step": 80000, "beta": 0.1})
    watch_model: bool = False
    watch_freq: int = 4800

    @classmethod
    def from_string(cls, logger_type_str: str, **kwargs) -> 'LoggerConfig':
        """Create LoggerConfig from string type."""
        # Handle legacy 'local' type
        if logger_type_str == 'local':
            logger_type_str = 'console'

        try:
            logger_type = LoggerType(logger_type_str.lower())
        except ValueError:
            warnings.warn(f"Unknown logger type '{logger_type_str}', falling back to console")
            logger_type = LoggerType.CONSOLE

        return cls(logger_type=logger_type, **kwargs)


class Logger(ABC):
    """Abstract base class for all loggers."""

    def __init__(self, config: LoggerConfig, distributed_state: DistributedState):
        self.config = config
        self.distributed_state = distributed_state
        self._is_initialized = False

    @abstractmethod
    def initialize(self, model=None) -> bool:
        """Initialize the logger. Returns True if successful."""
        pass

    @abstractmethod
    def log(self, metrics: Dict[str, Any]) -> None:
        """Log metrics."""
        pass

    @abstractmethod
    def finish(self) -> None:
        """Clean up and finish logging."""
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if logger is initialized."""
        return self._is_initialized


class ConsoleLogger(Logger):
    """Simple console logger that prints metrics."""

    def initialize(self, model=None) -> bool:
        """Initialize console logger."""
        self._is_initialized = True
        if self.distributed_state.is_main_process:
            print(f"🖥️  Console Logger initialized for project: {self.config.project_name}")
        return True

    def log(self, metrics: Dict[str, Any]) -> None:
        """Print metrics to console."""
        if not self.distributed_state.is_main_process:
            return

        if not self._is_initialized:
            return

        # Format metrics for readable console output
        formatted_metrics = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_metrics.append(f"{key}={value:.4f}")
            else:
                formatted_metrics.append(f"{key}={value}")

        print(f"📊 Metrics: {' | '.join(formatted_metrics)}")

    def finish(self) -> None:
        """Finish console logging."""
        if self.distributed_state.is_main_process:
            print("🏁 Console logging finished")


class WandbLogger(Logger):
    """Weights & Biases logger."""

    def __init__(self, config: LoggerConfig, distributed_state: DistributedState):
        super().__init__(config, distributed_state)
        self._wandb = None

    def initialize(self, model=None) -> bool:
        """Initialize wandb logger."""
        if not self.distributed_state.is_main_process:
            # In distributed training, only main process initializes wandb
            self._is_initialized = True
            return True

        try:
            import wandb
            self._wandb = wandb

            # Initialize wandb run
            wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                config=self.config.config
            )

            # Watch model if requested
            if model is not None and self.config.watch_model:
                wandb.watch(model, log_freq=self.config.watch_freq)

            self._is_initialized = True
            print(f"🚀 Wandb Logger initialized for project: {self.config.project_name}")
            return True

        except ImportError:
            warnings.warn("wandb not installed. Install with: pip install wandb")
            return False
        except Exception as e:
            warnings.warn(f"Failed to initialize wandb: {e}")
            return False

    def log(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to wandb."""
        if not self.distributed_state.is_main_process or not self._is_initialized:
            return

        if self._wandb is None:
            return

        try:
            self._wandb.log(metrics)
        except Exception as e:
            warnings.warn(f"Failed to log to wandb: {e}")

    def finish(self) -> None:
        """Finish wandb logging."""
        if self.distributed_state.is_main_process and self._wandb is not None:
            try:
                self._wandb.finish()
                print("🏁 Wandb logging finished")
            except Exception as e:
                warnings.warn(f"Failed to finish wandb: {e}")


class SwanlabLogger(Logger):
    """Swanlab logger."""

    def __init__(self, config: LoggerConfig, distributed_state: DistributedState):
        super().__init__(config, distributed_state)
        self._swanlab = None

    def initialize(self, model=None) -> bool:
        """Initialize swanlab logger."""
        if not self.distributed_state.is_main_process:
            # In distributed training, only main process initializes swanlab
            self._is_initialized = True
            return True

        try:
            import swanlab
            self._swanlab = swanlab

            # Initialize swanlab run
            swanlab.init(
                project=self.config.project_name,
                name=self.config.experiment_name,
                config=self.config.config
            )

            self._is_initialized = True
            print(f"🦢 Swanlab Logger initialized for project: {self.config.project_name}")
            return True

        except ImportError:
            warnings.warn("swanlab not installed. Install with: pip install swanlab")
            return False
        except Exception as e:
            warnings.warn(f"Failed to initialize swanlab: {e}")
            return False

    def log(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to swanlab."""
        if not self.distributed_state.is_main_process or not self._is_initialized:
            return

        if self._swanlab is None:
            return

        try:
            self._swanlab.log(metrics)
        except Exception as e:
            warnings.warn(f"Failed to log to swanlab: {e}")

    def finish(self) -> None:
        """Finish swanlab logging."""
        if self.distributed_state.is_main_process and self._swanlab is not None:
            try:
                # Swanlab doesn't have explicit finish method, just print
                print("🏁 Swanlab logging finished")
            except Exception as e:
                warnings.warn(f"Failed to finish swanlab: {e}")


class LoggerFactory:
    """Factory for creating logger instances."""

    _logger_classes = {
        LoggerType.CONSOLE: ConsoleLogger,
        LoggerType.LOCAL: ConsoleLogger,  # Alias
        LoggerType.WANDB: WandbLogger,
        LoggerType.SWANLAB: SwanlabLogger,
    }

    @classmethod
    def create(cls, config: LoggerConfig, distributed_state: DistributedState,
               model=None) -> Logger:
        """Create a logger instance based on configuration."""

        logger_class = cls._logger_classes.get(config.logger_type)
        if logger_class is None:
            warnings.warn(f"Unknown logger type: {config.logger_type}, using console logger")
            logger_class = ConsoleLogger

        # Create logger instance
        logger = logger_class(config, distributed_state)

        # Try to initialize, fallback to console if failed
        if not logger.initialize(model):
            if config.logger_type != LoggerType.CONSOLE:
                warnings.warn(f"Failed to initialize {config.logger_type.value} logger, "
                             f"falling back to console logger")
                fallback_config = LoggerConfig(
                    logger_type=LoggerType.CONSOLE,
                    project_name=config.project_name
                )
                logger = ConsoleLogger(fallback_config, distributed_state)
                logger.initialize(model)

        return logger

    @classmethod
    def create_callable(cls, config: LoggerConfig, distributed_state: DistributedState,
                       model=None) -> Callable[[Dict[str, Any]], None]:
        """Create a logger and return its log method as a callable."""
        logger = cls.create(config, distributed_state, model)
        return logger.log
