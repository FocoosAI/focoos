"""Focoos CLI Commands Module.

This module contains all the command implementations for the Focoos Command Line Interface.
Each command is implemented as a standalone function that can be used both programmatically
and through the CLI interface.

The module provides implementations for the core Focoos functionality:

**Training Commands:**
- [`train_command`][focoos.cli.commands.train.train_command]: Train computer vision models
- [`val_command`][focoos.cli.commands.val.val_command]: Validate model performance

**Inference Commands:**
- [`predict_command`][focoos.cli.commands.predict.predict_command]: Run inference on images

**Model Management Commands:**
- [`export_command`][focoos.cli.commands.export.export_command]: Export models to different formats
- [`benchmark_command`][focoos.cli.commands.benchmark.benchmark_command]: Benchmark model performance

**Architecture:**
Each command follows a consistent pattern:
1. **Input validation**: Validates and processes command arguments
2. **Resource loading**: Loads models, datasets, and other required resources
3. **Core operation**: Executes the main command functionality
4. **Result handling**: Processes and saves results, provides user feedback


See Also:
    - [`focoos.cli.cli`][focoos.cli.cli]: Main CLI application
    - [`focoos.model_manager`][focoos.model_manager]: Model management utilities
    - [`focoos.data.auto_dataset`][focoos.data.auto_dataset]: Dataset handling
"""

from .benchmark import benchmark_command
from .export import export_command
from .predict import predict_command
from .train import train_command
from .val import val_command

__all__ = [
    "train_command",
    "val_command",
    "predict_command",
    "export_command",
    "benchmark_command",
]
