"""
Focoos CLI Commands Module

This module contains all the command implementations for the Focoos CLI.
Commands can be used both with the Ultralytics-style syntax and as proper Typer commands.
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
