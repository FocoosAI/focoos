"""Focoos CLI Package.

This package provides the command-line interface for the Focoos computer vision framework.
The CLI offers a modern, user-friendly interface for training, validation, inference,
and model management operations.

**Architecture:**
The CLI uses a lazy-loading architecture to avoid circular dependencies and improve
startup performance. The main CLI application is accessed through the `get_app()`
function, which returns a fully configured Typer application instance.

**Key Features:**
- **Modern CLI Design**: Built with Typer for rich, interactive command-line experience
- **Comprehensive Commands**: Full coverage of Focoos functionality
- **Type Safety**: Built-in argument validation and type checking
- **Rich Output**: Colored output, progress bars, and detailed feedback
- **Programmatic API**: Commands can be used both via CLI and Python imports

**Available Commands:**
- `train`: Train computer vision models on datasets
- `val`: Validate model performance and accuracy
- `predict`: Run inference on images and videos
- `export`: Export models to different formats (ONNX, TorchScript)
- `benchmark`: Benchmark model performance and speed
- `hub`: Interact with Focoos Hub for models and datasets
- `version`: Display version information
- `checks`: Run system diagnostics
- `settings`: Show configuration settings

**Usage Examples:**
```bash
# Install and use via command line
pip install focoos
focoos train --model fai-detr-m-coco --dataset mydataset.zip

# Access programmatically
from focoos.cli import get_app
app = get_app()
```

**Lazy Loading Pattern:**
The CLI uses lazy loading to avoid importing heavy dependencies during package
initialization. This improves import speed and prevents circular dependency issues.

See Also:
    - [`focoos.cli.cli`][focoos.cli.cli]: Main CLI application implementation
    - [`focoos.cli.commands`][focoos.cli.commands]: Individual command implementations
"""


# Import will be done lazily to avoid circular dependencies
def get_app():
    """Get the CLI app instance.

    Returns a lazily-loaded Typer application instance for the Focoos CLI.
    This approach avoids circular dependencies during module initialization
    and improves package import performance.

    The returned application includes all CLI commands and is fully configured
    with proper help text, argument validation, and error handling.

    Returns:
        typer.Typer: The configured CLI application instance with all commands
            registered and ready for execution.

    Examples:
        ```python
        # Get the CLI app
        from focoos.cli import get_app

        app = get_app()

        # Access app info
        print(app.info.name)  # "focoos"

        # Can be used with typer testing utilities
        from typer.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(app, ["version"])
        ```

    Note:
        This function performs the actual import of the CLI module, which may
        take additional time on first call as it loads all CLI dependencies.
    """
    from .cli import app

    return app


__all__ = ["get_app"]
