"""Focoos CLI Package."""


# Import will be done lazily to avoid circular dependencies
def get_app():
    """Get the CLI app instance.

    Returns a lazily-loaded Typer application instance for the Focoos CLI.
    This approach avoids circular dependencies during module initialization.

    Returns:
        typer.Typer: The configured CLI application instance.
    """
    from .cli import app

    return app


__all__ = ["get_app"]
