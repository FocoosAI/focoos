"""Hub command implementation.

This module implements the hub-related commands for the Focoos CLI. It provides
functionality to interact with the Focoos Hub, including listing available models
and datasets, both private and shared resources.

The Hub commands allow users to:
- Browse and discover available pretrained models
- List private and shared datasets
- Get detailed information about model specifications
- Access dataset metadata and statistics

Examples:
    List available models:
    ```bash
    focoos hub models
    ```

    List datasets including shared ones:
    ```bash
    focoos hub datasets --include-shared
    ```

    Download a dataset:
    ```bash
    focoos hub dataset download --ref my-dataset --path ./data
    ```

    Upload a dataset:
    ```bash
    focoos hub dataset upload --ref my-dataset --path ./data
    ```

See Also:
    - [`focoos.hub.focoos_hub.FocoosHUB`][focoos.hub.focoos_hub.FocoosHUB]: Core hub functionality
"""

import os
from typing import Annotated, Optional

import typer

from focoos.hub.focoos_hub import FocoosHUB
from focoos.model_manager import ModelManager
from focoos.ports import MODELS_DIR, ArtifactName, HubSyncLocalTraining, ModelStatus

app = typer.Typer()
dataset_app = typer.Typer()
app.add_typer(dataset_app, name="dataset", help="Dataset operations (download, upload)")
model_app = typer.Typer()
app.add_typer(model_app, name="model", help="Model operations")


@app.callback()
def main():
    """Hub commands for interacting with the Focoos Hub.

    The Focoos Hub provides access to pretrained models and datasets.
    Use subcommands to list and discover available resources.

    Available subcommands:
    - `models`: List available pretrained models
    - `datasets`: List available datasets (private and optionally shared)
    - `dataset`: Dataset operations (download, upload)

    Examples:
        ```bash
        # List all available models
        focoos hub models

        # List your datasets
        focoos hub datasets

        # List datasets including shared ones
        focoos hub datasets --include-shared

        # Download a dataset
        focoos hub dataset download --ref my-dataset --path ./data

        # Upload a dataset
        focoos hub dataset upload --ref my-dataset --path ./data
        ```
    """
    pass


@app.command()
def models():
    """List all available pretrained models from the Focoos Hub.

    Retrieves and displays a comprehensive list of all pretrained models
    available on the Focoos Hub. For each model, shows detailed information
    including name, reference, task type, description, status, and model type.

    The displayed information includes:
    - **Name**: Human-readable model name
    - **Reference**: Unique model identifier for CLI usage
    - **Task**: Computer vision task (detection, segmentation, classification)
    - **Description**: Detailed model description and capabilities
    - **Status**: Model availability status
    - **Focoos Model**: Model architecture/family information

    Examples:
        ```bash
        # List all available models
        focoos hub models
        ```

        Output example:
        ```
        🔍 Found 12 model(s):

        📦 Model #1
           🏷️  Name: FAI DETR Medium COCO
           🔗 Reference: fai-detr-m-coco
           🎯 Task: object_detection
           📝 Description: Medium-sized DETR model trained on COCO dataset
           ⚡ Status: ready
           🤖 Focoos Model: fai_detr
        ```

    Raises:
        typer.Exit: If there's an error connecting to the Hub or no models are found.

    Note:
        This command requires an internet connection to access the Focoos Hub.
    """
    typer.echo("Listing models...")
    try:
        focoos_hub = FocoosHUB()
        models = focoos_hub.list_remote_models()
        if not models:
            typer.echo("❌ No models found.")
            return

        typer.echo(f"🔍 Found {len(models)} model(s):\n")

        for i, model in enumerate(models, 1):
            typer.echo(f"📦 Model #{i}")
            typer.echo(f"   🏷️  Name: {model.name}")
            typer.echo(f"   🔗 Reference: {model.ref}")
            typer.echo(f"   🎯 Task: {model.task}")
            typer.echo(f"   📝 Description: {model.description}")
            typer.echo(f"   ⚡ Status: {model.status}")
            typer.echo(f"   🤖 Focoos Model: {model.focoos_model}")

            if i < len(models):
                typer.echo("   " + "─" * 50)
                typer.echo("")
    except Exception as e:
        typer.echo(f"❌ Failed to list models: {e}")
        raise typer.Exit(1)


@app.command()
def datasets(
    include_shared: bool = typer.Option(False, help="Include shared/public datasets in addition to private ones"),
):
    """List available datasets from the Focoos Hub.

    Retrieves and displays datasets available on the Focoos Hub. By default,
    only shows private datasets associated with your account. Use the
    `--include-shared` flag to also include publicly shared datasets.

    For each dataset, displays comprehensive information including:
    - **Name**: Human-readable dataset name
    - **Reference**: Unique dataset identifier for CLI usage
    - **Task**: Computer vision task type
    - **Layout**: Dataset format/structure
    - **Description**: Dataset description and details
    - **Statistics**: Training/validation split sizes and total size

    Args:
        include_shared (bool, optional): Whether to include shared/public datasets
            in addition to private ones. Defaults to False.

    Examples:
        ```bash
        # List only your private datasets
        focoos hub datasets

        # List both private and shared datasets
        focoos hub datasets --include-shared
        ```

        Output example:
        ```
        🔍 Found 3 dataset(s):

        📦 Dataset #1
           🏷️  Name: My Custom Dataset
           🔗 Reference: my-custom-dataset
           🎯 Task: object_detection
           📋 Layout: roboflow_coco
           📝 Description: Custom dataset for object detection
           🤖 Train Length: 1000
           🤖 Val Length: 200
           🤖 Size MB: 150.5
        ```

    Raises:
        typer.Exit: If there's an error connecting to the Hub or no datasets are found.

    Note:
        This command requires an internet connection to access the Focoos Hub.
        Shared datasets may require appropriate permissions to access.
    """
    typer.echo("Listing datasets...")
    try:
        focoos_hub = FocoosHUB()
        datasets = focoos_hub.list_remote_datasets(include_shared=include_shared)
        if not datasets:
            dataset_type = "shared and private" if include_shared else "private"
            typer.echo(f"❌ No {dataset_type} datasets found.")
            return

        dataset_type_desc = "shared and private" if include_shared else "private"
        typer.echo(f"🔍 Found {len(datasets)} {dataset_type_desc} dataset(s):\n")

        for i, dataset in enumerate(datasets, 1):
            typer.echo(f"📦 Dataset #{i}")
            typer.echo(f"   🏷️  Name: {dataset.name}")
            typer.echo(f"   🔗 Reference: {dataset.ref}")
            typer.echo(f"   🎯 Task: {dataset.task}")
            typer.echo(f"   📋 Layout: {dataset.layout}")
            typer.echo(f"   📝 Description: {dataset.description}")
            if dataset.spec:
                typer.echo(f"   📊 Train Length: {dataset.spec.train_length}")
                typer.echo(f"   📊 Val Length: {dataset.spec.valid_length}")
                typer.echo(f"   📊 Size MB: {dataset.spec.size_mb}")

            if i < len(datasets):
                typer.echo("   " + "─" * 50)
                typer.echo("")
    except Exception as e:
        typer.echo(f"❌ Failed to list datasets: {e}")
        raise typer.Exit(1)


@dataset_app.command()
def download(
    ref: Annotated[str, typer.Option(..., help="The reference ID of the dataset to download")],
    path: Annotated[Optional[str], typer.Option(help="Path to download the dataset to")] = None,
):
    """Download a dataset from the Focoos Hub.

    Downloads a dataset from the Focoos Hub to a specified local path.

    Args:
        ref (str): The reference ID of the dataset to download
        path (str): Path to download the dataset to

    Examples:
        ```bash
        # Download a dataset to a specific path
        focoos hub dataset download --ref my-dataset-ref --path ./data
        ```

    Raises:
        typer.Exit: If there's an error connecting to the Hub or dataset not found.

    Note:
        This command requires an internet connection to access the Focoos Hub.
    """
    typer.echo(f"Downloading dataset from {ref} to {path}...")
    try:
        focoos_hub = FocoosHUB()
        dataset = focoos_hub.get_remote_dataset(ref)
        if path:
            dataset.download_data(path)
        else:
            dataset.download_data()
    except Exception as e:
        typer.echo(f"❌ Failed to download dataset: {e}")
        raise typer.Exit(1)


@dataset_app.command()
def upload(
    ref: Annotated[str, typer.Option(..., help="The reference ID of the dataset to upload")],
    path: Annotated[str, typer.Option(..., help="Path to upload the dataset from")],
):
    """Upload a dataset to the Focoos Hub.

    Uploads a dataset to the Focoos Hub from a specified local path.

    Args:
        ref (str): The reference ID of the dataset to upload
        path (str): Path to upload the dataset from

    Examples:
        ```bash
        # Upload a dataset from a specific path
        focoos hub dataset upload --ref my-dataset-ref --path ./data
        ```

    Raises:
        typer.Exit: If there's an error connecting to the Hub or upload fails.

    Note:
        This command requires an internet connection to access the Focoos Hub.
    """
    typer.echo(f"Uploading dataset from {path} to {ref}...")

    try:
        focoos_hub = FocoosHUB()
        dataset = focoos_hub.get_remote_dataset(ref)
        spec = dataset.upload_data(path)
        dataset_info = dataset.get_info()
        if not spec:
            raise Exception("Failed to upload dataset.")
        typer.echo("✅ Dataset uploaded successfully!")
        typer.echo(f"   🏷️  Name: {dataset_info.name}")
        typer.echo(f"   🔗 Reference: {dataset_info.ref}")
        typer.echo(f"   🎯 Task: {dataset_info.task}")
        typer.echo(f"   📋 Layout: {dataset_info.layout}")
        typer.echo(f"   📝 Description: {dataset_info.description}")
        typer.echo(f"   📊 Train Length: {spec.train_length}")
        typer.echo(f"   📊 Val Length: {spec.valid_length}")
        typer.echo(f"   📊 Size MB: {spec.size_mb}")
    except Exception as e:
        typer.echo(f"❌ Failed to upload dataset: {e}")
        raise typer.Exit(1)


@model_app.command()
def sync(
    path: Annotated[str, typer.Option(..., help="Path to sync the model to")],
):
    """Sync a local model to the Focoos Hub.

    Syncs a local model to the Focoos Hub from a specified local path.

    Args:
        path (str): Path to sync the model to
    """
    typer.echo(f"Syncing model from {path} to the Focoos Hub...")
    model_info = ModelManager._from_local_dir(path)
    typer.echo(f"Model info: {model_info}")
    focoos_hub = FocoosHUB()
    remote_model = focoos_hub.new_model(model_info)

    model_dir = path if os.path.exists(path) else os.path.join(MODELS_DIR, path)

    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory {model_dir} does not exist")

    remote_model.sync_local_training_job(
        local_training_info=HubSyncLocalTraining(
            training_info=model_info.training_info,
            status=ModelStatus.TRAINING_COMPLETED,
            focoos_version=model_info.focoos_version,
        ),
        dir=model_dir,
        upload_artifacts=[ArtifactName.INFO, ArtifactName.METRICS, ArtifactName.WEIGHTS],
    )
