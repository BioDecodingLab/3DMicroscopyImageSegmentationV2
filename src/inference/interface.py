from pathlib import Path
from typing import Annotated

import typer

from src.dataset_paths import DatasetPaths
from src.inference.utils import run_inference


def interface(
    dataset_path: Annotated[Path, typer.Argument(help="Path to dataset root")],
):
    """Run inference on test data using trained models.

    Applies both classical methods (otsu, frangi, etc.) and deep learning models
    to generate segmentation predictions at patch-level and image-level.
    """
    paths = DatasetPaths(dataset_path)
    paths.validate_for_inference()

    typer.echo("Running inference...")
    run_inference(
        patches_dir=paths.test_reconstruction_patches / "images",
        images_dir=paths.test_images,
        models_dir=paths.models,
        predictions_patch_level=paths.predictions_patch_level,
        predictions_image_level=paths.predictions_image_level,
    )
    typer.echo("Inference complete!")


if __name__ == "__main__":
    typer.run(interface)
