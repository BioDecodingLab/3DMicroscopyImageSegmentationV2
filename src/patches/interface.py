from pathlib import Path

import typer
from typing_extensions import Annotated

from src.dataset_paths import DatasetPaths

from .dataset import generate_patches


def interface(
    dataset_path: Annotated[Path, typer.Argument(help="Path to dataset root")],
):
    """Generate patches for training and test data.

    This generates:
    - Regular patches for training_data (no padding, used for training)
    - Regular patches for test_data (no padding, used for patch-level evaluation)
    - Reconstruction patches for test_data (with padding, used for image reconstruction)
    """
    paths = DatasetPaths(dataset_path)
    paths.validate_for_patches()

    # Generate regular patches for training (no padding)
    typer.echo("Generating regular patches for training data...")
    generate_patches(
        images_dir=paths.training_images,
        masks_dir=paths.training_masks,
        output_dir=paths.training_regular_patches,
        for_reconstruction=False,
    )

    # Generate regular patches for test data (no padding, for patch-level evaluation)
    typer.echo("Generating regular patches for test data...")
    generate_patches(
        images_dir=paths.test_images,
        masks_dir=paths.test_masks,
        output_dir=paths.test_regular_patches,
        for_reconstruction=False,
    )

    # Generate reconstruction patches for test data (with padding, for image reconstruction)
    typer.echo("Generating reconstruction patches for test data...")
    generate_patches(
        images_dir=paths.test_images,
        masks_dir=paths.test_masks,
        output_dir=paths.test_reconstruction_patches,
        for_reconstruction=True,
    )

    typer.echo("Patch generation complete!")


if __name__ == "__main__":
    typer.run(interface)
