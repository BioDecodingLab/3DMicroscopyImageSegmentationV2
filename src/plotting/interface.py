from pathlib import Path
from typing import Annotated

import typer

from src.dataset_paths import DatasetPaths
from src.plotting.utils import run_plotting


def interface(
    dataset_path: Annotated[Path, typer.Argument(help="Path to dataset root")],
):
    """Generate evaluation plots comparing segmentation methods.

    Computes metrics (accuracy, precision, recall, dice, etc.) for all
    predictions and generates boxplot visualizations.
    """
    paths = DatasetPaths(dataset_path)
    paths.validate_for_plotting()

    typer.echo("Generating evaluation plots...")
    run_plotting(
        predictions_patch_level=paths.predictions_patch_level,
        predictions_image_level=paths.predictions_image_level,
        ground_truth_patches_dir=paths.test_reconstruction_patches / "masks",
        ground_truth_masks_dir=paths.test_masks,
        output_dir=paths.figures,
    )
    typer.echo("Plotting complete!")


if __name__ == "__main__":
    typer.run(interface)
