from pathlib import Path

import tifffile
from tqdm import tqdm

from src.config import (
    ALLOWED_EXTENSIONS,
    MAX_WORKERS,
    PATCH_SIZE,
    PATCH_STEP,
)
from src.utils import overwrite_and_create_directory

from .utils import create_and_save_patches_from_image_and_mask


def generate_patches(
    images_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    for_reconstruction: bool,
) -> None:
    """Generate patches from images and masks.

    Args:
        images_dir: Directory containing source images
        masks_dir: Directory containing corresponding masks
        output_dir: Directory to write patches (e.g., regular_patches or reconstruction_patches)
        for_reconstruction: If True, pad images for reconstruction

    Note: Validation should be done via DatasetPaths.validate_for_patches() before calling.
    """
    # Create output directory structure; if it exists, delete it
    overwrite_and_create_directory(output_dir)

    patches_images_dir = output_dir / "images"
    patches_masks_dir = output_dir / "masks"
    overwrite_and_create_directory(patches_images_dir)
    overwrite_and_create_directory(patches_masks_dir)

    # Get corresponding pairs of images and masks
    image_mask_pairs = [
        (image, masks_dir / image.name)
        for image in images_dir.glob("*")
        if image.suffix.lower() in ALLOWED_EXTENSIONS
    ]

    for image_path, mask_path in tqdm(image_mask_pairs, desc="Processing images"):
        try:
            image = tifffile.imread(str(image_path))
            mask = tifffile.imread(str(mask_path))

            # Verify 3D dimensions
            if len(image.shape) != 3 or len(mask.shape) != 3:
                raise ValueError(
                    f"Expected 3D image but got shape {image.shape} for {image_path.name}"
                )

            # Create output directories for patches of this image
            image_patches_dir = patches_images_dir / image_path.name
            mask_patches_dir = patches_masks_dir / image_path.name

            overwrite_and_create_directory(image_patches_dir)
            overwrite_and_create_directory(mask_patches_dir)

            create_and_save_patches_from_image_and_mask(
                image_path,
                image,
                mask,
                PATCH_SIZE,
                PATCH_STEP,
                image_patches_dir,
                mask_patches_dir,
                for_reconstruction,
                MAX_WORKERS,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to generate patches for {image_path}: {e}") from e
