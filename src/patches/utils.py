import concurrent.futures
from pathlib import Path
from typing import Any, Tuple

from loguru import logger
import numpy
from numpy.typing import NDArray
from patchify import patchify
import tifffile
from tqdm import tqdm

from src.config import (
    ALLOWED_EXTENSIONS,
    MAX_WORKERS,
    PATCH_SIZE,
    PATCH_STEP,
)
from src.utils import overwrite_and_create_directory


def calculate_padding(
    image_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    step_size: Tuple[int, int, int],
) -> Tuple[Tuple[int, int], ...]:
    """Calculate padding needed for an image to be evenly divisible into patches."""
    padding = []
    for dim, patch_dim, step in zip(image_shape, patch_size, step_size):
        remainder = (dim - patch_dim) % step
        # Pad only after the dimension, not before
        padding.append((0, step - remainder if remainder != 0 else 0))
    return tuple(padding)


def compute_padding_and_pad_image(
    image: NDArray[Any],
    mask: NDArray[Any],
    patch_size: Tuple[int, int, int],
    step_size: Tuple[int, int, int],
) -> Tuple[NDArray[Any], NDArray[Any]]:
    """Pad image and mask to be evenly divisible into patches."""
    padding_info = calculate_padding(image.shape, patch_size, step_size)
    padded_image = numpy.pad(image, padding_info, mode="reflect")
    padded_mask = numpy.pad(mask, padding_info, mode="reflect")
    return padded_image, padded_mask


def save_single_image_mask_patches(
    image_patch: NDArray[Any],
    mask_patch: NDArray[Any],
    image_patch_path: Path,
    mask_patch_path: Path,
) -> None:
    """Save a single image-mask patch pair to disk."""
    try:
        tifffile.imwrite(str(image_patch_path), image_patch)
        tifffile.imwrite(str(mask_patch_path), mask_patch)
    except Exception as e:
        logger.error(f"Error saving patch {image_patch_path.name}: {e}")
        raise


def create_and_save_patches_from_image_and_mask(
    image_path: Path,
    image: NDArray[Any],
    mask: NDArray[Any],
    patch_size: Tuple[int, int, int],
    step_size: Tuple[int, int, int],
    output_dir_images: Path,
    output_dir_masks: Path,
    for_reconstruction: bool,
    num_workers: int,
) -> None:
    """Create patches from an image-mask pair and save them to disk.

    Patch filenames encode metadata needed for reconstruction:
    - Original image shape
    - Padded shape (if for_reconstruction=True)
    - Number of patches in each dimension
    - Patch index
    """
    shape_image_without_padding = image.shape

    if for_reconstruction:
        image, mask = compute_padding_and_pad_image(image, mask, patch_size, step_size)

    image_patches = patchify(image, patch_size, step_size)
    mask_patches = patchify(mask, patch_size, step_size)

    # Store number of patches in each dimension
    n_patches_z, n_patches_y, n_patches_x = image_patches.shape[:3]

    # Flatten the patches for iteration
    image_patches_flattened = image_patches.reshape(-1, *patch_size)
    mask_patches_flattened = mask_patches.reshape(-1, *patch_size)

    # Free memory from the original arrays
    del image_patches, mask_patches

    # Build tasks for parallel saving
    tasks = []
    for patch_index in range(len(image_patches_flattened)):
        patch_filename = (
            f"{image_path.stem}_"
            f"orig_{shape_image_without_padding[0]}_{shape_image_without_padding[1]}_{shape_image_without_padding[2]}_"
            f"{'' if not for_reconstruction else f'pad_{image.shape[0]}_{image.shape[1]}_{image.shape[2]}_'}"
            f"npatches_{n_patches_z}_{n_patches_y}_{n_patches_x}_"
            f"patch_{patch_index:04d}.tif"
        )
        tasks.append(
            (
                image_patches_flattened[patch_index],
                mask_patches_flattened[patch_index],
                output_dir_images / patch_filename,
                output_dir_masks / patch_filename,
            )
        )

    # Save patches in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(save_single_image_mask_patches, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Patch saving failed: {e}")
                raise


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
