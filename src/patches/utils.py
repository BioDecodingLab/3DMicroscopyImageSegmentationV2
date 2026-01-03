import concurrent.futures
from pathlib import Path
from typing import Annotated, Any, Tuple

from loguru import logger
import numpy
from numpy.typing import NDArray
from patchify import patchify
import tifffile


def calculate_padding(
    image_shape: Tuple[int, int, int], patch_size: Tuple[int, int, int], step_size: int
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    padding = []
    for dim, patch_dim in zip(image_shape, patch_size):
        remainder = (dim - patch_dim) % step_size
        # Pad only after the dimension, not before
        padding.append((0, step_size - remainder))
    return tuple(padding)


def compute_padding_and_pad_image(
    image: NDArray[Any],
    mask: NDArray[Any],
    patch_size: Tuple[int, int, int],
    step_size: Tuple[int, int, int],
):
    padding_info_type = Annotated[
        Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]],
        "Padding information: ((z_before, z_after), (y_before, y_after), (x_before, x_after))",
    ]
    padding_info: padding_info_type = calculate_padding(image.shape, patch_size, step_size)
    padded_image = numpy.pad(image, padding_info, mode="reflect")
    padded_mask = numpy.pad(mask, padding_info, mode="reflect")

    return padded_image, padded_mask


def save_single_image_mask_patches(
    image_patch: NDArray[Any],
    mask_patch: NDArray[Any],
    image_patch_path: Path,
    mask_patch_path: Path,
):
    try:
        tifffile.imwrite(str(image_patch_path), image_patch)
        tifffile.imwrite(str(mask_patch_path), mask_patch)
    except Exception as e:
        logger.error(f"Error saving patch {image_patch_path.name}: {e}")
        raise


def create_and_save_patches_from_image_and_mask(
    image_path: str,
    image: NDArray[Any],
    mask: NDArray[Any],
    patch_size: Tuple[int, int, int],
    step_size: Tuple[int, int, int],
    output_dir_images: Path,
    output_dir_masks: Path,
    for_reconstruction: bool,
    num_workers: int,
):
    """
    In order to reconstruct an image, patchify needs the original dimensions of the patched image (patchify output) which are (z_patches, y_patches and x_patches) and if it is padded, we need the original dimension of the image to cut it
    """
    shape_image_without_padding = image.shape

    if for_reconstruction:
        image, mask = compute_padding_and_pad_image(image, mask, patch_size, step_size)

    image_patches = patchify(image, patch_size, step_size)
    mask_patches = patchify(mask, patch_size, step_size)

    # Store number of patches in each dimension
    n_patches_z, n_patches_y, n_patches_x = image_patches.shape[:3]

    # Flatten the images to get patches idx
    image_patches_flattened = image_patches.reshape(-1, *patch_size)
    mask_patches_flattened = mask_patches.reshape(-1, *patch_size)

    # Garbage collect the original array
    del image_patches, mask_patches

    # Create array of tasks to be passed into processes
    tasks = []
    for patch_index in range(len(image_patches_flattened)):
        # Save metadata in patch_filename
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

    # Use pool of processing for optimizing patch saving operations and handle creation and termination of processes systematically
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Create an array of futures
        futures = [executor.submit(save_single_image_mask_patches, *task) for task in tasks]
        # Wait for futures to be completed
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()  # Obtain the exception if something happened
            except Exception as e:
                logger.error(f"Patch saving failed {e}")
                raise
