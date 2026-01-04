import concurrent.futures
from dataclasses import dataclass
import itertools
from pathlib import Path
import re
from typing import List, Tuple

import cv2
from loguru import logger
import numpy as np
from numpy.typing import NDArray
from patchify import unpatchify
from skimage.filters import frangi
import tensorflow
import tifffile

from src.config import (
    AVAILABLE_AUGMENTATIONS,
    AVAILABLE_CLASSICAL_METHODS,
    AVAILABLE_MODELS,
    BATCH_SIZE,
    MAX_WORKERS,
    PATCH_SIZE,
)
from src.utils import create_directory, overwrite_and_create_directory


@dataclass
class ImageMetadata:
    "Information necessary to reconstruct an image from patches"

    image_name: str
    original_shape: Tuple[int, int, int]
    padded_shape: Tuple[int, int, int] | None
    number_of_patches: int
    patch_id: int


def extract_patch_info_from_path(path: Path) -> ImageMetadata:
    """Extract information from patch path.

    Args:
        filename: Patch filename (e.g. '.../image1_orig_512_512_128_pad_520_520_130_npatches_4_8_8_patch_0000.tif')
    """
    # Extract information using regex
    pattern = r"(.+)_orig_(\d+)_(\d+)_(\d+)(?:_pad_(\d+)_(\d+)_(\d+))?_npatches_(\d+)_(\d+)_(\d+)_patch_(\d+)\.tiff?"
    match = re.match(pattern, path.name)

    if not match:
        raise ValueError(f"Invalid patch filename format: {path}")

    image_name = match.group(1)
    orig_shape = (int(match.group(2)), int(match.group(3)), int(match.group(4)))

    # Get padded shape if it exists, otherwise use original shape
    if match.group(5):
        padded_shape = (int(match.group(5)), int(match.group(6)), int(match.group(7)))
    else:
        padded_shape = None

    n_patches = (int(match.group(8)), int(match.group(9)), int(match.group(10)))

    patch_id = int(match.group(11))

    return ImageMetadata(image_name, orig_shape, padded_shape, n_patches, patch_id)


def normalize_image_to_0_1(image: NDArray):
    return (image - image.min()) / (image.max() - image.min() + np.finfo(float).eps)


def normalize_image_to_0_255(image: NDArray) -> NDArray[np.uint8]:
    return (normalize_image_to_0_1(image) * 255).astype(np.uint8)


def apply_threshold_to_image_and_convert_to_dtype(image: NDArray, threshold: int, dtype):
    return (image > threshold).astype(dtype)


def apply_classical_thresholding_method_to_2D_image(image: NDArray, method: str):
    normalized_image = normalize_image_to_0_255(image)

    if method == "otsu":
        _, mask = cv2.threshold(normalized_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive_mean":
        mask = cv2.adaptiveThreshold(
            normalized_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, 2
        )
    elif method == "adaptive_gaussian":
        mask = cv2.adaptiveThreshold(
            normalized_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, 2
        )
    elif method == "frangi":
        frangi_result = frangi(normalized_image)

        # Apply threshold to frangi result
        _, mask = cv2.threshold(
            normalize_image_to_0_255(frangi_result), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        raise ValueError(f"Thresholding method {method} not implemented")

    # Ensure that mask is of 1 and 0's, and that it is of type np.uint8
    return apply_threshold_to_image_and_convert_to_dtype(mask, 0, np.uint8)


def apply_classical_thresholding_method_to_3D_image(image: NDArray, method: str):
    # Apply classical thresholding to each slice in the z axis
    masks_from_slices = [
        apply_classical_thresholding_method_to_2D_image(image[z], method)
        for z in range(image.shape[0])
    ]

    # Merge masks into a sole volume
    return np.stack(masks_from_slices, axis=0)


def save_mask_in_disk(mask: NDArray, output_dir: Path):
    try:
        tifffile.imwrite(output_dir, mask)
    except Exception as e:
        raise Exception(f"The mask {output_dir} couldn't be saved: {e}")


# Create a single function to apply a thresholding method and save mask in order to execute it in a multiprocessing environment
def apply_method_and_save_mask(image_path: Path, method: str, save_dir: Path):
    image = tifffile.imread(str(image_path))
    mask = apply_classical_thresholding_method_to_3D_image(image, method)
    # Save in save_dir with the same name and extension
    save_mask_in_disk(mask, save_dir / image_path.name)


def apply_classical_thresholding_and_save_masks_for_array_of_filenames(
    array_of_patch_or_images_filenames: List[Path],
    save_dir_for_patches: Path,
    method: str,
    max_workers: int,
):
    save_dir = save_dir_for_patches / method
    create_directory(save_dir)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(apply_method_and_save_mask, image_path, method, save_dir)
            for image_path in array_of_patch_or_images_filenames
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"There was an error applying and saving a mask: {e}")


def extract_information_from_model_dir_path(model_path: Path):
    # Extract model name and augmentation type from model dir
    # The path format is: models_dir/dataset_name/model_name_augmentation

    parts = model_path.name.split("_")

    if len(parts) < 2:
        raise Exception(
            f"Invalid model directory name format. Expected form 'model_name_augmentation', i.e., UNet3D_OURS, got {model_path.name}"
        )

    # Return model name and augmentation
    return parts[0], parts[1]


def get_deep_learning_models_from_dir(
    models_dir: Path, available_models: List[str], available_augmentations: List[str]
):
    """
    The format of a models directory must be

    DATASET_NAME
        MODEL1_AUGMENTATION ("UNet3D_OURS")
            TIMESTAMP1
                best_model.h5
            TIMESTAMP2
            |   best_model.h5
            TIMESTAMP3
        MODEL2_AUGMENTATION ("AttentionUNet3D_STANDARD")

    This function assumes that there is one and only one best model per timestamp (called best_model.h5)

    It creates a dictionary mapping (model_name_augmentation_type) -> (path_to_model, augmentation_type)
    (UNet3D_OURS) -> (path_to_model, OURS)
    """

    if not models_dir.exists():
        raise Exception(f"Models directory doesn't exist: {models_dir}")

    if any((dir.stem).split("_")[0] not in available_models for dir in models_dir.glob("*")):
        raise Exception("Models' directory contains an invalid model")

    if any(
        (dir.stem).split("_")[1] not in available_augmentations for dir in models_dir.glob("*")
    ):
        raise Exception("Models' directory contains an invalid augmentation")

    models_list = []

    for model_dir in models_dir.glob("*/"):
        model_name, model_augmentation = extract_information_from_model_dir_path(model_dir)

        # Get timestamp directories within model_dir
        timestamps = list(model_dir.glob("*/"))

        if not timestamps:
            raise Exception(f"No timestamp directories in {model_dir}")

        # Get most recent timestamp directory
        latest_timestamp_model = sorted(timestamps, key=lambda d: d.name)[-1]

        # Get model files and check there's at least one .h5 model
        model_files = list(latest_timestamp_model.glob("*.h5"))
        if not model_files:
            raise Exception(f"No .h5 model files in {latest_timestamp_model}")

        best_model_path = model_files[0]
        logger.info(f"Found {model_name}_{model_augmentation} at {best_model_path}")

        models_list.append((model_name, model_augmentation, best_model_path))

    return models_list


def apply_deep_learning_model_to_batch(batch: list[NDArray], model, threshold: int) -> NDArray:
    # Normalize batch
    batch_normalized = np.stack([normalize_image_to_0_1(patch) for patch in batch])

    # Add a fourth channel
    batch_input_to_model = batch_normalized[..., np.newaxis]

    # Get preditions and delete fourth channel
    batch_preds = model.predict(batch_input_to_model, verbose=0)[..., 0]

    # Threshold the masks given by the model
    batch_pred_threshold = apply_threshold_to_image_and_convert_to_dtype(
        batch_preds, threshold, np.uint8
    )

    return batch_pred_threshold


def batch_iterable(iterable, n):
    """
    Implementation for batching iterables.
    In Python 3.10, there's no itertools.batched()

    >>> a = ['a.tiff', 'b.tiff', 'c.tiff', 'd.tiff']
    >>> list(batch_iterable(e, 2))a
    [['a.tiff', 'b.tiff'], ['c.tiff', 'd.tiff']]
    """
    if n < 1:
        raise Exception("Batch should be greater than 1")
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, n)):
        yield batch


def reconstruct_image_from_patches_and_metadata(
    patches: list[NDArray], metadata: ImageMetadata, patch_size: tuple[int, int, int]
):
    if metadata.padded_shape is None:
        raise Exception(
            "Image can't be reconstructed from regular patches, use reconstruction patches instead"
        )

    # Convert list of patches to np array and reshape patches to (z_patches, y_patches, x_patches, patch_size_z, patch_size_y, patch_size_x)
    patches_reshaped = np.array(patches).reshape(*metadata.number_of_patches, *patch_size)

    # Image is padded, so you have to reconstruct it with the padded_shape
    reconstructed_image = unpatchify(patches_reshaped, metadata.padded_shape)

    # Cut the padding with the original size of the image
    reconstructed_image = reconstructed_image[tuple(slice(0, s) for s in metadata.original_shape)]

    return reconstructed_image


def apply_deep_learning_method_to_array_of_filenames(
    array_of_patch_complete_paths: list[Path],
    save_dir_for_patches_predictions: Path,
    save_dir_for_complete_images_preditions: Path,
    model_name: str,
    model_augmentation: str,
    model_path: Path,
    batch_size: int,
    patch_size: tuple[int, int, int],
    max_workers: int,
):
    # Create folder to save predictions for a defined model
    create_directory(save_dir_for_patches_predictions / f"{model_name}_{model_augmentation}")
    create_directory(
        save_dir_for_complete_images_preditions / f"{model_name}_{model_augmentation}"
    )

    # Load model
    model = tensorflow.keras.models.load_model(model_path)

    # Create an array of predictions
    predictions = []

    # Get metadata from the first patch
    metadata = extract_patch_info_from_path(array_of_patch_complete_paths[0])

    # Create batches of filenames
    for batch_of_filenames in batch_iterable(array_of_patch_complete_paths, batch_size):
        # Create batches of patches
        batch_of_patches = map(tifffile.imread, batch_of_filenames)

        # Predict batch
        prediction_of_batch = apply_deep_learning_model_to_batch(batch_of_patches, model, 0.5)

        # Add prediction of batch to predictions
        predictions.extend(prediction_of_batch)

    # Use multiprogressing to save the patches
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                save_mask_in_disk,
                prediction,
                save_dir_for_patches_predictions
                / f"{model_name}_{model_augmentation}"
                / path.name,
            )
            for path, prediction in zip(array_of_patch_complete_paths, predictions)
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()

    # Use patches to reconstruct a single image and save it
    output_filename = (
        save_dir_for_complete_images_preditions
        / f"{model_name}_{model_augmentation}"
        / (metadata.image_name + ".tif")
    )
    reconstructed_image = reconstruct_image_from_patches_and_metadata(
        predictions, metadata, patch_size
    )

    save_mask_in_disk(reconstructed_image, output_filename)


def run_inference(
    patches_dir: Path,
    images_dir: Path,
    models_dir: Path,
    predictions_patch_level: Path,
    predictions_image_level: Path,
) -> None:
    """Run inference on patches using classical and deep learning methods.

    Args:
        patches_dir: Directory with reconstruction patches (nested by image name)
        images_dir: Directory with complete test images
        models_dir: Directory with trained models
        predictions_patch_level: Output directory for patch predictions
        predictions_image_level: Output directory for reconstructed image predictions
    """
    # Create output directories
    overwrite_and_create_directory(predictions_patch_level)
    overwrite_and_create_directory(predictions_image_level)

    # Get subdirectories corresponding to image names (each contains patches for one image)
    image_subdirs = sorted([d for d in patches_dir.glob("*/") if d.is_dir()])

    if not image_subdirs:
        raise ValueError(f"No image subdirectories found in {patches_dir}")

    # Get deep learning models
    models_list = get_deep_learning_models_from_dir(
        models_dir, AVAILABLE_MODELS, AVAILABLE_AUGMENTATIONS
    )

    for subdir in image_subdirs:
        # Get patches for this image, sorted by patch ID
        image_patches_paths = sorted(
            subdir.glob("*"),
            key=lambda filename: extract_patch_info_from_path(filename).patch_id,
        )

        # Get the corresponding complete image
        complete_image_path = images_dir / subdir.name

        # Apply classical thresholding methods
        logger.info(f"Applying classical methods to {subdir.name}...")
        for method in AVAILABLE_CLASSICAL_METHODS:
            # Patch-level predictions
            apply_classical_thresholding_and_save_masks_for_array_of_filenames(
                image_patches_paths,
                predictions_patch_level,
                method,
                MAX_WORKERS,
            )
            # Image-level predictions
            apply_classical_thresholding_and_save_masks_for_array_of_filenames(
                [complete_image_path],
                predictions_image_level,
                method,
                MAX_WORKERS,
            )

        # Apply deep learning methods
        logger.info(f"Applying deep learning methods to {subdir.name}...")
        for model_name, augmentation, best_model_path in models_list:
            apply_deep_learning_method_to_array_of_filenames(
                image_patches_paths,
                predictions_patch_level,
                predictions_image_level,
                model_name,
                augmentation,
                best_model_path,
                BATCH_SIZE,
                PATCH_SIZE,
                MAX_WORKERS,
            )
