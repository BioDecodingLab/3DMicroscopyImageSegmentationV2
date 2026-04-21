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
from skimage.filters import frangi, threshold_otsu
import tensorflow
import tifffile

from src.config import (
    AVAILABLE_AUGMENTATIONS,
    AVAILABLE_CLASSICAL_METHODS,
    AVAILABLE_MODELS,
    BATCH_SIZE,
    MAX_WORKERS,
    PATCH_SIZE,
    THRESHOLD,
)
from src.utils import create_directory, overwrite_and_create_directory

# Image-level predictions are whole volumes; keep their pool small to avoid OOM
# when multiple processes hold a full reconstructed volume in memory at once.
IMAGE_LEVEL_MAX_WORKERS = 1


@dataclass
class ImageMetadata:
    "Information necessary to reconstruct an image from patches"

    image_name: str
    original_shape: Tuple[int, int, int]
    padded_shape: Tuple[int, int, int] | None
    number_of_patches: Tuple[int, int, int]
    patch_id: int


def extract_patch_info_from_path(path: Path) -> ImageMetadata:
    """Extract information from patch path.

    Args:
        path: Patch file path (e.g. '.../image1_orig_512_512_128_pad_520_520_130_npatches_4_8_8_patch_0000.tif')
    """
    pattern = r"(.+)_orig_(\d+)_(\d+)_(\d+)(?:_pad_(\d+)_(\d+)_(\d+))?_npatches_(\d+)_(\d+)_(\d+)_patch_(\d+)\.tiff?"
    match = re.match(pattern, path.name)

    if not match:
        raise ValueError(f"Invalid patch filename format: {path}")

    image_name = match.group(1)
    orig_shape = (int(match.group(2)), int(match.group(3)), int(match.group(4)))

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


def apply_threshold_to_image_and_convert_to_dtype(image: NDArray, threshold: float, dtype):
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
    else:
        raise ValueError(f"2D thresholding method {method} not implemented")

    return apply_threshold_to_image_and_convert_to_dtype(mask, 0, np.uint8)


def apply_frangi_to_3D_image(image: NDArray) -> NDArray:
    """Apply 3D Frangi vesselness + Otsu threshold to a 3D volume.

    skimage's frangi handles 3D natively, so we preserve z-axis context instead
    of collapsing to per-slice 2D. For a uniform Frangi response,
    ``threshold_otsu`` returns that single value, which makes
    ``normalized > otsu`` False everywhere — the correct "no vessels" result.
    """
    frangi_result = frangi(image.astype(np.float32))
    normalized = normalize_image_to_0_1(frangi_result)
    otsu = threshold_otsu(normalized)
    return (normalized > otsu).astype(np.uint8)


def apply_classical_thresholding_method_to_3D_image(image: NDArray, method: str):
    if method == "frangi":
        return apply_frangi_to_3D_image(image)

    # otsu, adaptive_mean, adaptive_gaussian are inherently 2D in OpenCV,
    # so we apply them per z-slice and stack.
    masks_from_slices = [
        apply_classical_thresholding_method_to_2D_image(image[z], method)
        for z in range(image.shape[0])
    ]
    return np.stack(masks_from_slices, axis=0)


def save_mask_in_disk(mask: NDArray, output_dir: Path):
    try:
        tifffile.imwrite(output_dir, mask)
    except OSError as e:
        raise OSError(f"The mask {output_dir} couldn't be saved: {e}") from e


def apply_method_and_save_mask(image_path: Path, method: str, save_dir: Path):
    image = tifffile.imread(str(image_path))
    mask = apply_classical_thresholding_method_to_3D_image(image, method)
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
            future.result()


def extract_information_from_model_dir_path(model_path: Path):
    # Directory format: "<model_name>_<augmentation>". Split on the LAST underscore so
    # a multi-word augmentation like "SEMI_SUPERVISED" still parses correctly.
    name = model_path.name
    if "_" not in name:
        raise ValueError(
            f"Invalid model directory name format. Expected '<model_name>_<augmentation>', "
            f"got {name}"
        )
    model_name, augmentation = name.rsplit("_", 1)
    return model_name, augmentation


def get_deep_learning_models_from_dir(
    models_dir: Path, available_models: List[str], available_augmentations: List[str]
):
    """Discover trained model directories and return (name, augmentation, path) tuples.

    Expected layout:
        models_dir/
            <model_name>_<augmentation>/          # e.g. "UNet3D_OURS"
                <timestamp>/                      # "%Y%m%d-%H%M%S"
                    best_model.h5

    When a (model_name, augmentation) pair has multiple timestamp directories,
    only the most recent one is kept; older training runs are ignored.
    """

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory doesn't exist: {models_dir}")

    models_list = []

    for model_dir in sorted(models_dir.glob("*/")):
        model_name, model_augmentation = extract_information_from_model_dir_path(model_dir)

        if model_name not in available_models:
            raise ValueError(
                f"Unknown model name '{model_name}' in {model_dir.name}. "
                f"Available: {available_models}"
            )
        if model_augmentation not in available_augmentations:
            raise ValueError(
                f"Unknown augmentation '{model_augmentation}' in {model_dir.name}. "
                f"Available: {available_augmentations}"
            )

        timestamps = list(model_dir.glob("*/"))
        if not timestamps:
            raise FileNotFoundError(f"No timestamp directories in {model_dir}")

        # Timestamps are "%Y%m%d-%H%M%S", so lexicographic sort == chronological sort.
        latest_timestamp_model = sorted(timestamps, key=lambda d: d.name)[-1]

        model_files = list(latest_timestamp_model.glob("*.h5"))
        if not model_files:
            raise FileNotFoundError(f"No .h5 model files in {latest_timestamp_model}")

        best_model_path = model_files[0]
        logger.info(f"Found {model_name}_{model_augmentation} at {best_model_path}")

        models_list.append((model_name, model_augmentation, best_model_path))

    return models_list


def apply_deep_learning_model_to_batch(batch, model, threshold: float) -> NDArray:
    batch_normalized = np.stack([normalize_image_to_0_1(patch) for patch in batch])
    batch_input_to_model = batch_normalized[..., np.newaxis]
    batch_preds = model.predict(batch_input_to_model, verbose=0)[..., 0]
    return apply_threshold_to_image_and_convert_to_dtype(batch_preds, threshold, np.uint8)


def batch_iterable(iterable, n):
    """Batch an iterable into lists of size n.

    Python 3.10 doesn't ship itertools.batched(), so we implement it here.

    >>> list(batch_iterable(['a.tiff', 'b.tiff', 'c.tiff', 'd.tiff'], 2))
    [['a.tiff', 'b.tiff'], ['c.tiff', 'd.tiff']]
    """
    if n < 1:
        raise ValueError("Batch size should be at least 1")
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, n)):
        yield batch


def reconstruct_image_from_patches_and_metadata(
    patches: list[NDArray], metadata: ImageMetadata, patch_size: tuple[int, int, int]
):
    if metadata.padded_shape is None:
        raise ValueError(
            "Image can't be reconstructed from regular patches, use reconstruction patches instead"
        )

    patches_reshaped = np.array(patches).reshape(*metadata.number_of_patches, *patch_size)
    reconstructed_image = unpatchify(patches_reshaped, metadata.padded_shape)
    reconstructed_image = reconstructed_image[tuple(slice(0, s) for s in metadata.original_shape)]
    return reconstructed_image


def predict_patches_with_model(
    patch_paths: list[Path],
    model,
    batch_size: int,
    threshold: float,
) -> list[NDArray]:
    """Run a pre-loaded model on the given patches and return thresholded predictions."""
    predictions = []
    for batch_of_filenames in batch_iterable(patch_paths, batch_size):
        batch_of_patches = [tifffile.imread(str(p)) for p in batch_of_filenames]
        prediction_of_batch = apply_deep_learning_model_to_batch(
            batch_of_patches, model, threshold
        )
        predictions.extend(prediction_of_batch)
    return predictions


def save_predictions_in_parallel(
    predictions: list[NDArray],
    patch_paths: list[Path],
    save_dir: Path,
    max_workers: int,
) -> None:
    """Save each prediction under save_dir using the corresponding source patch filename."""
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(save_mask_in_disk, prediction, save_dir / path.name)
            for path, prediction in zip(patch_paths, predictions)
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def _sorted_patches_in_subdir(subdir: Path) -> list[Path]:
    return sorted(
        subdir.glob("*"),
        key=lambda filename: extract_patch_info_from_path(filename).patch_id,
    )


def run_inference(
    regular_patches_dir: Path,
    reconstruction_patches_dir: Path,
    images_dir: Path,
    models_dir: Path,
    predictions_patch_level: Path,
    predictions_image_level: Path,
) -> None:
    """Run inference using classical methods and every discovered deep-learning model.

    Patch-level predictions are produced from the no-padding regular patches so that
    metrics reflect real ground truth (not reflection-padded ground truth). Image-level
    predictions come from (a) classical methods applied to the complete image and
    (b) deep-learning models applied to reconstruction patches, which are then
    reassembled into a full volume.

    Args:
        regular_patches_dir: test_regular_patches/images/ (nested by source image)
        reconstruction_patches_dir: test_reconstruction_patches/images/ (nested by source image)
        images_dir: Directory with complete test images (.tif/.tiff)
        models_dir: Directory with trained models
        predictions_patch_level: Output directory for patch-level predictions
        predictions_image_level: Output directory for image-level predictions
    """
    overwrite_and_create_directory(predictions_patch_level)
    overwrite_and_create_directory(predictions_image_level)

    regular_subdirs = sorted([d for d in regular_patches_dir.glob("*/") if d.is_dir()])
    if not regular_subdirs:
        raise ValueError(f"No image subdirectories found in {regular_patches_dir}")

    reconstruction_subdirs = sorted(
        [d for d in reconstruction_patches_dir.glob("*/") if d.is_dir()]
    )
    if not reconstruction_subdirs:
        raise ValueError(f"No image subdirectories found in {reconstruction_patches_dir}")

    # Load each deep-learning model once and reuse across all test images. Previously
    # the model was reloaded inside the per-image loop, paying the full Keras
    # deserialization cost N_images * N_models times.
    models_info = get_deep_learning_models_from_dir(
        models_dir, AVAILABLE_MODELS, AVAILABLE_AUGMENTATIONS
    )
    loaded_models = [
        (name, augmentation, tensorflow.keras.models.load_model(path))
        for name, augmentation, path in models_info
    ]

    # === Patch-level classical predictions (regular patches, no padding) ===
    for subdir in regular_subdirs:
        image_patches_paths = _sorted_patches_in_subdir(subdir)
        logger.info(f"Applying classical methods to regular patches of {subdir.name}...")
        for method in AVAILABLE_CLASSICAL_METHODS:
            apply_classical_thresholding_and_save_masks_for_array_of_filenames(
                image_patches_paths,
                predictions_patch_level,
                method,
                MAX_WORKERS,
            )

    # === Image-level classical predictions (complete images) ===
    complete_image_paths = [images_dir / subdir.name for subdir in reconstruction_subdirs]
    logger.info("Applying classical methods to complete images...")
    for method in AVAILABLE_CLASSICAL_METHODS:
        apply_classical_thresholding_and_save_masks_for_array_of_filenames(
            complete_image_paths,
            predictions_image_level,
            method,
            IMAGE_LEVEL_MAX_WORKERS,
        )

    # === Deep-learning predictions ===
    for model_name, augmentation, model in loaded_models:
        model_tag = f"{model_name}_{augmentation}"

        # Patch-level: predict on regular patches
        patch_out_dir = predictions_patch_level / model_tag
        create_directory(patch_out_dir)
        for subdir in regular_subdirs:
            image_patches_paths = _sorted_patches_in_subdir(subdir)
            logger.info(f"Applying {model_tag} to regular patches of {subdir.name}...")
            predictions = predict_patches_with_model(
                image_patches_paths, model, BATCH_SIZE, THRESHOLD
            )
            save_predictions_in_parallel(
                predictions, image_patches_paths, patch_out_dir, MAX_WORKERS
            )

        # Image-level: predict on reconstruction patches, then reassemble
        image_out_dir = predictions_image_level / model_tag
        create_directory(image_out_dir)
        for subdir in reconstruction_subdirs:
            image_patches_paths = _sorted_patches_in_subdir(subdir)
            logger.info(f"Reconstructing {model_tag} prediction for {subdir.name}...")
            predictions = predict_patches_with_model(
                image_patches_paths, model, BATCH_SIZE, THRESHOLD
            )
            metadata = extract_patch_info_from_path(image_patches_paths[0])
            reconstructed = reconstruct_image_from_patches_and_metadata(
                predictions, metadata, PATCH_SIZE
            )
            # Preserve the original image filename (including extension) so plotting
            # can match predictions to ground truth masks without extension drift.
            save_mask_in_disk(reconstructed, image_out_dir / subdir.name)
