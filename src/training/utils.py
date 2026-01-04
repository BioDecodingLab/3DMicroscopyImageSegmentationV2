from datetime import datetime
import inspect
from pathlib import Path
import types

import keras
from loguru import logger
from sklearn.model_selection import train_test_split
import tensorflow

from src.config import (
    ALLOWED_EXTENSIONS,
    CHECKPOINT_MODE,
    CHECKPOINT_MONITOR,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
    SAVE_BEST_ONLY,
    TENSORBOARD_UPDATE_FREQ,
    VALIDATION_SPLIT,
)


def configure_gpu():
    """Configure GPU memory growth to avoid taking all memory.

    This should be called at the beginning of scripts that use GPU.
    """
    gpus = tensorflow.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tensorflow.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tensorflow.config.list_logical_devices("GPU")
            logger.info(f"Using {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            logger.error(e)
    else:
        logger.warning("No GPUs found. Running on CPU only.")


def set_random_seed(seed: int):
    keras.utils.set_random_seed(seed)
    tensorflow.config.experimental.enable_op_determinism()


def get_available_models(module: types.ModuleType):
    "Get models from a module. A model should be a class and have the method attribute build_model"
    model_classes = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and hasattr(obj, "build_model"):
            model_classes[name] = obj
    return model_classes


def get_model_from_class(model_name: str, module: types.ModuleType):
    "Get a model class from a module"
    model_classes = get_available_models(module)
    if model_name not in model_classes:
        available_models = list(model_classes.keys())
        raise ValueError(f"Model {model_name} not found. Available models: {available_models}")
    return model_classes[model_name]


def get_patch_paths_for_training(
    patches_dir: Path,
    validation_split: float = VALIDATION_SPLIT,
    random_state: int = RANDOM_SEED,
) -> tuple[list[Path], list[Path], list[Path], list[Path]]:
    """Get patch file paths from nested subdirectory structure.

    Structure: patches_dir/images/img1.tif/*.tif
               patches_dir/masks/img1.tif/*.tif

    Args:
        patches_dir: Path to patches directory (e.g., training_regular_patches)
        validation_split: Fraction of data to use for validation
        random_state: Random seed for reproducible splits

    Returns:
        Tuple of (train_image_paths, val_image_paths, train_mask_paths, val_mask_paths)
    """
    images_dir = patches_dir / "images"
    masks_dir = patches_dir / "masks"

    # Collect paths for all allowed extensions (files only, not directories)
    image_paths = []
    mask_paths = []
    for ext in ALLOWED_EXTENSIONS:
        image_paths.extend(f for f in images_dir.rglob(f"*{ext}") if f.is_file())
        mask_paths.extend(f for f in masks_dir.rglob(f"*{ext}") if f.is_file())

    image_paths = sorted(image_paths)
    mask_paths = sorted(mask_paths)

    if not image_paths:
        raise ValueError(f"No files with extensions {ALLOWED_EXTENSIONS} found in {images_dir}")
    if not mask_paths:
        raise ValueError(f"No files with extensions {ALLOWED_EXTENSIONS} found in {masks_dir}")

    if len(image_paths) != len(mask_paths):
        raise ValueError(
            f"Number of images ({len(image_paths)}) does not match "
            f"number of masks ({len(mask_paths)})"
        )

    # Split the data
    train_image_paths, val_image_paths, train_mask_paths, val_mask_paths = train_test_split(
        image_paths, mask_paths, test_size=validation_split, random_state=random_state
    )

    logger.info(
        f"Found {len(train_image_paths)} training samples and "
        f"{len(val_image_paths)} validation samples"
    )

    return train_image_paths, val_image_paths, train_mask_paths, val_mask_paths


def create_callbacks(
    model_name: str,
    augmentation: str,
    logs_dir: Path,
    models_dir: Path,
) -> list:
    """Create training callbacks.

    Saves to: {models_dir}/{model_name}_{augmentation}/{timestamp}/best_model.h5

    Args:
        model_name: Name of the model (e.g., UNet3D, AttentionUNet3D)
        augmentation: Type of augmentation (e.g., NONE, STANDARD, OURS)
        logs_dir: Directory for TensorBoard logs
        models_dir: Directory for model checkpoints

    Returns:
        List of Keras callbacks
    """
    callbacks = []

    # Create directories for logs and checkpoints
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir_name = f"{model_name}_{augmentation}"

    log_dir = logs_dir / model_dir_name / timestamp
    checkpoint_dir = models_dir / model_dir_name / timestamp

    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard callback
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(
        log_dir=str(log_dir),
        update_freq=TENSORBOARD_UPDATE_FREQ,
    )
    callbacks.append(tensorboard_callback)

    # Early stopping callback
    early_stopping = tensorflow.keras.callbacks.EarlyStopping(
        monitor=CHECKPOINT_MONITOR,
        min_delta=EARLY_STOPPING_MIN_DELTA,
        patience=EARLY_STOPPING_PATIENCE,
        mode=CHECKPOINT_MODE,
        restore_best_weights=True,
    )
    callbacks.append(early_stopping)

    # Model checkpoint callback
    checkpoint_path = checkpoint_dir / "best_model.h5"

    checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(
        str(checkpoint_path),
        monitor=CHECKPOINT_MONITOR,
        save_best_only=SAVE_BEST_ONLY,
        mode=CHECKPOINT_MODE,
    )
    callbacks.append(checkpoint)

    logger.info(f"Model will be saved to: {checkpoint_path}")

    return callbacks
