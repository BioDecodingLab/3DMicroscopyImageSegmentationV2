"""DatasetPaths dataclass for managing all paths derived from a single dataset root."""

from dataclasses import dataclass
from pathlib import Path

from src.config import ALLOWED_EXTENSIONS


@dataclass
class DatasetPaths:
    """All paths derived from a single dataset root.

    This class provides a centralized way to access all paths within a dataset,
    ensuring consistent structure across all pipeline stages (patches, training,
    inference, plotting).

    Dataset Structure:
        dataset/
        ├── training_data/
        │   ├── images/                        # User provides (e.g., img1.tif)
        │   ├── masks/                         # User provides (e.g., img1.tif)
        │   └── regular_patches/               # Generated (for training, no padding)
        │       ├── images/
        │       │   └── img1.tif/              # Subdirectory named after source image
        │       │       └── img1_orig_Z_Y_X_npatches_..._patch_0000.tif
        │       └── masks/
        │           └── img1.tif/
        │               └── img1_orig_Z_Y_X_npatches_..._patch_0000.tif
        ├── test_data/
        │   ├── images/                        # User provides
        │   ├── masks/                         # User provides
        │   ├── regular_patches/               # Generated (for patch-level evaluation)
        │   │   ├── images/
        │   │   │   └── img1.tif/
        │   │   └── masks/
        │   │       └── img1.tif/
        │   └── reconstruction_patches/        # Generated (with padding for reconstruction)
        │       ├── images/
        │       │   └── img1.tif/
        │       └── masks/
        │           └── img1.tif/
        ├── models/                            # Trained models
        │   └── UNet3D_OURS/                   # <model_name>_<augmentation>
        │       └── 20260102-143000/           # <timestamp>
        │           └── best_model.h5
        ├── predictions/                       # Inference outputs
        │   ├── patch_level/
        │   │   └── otsu/                      # <method>: otsu, frangi, UNet3D_OURS, etc.
        │   │       └── img1.tif/              # Subdirectory named after source image
        │   │           └── img1_orig_..._patch_0000.tif
        │   └── image_level/
        │       └── otsu/
        │           └── img1.tif               # Reconstructed full image
        ├── reports/                           # Evaluation outputs
        │   └── figures/
        └── logs/                              # TensorBoard logs
    """

    root: Path

    def __post_init__(self):
        self.root = Path(self.root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset not found: {self.root}")

    # === Base directories ===
    @property
    def training_data(self) -> Path:
        return self.root / "training_data"

    @property
    def test_data(self) -> Path:
        return self.root / "test_data"

    # === Training data paths ===
    @property
    def training_images(self) -> Path:
        return self.training_data / "images"

    @property
    def training_masks(self) -> Path:
        return self.training_data / "masks"

    @property
    def training_regular_patches(self) -> Path:
        return self.training_data / "regular_patches"

    @property
    def training_reconstruction_patches(self) -> Path:
        return self.training_data / "reconstruction_patches"

    # === Test data paths ===
    @property
    def test_images(self) -> Path:
        return self.test_data / "images"

    @property
    def test_masks(self) -> Path:
        return self.test_data / "masks"

    @property
    def test_regular_patches(self) -> Path:
        return self.test_data / "regular_patches"

    @property
    def test_reconstruction_patches(self) -> Path:
        return self.test_data / "reconstruction_patches"

    # === Output directories ===
    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def predictions_patch_level(self) -> Path:
        return self.predictions / "patch_level"

    @property
    def predictions_image_level(self) -> Path:
        return self.predictions / "image_level"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    # === Nested patch structure helpers ===
    def get_patch_image_subdirs(self, patches_dir: Path) -> list[Path]:
        """Get subdirectories in patches/images/, one per source image."""
        images_dir = patches_dir / "images"
        return sorted([d for d in images_dir.glob("*/") if d.is_dir()])

    def get_patch_mask_subdir(self, patches_dir: Path, image_name: str) -> Path:
        """Get corresponding mask subdirectory for a given image."""
        return patches_dir / "masks" / image_name

    # === Validation ===
    def _validate_data_folder_for_patches(self, data_folder: Path) -> None:
        """Validate a data folder (training_data or test_data) for patch generation.

        Enforces invariants from patches/dataset.py:
        1. Both images/ and masks/ directories must exist
        2. At least one image with valid extension exists
        3. Each image must have a corresponding mask with the same filename
        """
        images_dir = data_folder / "images"
        masks_dir = data_folder / "masks"

        # Check directories exist
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Images directory missing: {images_dir}")
        if not masks_dir.is_dir():
            raise FileNotFoundError(f"Masks directory missing: {masks_dir}")

        # Get images with valid extensions
        images = [f for f in images_dir.iterdir() if f.suffix.lower() in ALLOWED_EXTENSIONS]

        if not images:
            raise FileNotFoundError(
                f"No images with valid extensions {ALLOWED_EXTENSIONS} found in: {images_dir}"
            )

        # Check each image has a corresponding mask
        missing_masks = []
        for image in images:
            mask_path = masks_dir / image.name
            if not mask_path.exists():
                missing_masks.append(image.name)

        if missing_masks:
            raise FileNotFoundError(
                f"Missing masks for images: {missing_masks}. "
                f"Each image must have a corresponding mask with the same filename."
            )

    def validate_for_patches(self) -> None:
        """Check required directories and files exist for patch generation.

        Validates both training_data and test_data folders.
        """
        self._validate_data_folder_for_patches(self.training_data)
        self._validate_data_folder_for_patches(self.test_data)

    def validate_for_training(self) -> None:
        """Check required directories exist for training."""
        patches_images = self.training_regular_patches / "images"
        patches_masks = self.training_regular_patches / "masks"

        if not patches_images.exists() or not any(patches_images.iterdir()):
            raise FileNotFoundError(
                f"No image patches found. Run patch generation first: {patches_images}"
            )

        if not patches_masks.exists() or not any(patches_masks.iterdir()):
            raise FileNotFoundError(
                f"No mask patches found. Run patch generation first: {patches_masks}"
            )

    def validate_for_inference(self) -> None:
        """Check required directories exist for inference."""
        required = [
            self.test_reconstruction_patches / "images",
            self.test_images,
            self.models,
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Required directory missing: {path}")

        # Check models directory has at least one model
        # Structure: models/<model_name>_<augmentation>/<timestamp>/best_model.h5
        if not any(self.models.glob("*/*/*.h5")):
            raise FileNotFoundError(f"No trained models found in: {self.models}")

    def validate_for_plotting(self) -> None:
        """Check required directories exist for evaluation/plotting."""
        required = [
            self.predictions_patch_level,
            self.predictions_image_level,
            self.test_masks,
        ]
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"Required directory missing: {path}")
