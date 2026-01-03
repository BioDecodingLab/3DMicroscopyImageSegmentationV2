"""Tests for DatasetPaths dataclass."""

import pytest
from pathlib import Path

from src.dataset_paths import DatasetPaths


class TestDatasetPathsInitialization:
    """Tests for DatasetPaths initialization."""

    def test_init_with_valid_path(self, tmp_path: Path):
        """DatasetPaths initializes with a valid directory."""
        paths = DatasetPaths(tmp_path)
        assert paths.root == tmp_path

    def test_init_with_string_path(self, tmp_path: Path):
        """DatasetPaths converts string to Path."""
        paths = DatasetPaths(str(tmp_path))
        assert isinstance(paths.root, Path)
        assert paths.root == tmp_path

    def test_init_with_nonexistent_path_raises(self):
        """DatasetPaths raises FileNotFoundError for nonexistent directory."""
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            DatasetPaths(Path("/nonexistent/path"))


class TestBaseDirectories:
    """Tests for base directory properties."""

    def test_training_data(self, tmp_path: Path):
        """training_data returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert paths.training_data == tmp_path / "training_data"

    def test_test_data(self, tmp_path: Path):
        """test_data returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert paths.test_data == tmp_path / "test_data"


class TestTrainingDataPaths:
    """Tests for training data path properties."""

    def test_training_images(self, tmp_path: Path):
        """training_images returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert paths.training_images == tmp_path / "training_data" / "images"

    def test_training_masks(self, tmp_path: Path):
        """training_masks returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert paths.training_masks == tmp_path / "training_data" / "masks"

    def test_training_regular_patches(self, tmp_path: Path):
        """training_regular_patches returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert paths.training_regular_patches == tmp_path / "training_data" / "regular_patches"

    def test_training_reconstruction_patches(self, tmp_path: Path):
        """training_reconstruction_patches returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert (
            paths.training_reconstruction_patches
            == tmp_path / "training_data" / "reconstruction_patches"
        )


class TestTestDataPaths:
    """Tests for test data path properties."""

    def test_test_images(self, tmp_path: Path):
        """test_images returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert paths.test_images == tmp_path / "test_data" / "images"

    def test_test_masks(self, tmp_path: Path):
        """test_masks returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert paths.test_masks == tmp_path / "test_data" / "masks"

    def test_test_regular_patches(self, tmp_path: Path):
        """test_regular_patches returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert paths.test_regular_patches == tmp_path / "test_data" / "regular_patches"

    def test_test_reconstruction_patches(self, tmp_path: Path):
        """test_reconstruction_patches returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert (
            paths.test_reconstruction_patches
            == tmp_path / "test_data" / "reconstruction_patches"
        )


class TestOutputDirectories:
    """Tests for output directory properties."""

    def test_models(self, tmp_path: Path):
        """models returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert paths.models == tmp_path / "models"

    def test_predictions(self, tmp_path: Path):
        """predictions returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert paths.predictions == tmp_path / "predictions"

    def test_predictions_patch_level(self, tmp_path: Path):
        """predictions_patch_level returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert paths.predictions_patch_level == tmp_path / "predictions" / "patch_level"

    def test_predictions_image_level(self, tmp_path: Path):
        """predictions_image_level returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert paths.predictions_image_level == tmp_path / "predictions" / "image_level"

    def test_reports(self, tmp_path: Path):
        """reports returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert paths.reports == tmp_path / "reports"

    def test_figures(self, tmp_path: Path):
        """figures returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert paths.figures == tmp_path / "reports" / "figures"

    def test_logs(self, tmp_path: Path):
        """logs returns correct path."""
        paths = DatasetPaths(tmp_path)
        assert paths.logs == tmp_path / "logs"


class TestPatchStructureHelpers:
    """Tests for nested patch structure helper methods."""

    def test_get_patch_image_subdirs_returns_sorted_subdirs(self, tmp_path: Path):
        """get_patch_image_subdirs returns sorted list of subdirectories."""
        paths = DatasetPaths(tmp_path)

        # Create patches directory structure
        patches_dir = tmp_path / "regular_patches"
        images_dir = patches_dir / "images"
        (images_dir / "img_c.tif").mkdir(parents=True)
        (images_dir / "img_a.tif").mkdir(parents=True)
        (images_dir / "img_b.tif").mkdir(parents=True)

        subdirs = paths.get_patch_image_subdirs(patches_dir)

        assert len(subdirs) == 3
        assert subdirs[0].name == "img_a.tif"
        assert subdirs[1].name == "img_b.tif"
        assert subdirs[2].name == "img_c.tif"

    def test_get_patch_image_subdirs_ignores_files(self, tmp_path: Path):
        """get_patch_image_subdirs only returns directories, not files."""
        paths = DatasetPaths(tmp_path)

        patches_dir = tmp_path / "regular_patches"
        images_dir = patches_dir / "images"
        images_dir.mkdir(parents=True)
        (images_dir / "img_a.tif").mkdir()
        (images_dir / "some_file.txt").touch()

        subdirs = paths.get_patch_image_subdirs(patches_dir)

        assert len(subdirs) == 1
        assert subdirs[0].name == "img_a.tif"

    def test_get_patch_image_subdirs_empty_when_no_subdirs(self, tmp_path: Path):
        """get_patch_image_subdirs returns empty list when no subdirectories exist."""
        paths = DatasetPaths(tmp_path)

        patches_dir = tmp_path / "regular_patches"
        images_dir = patches_dir / "images"
        images_dir.mkdir(parents=True)

        subdirs = paths.get_patch_image_subdirs(patches_dir)

        assert subdirs == []

    def test_get_patch_mask_subdir(self, tmp_path: Path):
        """get_patch_mask_subdir returns correct mask subdirectory path."""
        paths = DatasetPaths(tmp_path)
        patches_dir = tmp_path / "regular_patches"

        mask_subdir = paths.get_patch_mask_subdir(patches_dir, "img_a.tif")

        assert mask_subdir == patches_dir / "masks" / "img_a.tif"


class TestValidateForPatches:
    """Tests for validate_for_patches method."""

    @staticmethod
    def _setup_valid_data_folder(images_dir: Path, masks_dir: Path) -> None:
        """Helper to create valid images and masks directories with matching files."""
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        (images_dir / "img1.tif").touch()
        (masks_dir / "img1.tif").touch()

    def test_validate_for_patches_succeeds_with_valid_data(self, tmp_path: Path):
        """validate_for_patches succeeds when both training and test data are valid."""
        paths = DatasetPaths(tmp_path)

        # Setup valid training data
        self._setup_valid_data_folder(paths.training_images, paths.training_masks)

        # Setup valid test data
        self._setup_valid_data_folder(paths.test_images, paths.test_masks)

        # Should not raise
        paths.validate_for_patches()

    def test_validate_for_patches_raises_when_training_images_dir_missing(self, tmp_path: Path):
        """validate_for_patches raises when training images directory is missing."""
        paths = DatasetPaths(tmp_path)
        paths.training_masks.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Images directory missing"):
            paths.validate_for_patches()

    def test_validate_for_patches_raises_when_training_masks_dir_missing(self, tmp_path: Path):
        """validate_for_patches raises when training masks directory is missing."""
        paths = DatasetPaths(tmp_path)
        paths.training_images.mkdir(parents=True)
        (paths.training_images / "img1.tif").touch()

        with pytest.raises(FileNotFoundError, match="Masks directory missing"):
            paths.validate_for_patches()

    def test_validate_for_patches_raises_when_test_images_dir_missing(self, tmp_path: Path):
        """validate_for_patches raises when test images directory is missing."""
        paths = DatasetPaths(tmp_path)

        # Valid training data
        self._setup_valid_data_folder(paths.training_images, paths.training_masks)

        # Missing test images dir
        paths.test_masks.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Images directory missing"):
            paths.validate_for_patches()

    def test_validate_for_patches_raises_when_test_masks_dir_missing(self, tmp_path: Path):
        """validate_for_patches raises when test masks directory is missing."""
        paths = DatasetPaths(tmp_path)

        # Valid training data
        self._setup_valid_data_folder(paths.training_images, paths.training_masks)

        # Missing test masks dir
        paths.test_images.mkdir(parents=True)
        (paths.test_images / "img1.tif").touch()

        with pytest.raises(FileNotFoundError, match="Masks directory missing"):
            paths.validate_for_patches()

    def test_validate_for_patches_raises_when_no_valid_images(self, tmp_path: Path):
        """validate_for_patches raises when no images with valid extensions exist."""
        paths = DatasetPaths(tmp_path)
        paths.training_images.mkdir(parents=True)
        paths.training_masks.mkdir(parents=True)

        # Create file with invalid extension
        (paths.training_images / "img1.png").touch()

        with pytest.raises(FileNotFoundError, match="No images with valid extensions"):
            paths.validate_for_patches()

    def test_validate_for_patches_raises_when_empty_images_dir(self, tmp_path: Path):
        """validate_for_patches raises when images directory is empty."""
        paths = DatasetPaths(tmp_path)
        paths.training_images.mkdir(parents=True)
        paths.training_masks.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="No images with valid extensions"):
            paths.validate_for_patches()

    def test_validate_for_patches_raises_when_mask_missing(self, tmp_path: Path):
        """validate_for_patches raises when an image has no corresponding mask."""
        paths = DatasetPaths(tmp_path)
        paths.training_images.mkdir(parents=True)
        paths.training_masks.mkdir(parents=True)

        # Create image without corresponding mask
        (paths.training_images / "img1.tif").touch()
        (paths.training_images / "img2.tif").touch()
        (paths.training_masks / "img1.tif").touch()  # Only mask for img1

        with pytest.raises(FileNotFoundError, match="Missing masks for images"):
            paths.validate_for_patches()

    def test_validate_for_patches_accepts_tiff_extension(self, tmp_path: Path):
        """validate_for_patches accepts .tiff extension."""
        paths = DatasetPaths(tmp_path)

        # Training with .tiff
        paths.training_images.mkdir(parents=True)
        paths.training_masks.mkdir(parents=True)
        (paths.training_images / "img1.tiff").touch()
        (paths.training_masks / "img1.tiff").touch()

        # Test with .tiff
        paths.test_images.mkdir(parents=True)
        paths.test_masks.mkdir(parents=True)
        (paths.test_images / "img1.tiff").touch()
        (paths.test_masks / "img1.tiff").touch()

        # Should not raise
        paths.validate_for_patches()

    def test_validate_for_patches_case_insensitive_extension(self, tmp_path: Path):
        """validate_for_patches handles case-insensitive extensions."""
        paths = DatasetPaths(tmp_path)

        # Training with uppercase
        paths.training_images.mkdir(parents=True)
        paths.training_masks.mkdir(parents=True)
        (paths.training_images / "img1.TIF").touch()
        (paths.training_masks / "img1.TIF").touch()

        # Test with uppercase
        paths.test_images.mkdir(parents=True)
        paths.test_masks.mkdir(parents=True)
        (paths.test_images / "img1.TIF").touch()
        (paths.test_masks / "img1.TIF").touch()

        # Should not raise
        paths.validate_for_patches()


class TestValidateForTraining:
    """Tests for validate_for_training method."""

    def test_validate_for_training_succeeds_with_patches(self, tmp_path: Path):
        """validate_for_training succeeds when patches exist."""
        paths = DatasetPaths(tmp_path)

        # Create patches directory with at least one subdirectory
        patches_images = paths.training_regular_patches / "images"
        (patches_images / "img1.tif").mkdir(parents=True)

        # Should not raise
        paths.validate_for_training()

    def test_validate_for_training_raises_when_patches_dir_missing(self, tmp_path: Path):
        """validate_for_training raises when patches directory doesn't exist."""
        paths = DatasetPaths(tmp_path)

        with pytest.raises(FileNotFoundError, match="No patches found"):
            paths.validate_for_training()

    def test_validate_for_training_raises_when_patches_dir_empty(self, tmp_path: Path):
        """validate_for_training raises when patches directory is empty."""
        paths = DatasetPaths(tmp_path)
        (paths.training_regular_patches / "images").mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="No patches found"):
            paths.validate_for_training()


class TestValidateForInference:
    """Tests for validate_for_inference method."""

    def test_validate_for_inference_succeeds_with_all_requirements(self, tmp_path: Path):
        """validate_for_inference succeeds when all required directories and models exist."""
        paths = DatasetPaths(tmp_path)

        # Create required directories
        (paths.test_reconstruction_patches / "images").mkdir(parents=True)
        paths.test_images.mkdir(parents=True)
        paths.models.mkdir(parents=True)

        # Create a model file (models/<model_name>/<timestamp>/best_model.h5)
        model_dir = paths.models / "UNet3D_NONE" / "20260102-120000"
        model_dir.mkdir(parents=True)
        (model_dir / "best_model.h5").touch()

        # Should not raise
        paths.validate_for_inference()

    def test_validate_for_inference_raises_when_reconstruction_patches_missing(
        self, tmp_path: Path
    ):
        """validate_for_inference raises when reconstruction patches are missing."""
        paths = DatasetPaths(tmp_path)
        paths.test_images.mkdir(parents=True)
        paths.models.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Required directory missing"):
            paths.validate_for_inference()

    def test_validate_for_inference_raises_when_test_images_missing(self, tmp_path: Path):
        """validate_for_inference raises when test images are missing."""
        paths = DatasetPaths(tmp_path)
        (paths.test_reconstruction_patches / "images").mkdir(parents=True)
        paths.models.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Required directory missing"):
            paths.validate_for_inference()

    def test_validate_for_inference_raises_when_models_dir_missing(self, tmp_path: Path):
        """validate_for_inference raises when models directory is missing."""
        paths = DatasetPaths(tmp_path)
        (paths.test_reconstruction_patches / "images").mkdir(parents=True)
        paths.test_images.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Required directory missing"):
            paths.validate_for_inference()

    def test_validate_for_inference_raises_when_no_models_found(self, tmp_path: Path):
        """validate_for_inference raises when models directory has no .h5 files."""
        paths = DatasetPaths(tmp_path)
        (paths.test_reconstruction_patches / "images").mkdir(parents=True)
        paths.test_images.mkdir(parents=True)
        paths.models.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="No trained models found"):
            paths.validate_for_inference()


class TestValidateForPlotting:
    """Tests for validate_for_plotting method."""

    def test_validate_for_plotting_succeeds_with_all_requirements(self, tmp_path: Path):
        """validate_for_plotting succeeds when all required directories exist."""
        paths = DatasetPaths(tmp_path)

        paths.predictions_patch_level.mkdir(parents=True)
        paths.predictions_image_level.mkdir(parents=True)
        paths.test_masks.mkdir(parents=True)

        # Should not raise
        paths.validate_for_plotting()

    def test_validate_for_plotting_raises_when_patch_level_missing(self, tmp_path: Path):
        """validate_for_plotting raises when patch_level predictions are missing."""
        paths = DatasetPaths(tmp_path)
        paths.predictions_image_level.mkdir(parents=True)
        paths.test_masks.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Required directory missing"):
            paths.validate_for_plotting()

    def test_validate_for_plotting_raises_when_image_level_missing(self, tmp_path: Path):
        """validate_for_plotting raises when image_level predictions are missing."""
        paths = DatasetPaths(tmp_path)
        paths.predictions_patch_level.mkdir(parents=True)
        paths.test_masks.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Required directory missing"):
            paths.validate_for_plotting()

    def test_validate_for_plotting_raises_when_test_masks_missing(self, tmp_path: Path):
        """validate_for_plotting raises when test masks are missing."""
        paths = DatasetPaths(tmp_path)
        paths.predictions_patch_level.mkdir(parents=True)
        paths.predictions_image_level.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Required directory missing"):
            paths.validate_for_plotting()
