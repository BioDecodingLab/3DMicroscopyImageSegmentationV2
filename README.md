# 3D Microscopy Image Segmentation (V2)

Deep learning segmentation of tubular structures in 3D microscopy images.

**Paper**: [An Open-Source Protocol for Deep Learning-Based Segmentation of Tubular Structures](https://app.jove.com/t/68004)

---

## What's New in V2

Complete rewrite of [V1](https://github.com/hernanmorales-navarrete/3DMicroscopyImageSegmentation) with major improvements:

- **Simplified CLI**: 1 argument (dataset path) instead of 5-6 arguments per command
- **Domain-Driven Design**: Clear separation into `patches/`, `training/`, `inference/`, `plotting/`
- **DatasetPaths dataclass**: Single source of truth for all paths
- **Parallel processing**: `ProcessPoolExecutor` for patch generation and inference
- **Self-contained datasets**: All outputs (models, predictions, plots) stored in dataset folder
- **Clean code**: Consistent `interface.py` + `utils.py` pattern in each module
- **Type annotations**: Full type hints throughout the codebase
- **Validation**: Each stage validates requirements before running

---

## Requirements

- Windows with **WSL2** (Ubuntu recommended)
- NVIDIA GPU with CUDA support
- Python 3.10

---

## Installation

Open your WSL terminal and run:

```bash
# 1. Clone the repository
git clone https://github.com/hernanmorales-navarrete/3DMicroscopyImageSegmentationV2.git
cd 3DMicroscopyImageSegmentationV2

# 2. Create conda environment
conda create -n img-seg python=3.10 -y
conda activate img-seg

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Prepare Your Dataset

Create this folder structure:

```
my_dataset/
├── training_data/
│   ├── images/           # Your training images (.tif or .tiff)
│   │   ├── img1.tif
│   │   └── img2.tif
│   └── masks/            # Corresponding masks (.tif or .tiff)
│       ├── img1.tif
│       └── img2.tif
└── test_data/
    ├── images/           # Your test images (.tif or .tiff)
    │   ├── img3.tif
    │   └── img4.tif
    └── masks/            # Corresponding masks (.tif or .tiff)
        ├── img3.tif
        └── img4.tif
```

**Important**:

- Image and mask files must have the **same filename**
- Supported formats: `.tif` and `.tiff`

---

## Usage

All commands require only the **dataset path**. Everything else is automatic.

### Step 1: Generate Patches

Splits your 3D images into 64x64x64 patches for training and inference.

```bash
python -m src.patches.interface /path/to/my_dataset
```

### Step 2: Train Model

Train a segmentation model on your patches.

```bash
python -m src.training.interface /path/to/my_dataset MODEL AUGMENTATION
```

**Arguments:**

- `MODEL`: `UNet3D` or `AttentionUNet3D`
- `AUGMENTATION`: `NONE`, `STANDARD`, or `OURS`

**Options:**

- `--psf-path /path/to/PSF.tif` - Required for `OURS` augmentation
- `--no-reproducibility` - Disable random seed (enabled by default)

**Examples:**

```bash
# Basic training
python -m src.training.interface /path/to/my_dataset UNet3D STANDARD

# With microscopy-specific augmentation
python -m src.training.interface /path/to/my_dataset AttentionUNet3D OURS --psf-path /path/to/PSF.tif
```

### Step 3: Run Inference

Generate predictions using all trained models (classical + deep learning).

```bash
python -m src.inference.interface /path/to/my_dataset
```

### Step 4: Generate Evaluation Plots

Create boxplot comparisons of all methods.

```bash
python -m src.plotting.interface /path/to/my_dataset
```

---

## Output Structure

After running all steps, your dataset will contain:

```
my_dataset/
├── training_data/
│   └── regular_patches/          # Training patches
├── test_data/
│   ├── regular_patches/          # Test patches (for evaluation)
│   └── reconstruction_patches/   # Test patches (for image reconstruction)
├── models/                       # Trained models
│   ├── UNet3D_STANDARD/
│   └── AttentionUNet3D_OURS/
├── predictions/                  # Segmentation results
│   ├── patch_level/              # Patch predictions
│   └── image_level/              # Reconstructed full images
├── reports/
│   └── figures/                  # Evaluation plots
└── logs/                         # TensorBoard training logs
```

---

## Quick Start Example

```bash
# Activate environment
conda activate img-seg

# Run full pipeline
python -m src.patches.interface /mnt/c/Data/my_dataset
python -m src.training.interface /mnt/c/Data/my_dataset UNet3D STANDARD
python -m src.training.interface /mnt/c/Data/my_dataset AttentionUNet3D STANDARD
python -m src.inference.interface /mnt/c/Data/my_dataset
python -m src.plotting.interface /mnt/c/Data/my_dataset
```

---

## License

GPL-3.0
