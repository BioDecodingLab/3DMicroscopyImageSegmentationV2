# File format configuration
ALLOWED_EXTENSIONS = (".tiff", ".tif")
OUTPUT_EXTENSION = ".tif"

# Patch configuration
PATCH_SIZE = (64, 64, 64)
PATCH_STEP = 64

MAX_WORKERS = 16  # Number of cores used in multiprocessing tasks
THRESHOLD = 0.5  # Thresholding for

# Intensity augmentation parameters
INTENSITY_PARAMS = {
    "background_level": 0.1,  # Background intensity level
    "local_variation_scale": 5,  # Scale of local variations
    "z_decay_rate": 0.999,  # Rate of intensity decay along z-axis
    "noise_std": 0.1,  # Standard deviation for Gaussian noise
    "poisson_scale": 1.0,  # Scaling factor for Poisson noise
    "intensity_scale": 1000.0,  # Scale factor for image intensity before augmentation,
    "snr_tolerance": 0.1,  # Tolerance for SNR
    "max_iterations": 200,  # Maximum number of iterations for intensity augmentation
    "std_dev": 10,  # Standard deviation for Gaussian noise
    "snr_targets": [
        15,
        10,
        5,
        4,
        3,
        2,
    ],  # Target SNR values for augmentation
}

# Training Configuration
RANDOM_SEED = 42
AVAILABLE_MODELS = ["UNet3D", "AttentionUNet3D"]
AVAILABLE_AUGMENTATIONS = ["NONE", "STANDARD", "OURS"]
AVAILABLE_CLASSICAL_METHODS = ["otsu", "adaptive_gaussian", "adaptive_mean", "frangi"]
AVAILABLE_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "f1",
    "iou",
    "dice",
    "volume_similarity",
]

# Training hyperparameters
LEARNING_RATE = 1e-5
BATCH_SIZE = 8
NUM_EPOCHS = 100
VALIDATION_SPLIT = 0.2

# Early stopping parameters
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0  # Use Keras default

# Model checkpoint parameters
CHECKPOINT_MONITOR = "val_loss"
CHECKPOINT_MODE = "min"
SAVE_BEST_ONLY = True
CHECKPOINT_SAVE_FREQ = "epoch"
CHECKPOINT_SAVE_WEIGHTS_ONLY = False
CHECKPOINT_SAVE_FORMAT = "h5"  # or 'tf' for SavedModel format

# Loss and metrics
LOSS_FUNCTION = "binary_crossentropy"
METRICS = ["accuracy"]

# Tensorboard
TENSORBOARD_UPDATE_FREQ = "epoch"
PROFILE_BATCH = 0  # Disable profiling
