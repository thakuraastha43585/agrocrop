"""Centralized configuration for paths, image preprocessing, and training parameters."""

from pathlib import Path

# Project root and key directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DATASET_SOURCE_DIR = RAW_DATA_DIR / "PlantVillage"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Saved artifact paths
SPLITS_MANIFEST_PATH = PROCESSED_DATA_DIR / "splits_manifest.csv"
MODEL_PATH = MODELS_DIR / "crop_disease_model.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
HISTORY_PATH = REPORTS_DIR / "training_history.json"
MODEL_CONFIG_PATH = REPORTS_DIR / "model_config.json"
TRAINING_LOG_CSV_PATH = REPORTS_DIR / "training_log.csv"
EVALUATION_RESULTS_PATH = REPORTS_DIR / "evaluation_results.json"
CLASS_DISTRIBUTION_FIG = FIGURES_DIR / "class_distribution.png"
SAMPLE_IMAGES_FIG = FIGURES_DIR / "sample_images.png"
AUGMENTED_IMAGES_FIG = FIGURES_DIR / "augmented_samples.png"
CONFUSION_MATRIX_FIG = FIGURES_DIR / "confusion_matrix.png"
TRAINING_ACCURACY_FIG = FIGURES_DIR / "training_accuracy.png"
TRAINING_LOSS_FIG = FIGURES_DIR / "training_loss.png"

# Target tomato leaf classes (canonical folder names under DATASET_SOURCE_DIR)
SELECTED_CLASSES = sorted(
    [
        "Tomato_healthy",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Bacterial_spot",
        "Tomato_Leaf_Mold",
    ]
)
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Image and dataset settings (MobileNetV2 standard input size)
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
INPUT_SHAPE = (*IMAGE_SIZE, IMAGE_CHANNELS)
BATCH_SIZE = 32
RANDOM_SEED = 42

# Dataset split ratios (train + val + test = 1.0)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training hyperparameters
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 10
INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
DROPOUT_RATE = 0.3
