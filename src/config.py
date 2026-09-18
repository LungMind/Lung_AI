"""Configuration settings for Lung Cancer Classification Framework."""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_data_dir():
    """Dynamically resolve the dataset directory."""
    env_dir = os.getenv("LUNG_DATA_DIR")
    if env_dir and Path(env_dir).exists():
        return Path(env_dir).resolve()

    iq_nested = PROJECT_ROOT / "The IQ-OTHNCCD lung cancer dataset" / "The IQ-OTHNCCD lung cancer dataset"
    if iq_nested.exists():
        return iq_nested.resolve()

    iq_root = PROJECT_ROOT / "The IQ-OTHNCCD lung cancer dataset"
    if iq_root.exists():
        return iq_root.resolve()

    return (PROJECT_ROOT / "data" / "lung_cancer").resolve()


DATA_DIR = resolve_data_dir()

# Target Classes
CLASS_NAMES = ["Normal", "Benign", "Malignant"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# Image Specifications
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_CHANNELS = 3
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)

# Data Splitting Ratios
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Training Hyperparameters
BATCH_SIZE = 16
EPOCHS = 30
INITIAL_LEARNING_RATE = 1e-4
LEARNING_RATE = INITIAL_LEARNING_RATE
PATIENCE_EARLY_STOP = 5
PATIENCE_REDUCE_LR = 2
FACTOR_REDUCE_LR = 0.5
MIN_LR = 1e-6

# Architecture Parameters
LSTM_UNITS = 64
DENSE_UNITS = 64
DROPOUT_RATE = 0.3

# Output Directories
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
CONFUSION_MATRIX_DIR = RESULTS_DIR / "confusion_matrix"
METRICS_DIR = RESULTS_DIR / "metrics"
GRADCAM_DIR = RESULTS_DIR / "gradcam"

# Split manifest CSV paths
TRAIN_MANIFEST_CSV = METRICS_DIR / "train_manifest.csv"
VAL_MANIFEST_CSV = METRICS_DIR / "val_manifest.csv"
TEST_MANIFEST_CSV = METRICS_DIR / "test_manifest.csv"
DATASET_SUMMARY_JSON = METRICS_DIR / "dataset_inspection_summary.json"

# CNN Baseline Specific Artifacts
CNN_BASELINE_MODEL_PATH = MODELS_DIR / "cnn_baseline.keras"
CNN_BASELINE_HISTORY_CSV = METRICS_DIR / "cnn_baseline_history.csv"
CNN_BASELINE_ACCURACY_PLOT = PLOTS_DIR / "cnn_baseline_accuracy.png"
CNN_BASELINE_LOSS_PLOT = PLOTS_DIR / "cnn_baseline_loss.png"
CNN_BASELINE_REPORT_TXT = METRICS_DIR / "cnn_baseline_classification_report.txt"
CNN_BASELINE_METRICS_JSON = METRICS_DIR / "cnn_baseline_metrics.json"
CNN_BASELINE_CM_PNG = CONFUSION_MATRIX_DIR / "cnn_baseline_confusion_matrix.png"

# Hybrid CNN-BiLSTM-Attention Specific Artifacts
HYBRID_MODEL_PATH = MODELS_DIR / "hybrid_cnn_bilstm_attention.keras"
BEST_MODEL_PATH = HYBRID_MODEL_PATH  # Default alias for downstream apps
HYBRID_HISTORY_CSV = METRICS_DIR / "hybrid_history.csv"
HYBRID_ACCURACY_PLOT = PLOTS_DIR / "hybrid_accuracy.png"
HYBRID_LOSS_PLOT = PLOTS_DIR / "hybrid_loss.png"
HYBRID_REPORT_TXT = METRICS_DIR / "hybrid_classification_report.txt"
HYBRID_METRICS_JSON = METRICS_DIR / "hybrid_metrics.json"
HYBRID_CM_PNG = CONFUSION_MATRIX_DIR / "hybrid_confusion_matrix.png"
HYBRID_TEST_PREDICTIONS_CSV = METRICS_DIR / "hybrid_test_predictions.csv"
MISCLASSIFIED_SAMPLES_CSV = METRICS_DIR / "misclassified_samples.csv"

# CNN + BiLSTM (Without Attention) Ablation Artifacts
CNN_BILSTM_MODEL_PATH = MODELS_DIR / "cnn_bilstm.keras"
CNN_BILSTM_METRICS_JSON = METRICS_DIR / "cnn_bilstm_metrics.json"
CNN_BILSTM_HISTORY_CSV = METRICS_DIR / "cnn_bilstm_history.csv"
CNN_BILSTM_ACCURACY_PLOT = PLOTS_DIR / "cnn_bilstm_accuracy.png"
CNN_BILSTM_LOSS_PLOT = PLOTS_DIR / "cnn_bilstm_loss.png"
CNN_BILSTM_REPORT_TXT = METRICS_DIR / "cnn_bilstm_classification_report.txt"
CNN_BILSTM_CM_PNG = CONFUSION_MATRIX_DIR / "cnn_bilstm_confusion_matrix.png"

# Model Comparison and Ablation Artifacts
MODEL_COMPARISON_CSV = METRICS_DIR / "model_comparison.csv"
ABLATION_COMPARISON_CSV = METRICS_DIR / "ablation_comparison.csv"
ABLATION_COMPARISON_PLOT = PLOTS_DIR / "ablation_comparison.png"



def ensure_directories():
    """Ensure all required output directories exist."""
    directories = [
        MODELS_DIR,
        RESULTS_DIR,
        PLOTS_DIR,
        CONFUSION_MATRIX_DIR,
        METRICS_DIR,
        GRADCAM_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_directories()
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Configured Dataset Directory: {DATA_DIR}")
