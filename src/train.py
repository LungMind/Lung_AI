import sys
from pathlib import Path

# Add project root to sys.path so both 'python src/train.py' and 'python -m src.train' work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.config import (
    DATA_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    PATIENCE_EARLY_STOP,
    PATIENCE_REDUCE_LR,
    FACTOR_REDUCE_LR,
    MIN_LR,
    CNN_BASELINE_MODEL_PATH,
    CNN_BASELINE_HISTORY_CSV,
    CNN_BASELINE_ACCURACY_PLOT,
    CNN_BASELINE_LOSS_PLOT,
    CNN_BASELINE_REPORT_TXT,
    CNN_BASELINE_METRICS_JSON,
    CNN_BASELINE_CM_PNG,
    ensure_directories,
)
from src.preprocessing import load_and_preprocess_image
from src.dataset import get_data_loaders, get_balanced_class_weights
from src.model import build_cnn_baseline, get_compiled_cnn_baseline


def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def detect_device():
    """Detect and log compute device (GPU vs CPU)."""
    gpus = tf.config.list_physical_devices("GPU")
    print("\n" + "=" * 60)
    print("HARDWARE ACCELERATION DETECTION")
    print("=" * 60)
    if gpus:
        print(f"[COMPUTE] GPU detected: {len(gpus)} device(s) found.")
        for gpu in gpus:
            print(f"          - {gpu.name}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"          [NOTE] Memory growth setting: {e}")
        gpu_detected = True
    else:
        print("[COMPUTE] No NVIDIA GPU detected. Training will run on CPU.")
        gpu_detected = False
    print("=" * 60)
    return gpu_detected


def run_sanity_check(learning_rate=LEARNING_RATE):
    """Perform quick validation that the model builds, computes loss, and propagates cleanly."""
    print("\n" + "=" * 60)
    print("RUNNING CNN BASELINE SANITY CHECK")
    print("=" * 60)

    # 1. Build Model
    print("[1/5] Building CNN Baseline architecture...")
    model = get_compiled_cnn_baseline(learning_rate=learning_rate)
    print("      Model built successfully.")

    # 2. Inspect Model Summary
    print("\n[2/5] Model Architecture Summary:")
    model.summary()

    # 3. Test Forward Propagation with Dummy Batch
    print("\n[3/5] Testing Forward Propagation...")
    dummy_input = np.random.uniform(0.0, 1.0, size=(4, IMAGE_SIZE[0], IMAGE_SIZE[1], 3)).astype(np.float32)
    dummy_pred = model(dummy_input, training=False)
    output_shape = tuple(dummy_pred.shape)
    print(f"      Input shape:  {dummy_input.shape}")
    print(f"      Output shape: {output_shape}")
    assert output_shape == (4, 3), f"Expected shape (4, 3), but got {output_shape}"
    print("      [PASS] Output shape verified as (None, 3).")

    # 4. Verify Probabilities Sum to 1
    prob_sums = np.sum(dummy_pred.numpy(), axis=-1)
    print(f"      Softmax probability row sums: {prob_sums.round(4)}")
    assert np.allclose(prob_sums, 1.0, atol=1e-5), "Softmax probabilities do not sum to 1.0!"
    print("      [PASS] Softmax calibration verified.")

    # 5. Verify Loss Calculation on a Real Data Batch
    print("\n[4/5] Testing Loss Calculation on Sample Batch from Dataset...")
    _, _, test_ds, _, _, _ = get_data_loaders(batch_size=4)
    sample_images, sample_labels = next(iter(test_ds))

    sample_pred = model(sample_images, training=False)
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss_val = float(cce(sample_labels, sample_pred))
    print(f"      Batch loss calculated: {loss_val:.4f}")
    assert not np.isnan(loss_val) and not np.isinf(loss_val), "Loss is NaN or Inf!"
    print("      [PASS] Loss calculation verified.")

    # 5. One Gradient Step Test
    print("\n[5/5] Testing Single Optimization Gradient Step...")
    with tf.GradientTape() as tape:
        preds = model(sample_images, training=True)
        loss = cce(sample_labels, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
    print("      [PASS] Optimizer backpropagation and gradient update verified.")

    print("\n" + "=" * 60)
    print("[SANITY CHECK COMPLETE] All checks passed successfully.")
    print("=" * 60)
    return True


def save_plots(history):
    """Generate and save separate accuracy and loss plots."""
    # 1. Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy", color="#1f77b4", linewidth=2)
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    plt.title("CNN Baseline - Accuracy Across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(CNN_BASELINE_ACCURACY_PLOT, dpi=300)
    plt.close()
    print(f"[OUTPUT] Accuracy plot saved to: {CNN_BASELINE_ACCURACY_PLOT}")

    # 2. Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss", color="#1f77b4", linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2)
    plt.title("CNN Baseline - Cross-Entropy Loss Across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(CNN_BASELINE_LOSS_PLOT, dpi=300)
    plt.close()
    print(f"[OUTPUT] Loss plot saved to: {CNN_BASELINE_LOSS_PLOT}")


def evaluate_test_set(model, test_df):
    """Evaluate model strictly on the held-out test set and save all evaluation artifacts."""
    print("\n" + "=" * 60)
    print("EVALUATION ON HELD-OUT TEST SET (165 images)")
    print("=" * 60)

    X_test = []
    y_true = test_df["label"].values

    for path in test_df["file_path"]:
        img = load_and_preprocess_image(path, target_size=IMAGE_SIZE)
        X_test.append(img)
    X_test = np.array(X_test, dtype=np.float32)

    # Predict probabilities
    y_probs = model.predict(X_test, batch_size=16, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # Calculate actual observed metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Multiclass ROC-AUC (One-vs-Rest)
    try:
        y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(CLASS_NAMES))
        roc_auc_macro = float(roc_auc_score(y_true_onehot, y_probs, multi_class="ovr", average="macro"))
        roc_auc_weighted = float(roc_auc_score(y_true_onehot, y_probs, multi_class="ovr", average="weighted"))
    except Exception as e:
        print(f"[WARNING] Multiclass ROC-AUC calculation notice: {e}")
        roc_auc_macro = None
        roc_auc_weighted = None

    cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

    # Save Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("CNN Baseline - Confusion Matrix", fontsize=12, fontweight="bold")
    plt.ylabel("Ground Truth Label", fontsize=10)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(CNN_BASELINE_CM_PNG, dpi=300)
    plt.close()
    print(f"[OUTPUT] Confusion matrix heatmap saved to: {CNN_BASELINE_CM_PNG}")

    # Save Classification Report Text
    with open(CNN_BASELINE_REPORT_TXT, "w") as f:
        f.write("CNN BASELINE CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Test Samples: {len(test_df)}\n\n")
        f.write(clf_report)
    print(f"[OUTPUT] Classification report saved to: {CNN_BASELINE_REPORT_TXT}")

    # Save Metrics JSON
    metrics_data = {
        "model": "CNN_Baseline",
        "total_test_samples": len(test_df),
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_score_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_score_weighted": f1_weighted,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "confusion_matrix": cm.tolist(),
    }
    with open(CNN_BASELINE_METRICS_JSON, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[OUTPUT] Evaluation metrics saved to: {CNN_BASELINE_METRICS_JSON}")

    # Console Summary
    print("\n" + "=" * 60)
    print("TEST EVALUATION METRICS (ACTUAL OBSERVED):")
    print("=" * 60)
    print(f"Accuracy:           {acc * 100:.2f}%")
    print(f"Precision (Macro):  {prec_macro:.4f}")
    print(f"Recall (Macro):     {rec_macro:.4f}")
    print(f"F1-Score (Macro):   {f1_macro:.4f}")
    if roc_auc_macro is not None:
        print(f"ROC-AUC (Macro):    {roc_auc_macro:.4f}")
    print("\nClassification Report:\n" + clf_report)
    print("=" * 60)
    return metrics_data


def train(epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE):
    """Execute full training pipeline for CNN baseline."""
    ensure_directories()
    set_seed(RANDOM_SEED)
    gpu_detected = detect_device()

    print("\n" + "=" * 60)
    print("DATASET & SPLIT CONFIGURATION")
    print("=" * 60)
    train_ds, val_ds, test_ds, train_df, val_df, test_df = get_data_loaders(batch_size=batch_size)

    print(f"Training samples:   {len(train_df)} images")
    print(f"Validation samples: {len(val_df)} images")
    print(f"Test samples:       {len(test_df)} images")
    print("\nClass Distribution in Training Set:")
    for c in CLASS_NAMES:
        count = (train_df["class_name"] == c).sum()
        pct = count / len(train_df) * 100
        print(f" - {c:<10s}: {count:4d} images ({pct:5.1f}%)")

    # Compute balanced class weights to handle imbalance
    class_weights = get_balanced_class_weights(train_df)
    print("\nComputed Balanced Class Weights (sklearn.utils.class_weight):")
    for cls_idx, weight in class_weights.items():
        print(f" - Class {cls_idx} ({CLASS_NAMES[cls_idx]}): {weight:.4f}")

    # Build model
    print("\n" + "=" * 60)
    print("BUILDING CNN BASELINE MODEL")
    print("=" * 60)
    model = get_compiled_cnn_baseline(learning_rate=learning_rate)
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE_EARLY_STOP,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CNN_BASELINE_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=FACTOR_REDUCE_LR,
            patience=PATIENCE_REDUCE_LR,
            min_lr=MIN_LR,
            verbose=1,
        ),
    ]

    print("\n" + "=" * 60)
    print(f"STARTING TRAINING ({epochs} Epochs, Batch Size: {batch_size}, LR: {learning_rate})")
    print("=" * 60)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Save training history as CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(CNN_BASELINE_HISTORY_CSV, index_label="epoch")
    print(f"[OUTPUT] Training history saved to: {CNN_BASELINE_HISTORY_CSV}")

    # Save Accuracy & Loss plots
    save_plots(history)

    # Load best checkpoint for test evaluation
    print(f"\n[EVAL] Loading best checkpoint from {CNN_BASELINE_MODEL_PATH} for final test evaluation...")
    best_model = tf.keras.models.load_model(str(CNN_BASELINE_MODEL_PATH))

    # Evaluate on held-out test set
    evaluate_test_set(best_model, test_df)

    print("\n" + "=" * 60)
    print("CNN BASELINE TRAINING & EVALUATION COMPLETE")
    print(f"Model saved: {CNN_BASELINE_MODEL_PATH}")
    print("=" * 60)
    return best_model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test CNN Baseline Model for Lung Cancer CT")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run model build, forward propagation, and batch loss sanity checks without full training",
    )

    args = parser.parse_args()

    if args.sanity_check:
        run_sanity_check(learning_rate=args.lr)
    else:
        train(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)
