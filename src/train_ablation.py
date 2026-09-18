"""Ablation Experiment: Training and Evaluation for CNN + BiLSTM Model WITHOUT Attention.

Reuses the exact same:
- Dataset split (from train_manifest.csv, val_manifest.csv, test_manifest.csv)
- Preprocessing and data loading pipeline
- Training conditions (Optimizer: Adam, lr=1e-4, batch_size=16, class weights, early stopping, LR reduction)
- Held-out test set evaluation methodology

Artifacts generated:
- models/cnn_bilstm.keras
- results/metrics/cnn_bilstm_metrics.json
- results/metrics/cnn_bilstm_history.csv
- results/metrics/cnn_bilstm_classification_report.txt
- results/confusion_matrix/cnn_bilstm_confusion_matrix.png
- results/plots/cnn_bilstm_accuracy.png
- results/plots/cnn_bilstm_loss.png
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import time
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
    CLASS_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
    EPOCHS,
    INITIAL_LEARNING_RATE,
    LEARNING_RATE,
    RANDOM_SEED,
    PATIENCE_EARLY_STOP,
    PATIENCE_REDUCE_LR,
    FACTOR_REDUCE_LR,
    MIN_LR,
    CNN_BILSTM_MODEL_PATH,
    CNN_BILSTM_METRICS_JSON,
    CNN_BILSTM_HISTORY_CSV,
    CNN_BILSTM_ACCURACY_PLOT,
    CNN_BILSTM_LOSS_PLOT,
    CNN_BILSTM_REPORT_TXT,
    CNN_BILSTM_CM_PNG,
    ensure_directories,
)
from src.preprocessing import load_and_preprocess_image
from src.dataset import get_data_loaders, get_balanced_class_weights
from src.model import build_cnn_bilstm_ablation, get_compiled_cnn_bilstm_ablation


def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class EpochProgressCallback(tf.keras.callbacks.Callback):
    """Callback printing per-epoch train/val loss and accuracy."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        tr_loss = logs.get("loss", float("nan"))
        tr_acc = logs.get("accuracy", float("nan")) * 100.0
        val_loss = logs.get("val_loss", float("nan"))
        val_acc = logs.get("val_accuracy", float("nan")) * 100.0
        lr = float(self.model.optimizer.learning_rate)
        print(
            f"Epoch {epoch + 1:2d}/{self.params['epochs']} | "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
            f"LR: {lr:.6f}"
        )


def evaluate_ablation_model(model, test_df):
    """Evaluate CNN + BiLSTM model on the identical held-out test set."""
    print("\n" + "=" * 70)
    print("EVALUATING CNN + BiLSTM (WITHOUT ATTENTION) ON HELD-OUT TEST SET (165 images)")
    print("=" * 70)

    X_test = []
    y_true = test_df["label"].values.astype(int)

    for path in test_df["file_path"]:
        img = load_and_preprocess_image(path, target_size=IMAGE_SIZE)
        X_test.append(img)
    X_test = np.array(X_test, dtype=np.float32)

    # Inference
    print("Running inference on test set...")
    y_probs = model.predict(X_test, batch_size=16, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # Compute metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Multiclass ROC-AUC
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(CLASS_NAMES))
    try:
        roc_auc_macro = float(roc_auc_score(y_true_onehot, y_probs, multi_class="ovr", average="macro"))
        roc_auc_weighted = float(roc_auc_score(y_true_onehot, y_probs, multi_class="ovr", average="weighted"))
    except Exception as e:
        print(f"[WARNING] Multiclass ROC-AUC note: {e}")
        roc_auc_macro = None
        roc_auc_weighted = None

    cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

    # Save metrics JSON
    metrics_data = {
        "model": "CNN_BiLSTM_No_Attention",
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
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(prec_per_class[i]),
                "recall": float(rec_per_class[i]),
                "f1_score": float(f1_per_class[i]),
            }
            for i in range(len(CLASS_NAMES))
        },
        "confusion_matrix": cm.tolist(),
    }

    with open(CNN_BILSTM_METRICS_JSON, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[OUTPUT] Metrics saved to: {CNN_BILSTM_METRICS_JSON}")

    # Save Classification Report
    with open(CNN_BILSTM_REPORT_TXT, "w") as f:
        f.write("CNN + BiLSTM (WITHOUT ATTENTION) CLASSIFICATION REPORT\n")
        f.write("=" * 65 + "\n")
        f.write(f"Total Test Samples: {len(test_df)}\n\n")
        f.write(clf_report)
    print(f"[OUTPUT] Classification report saved to: {CNN_BILSTM_REPORT_TXT}")

    # Save Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("CNN + BiLSTM (No Attention) - Confusion Matrix", fontsize=12, fontweight="bold")
    plt.ylabel("Ground Truth Label", fontsize=10)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(CNN_BILSTM_CM_PNG, dpi=300)
    plt.close()
    print(f"[OUTPUT] Confusion matrix saved to: {CNN_BILSTM_CM_PNG}")

    # Print summary to console
    print("\n" + "=" * 70)
    print("CNN + BiLSTM (WITHOUT ATTENTION) MEASURED TEST RESULTS:")
    print("=" * 70)
    print(f"Accuracy:           {acc * 100:.2f}%")
    print(f"Precision (Macro):  {prec_macro:.4f}")
    print(f"Recall (Macro):     {rec_macro:.4f}")
    print(f"F1-Score (Macro):   {f1_macro:.4f}")
    if roc_auc_macro is not None:
        print(f"ROC-AUC (Macro):    {roc_auc_macro:.4f}")
    print("\nClassification Report:\n" + clf_report)
    print("=" * 70)

    return metrics_data


def save_training_plots(history):
    """Plot and save ablation training accuracy and loss curves."""
    # Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy", color="#2ca02c", linewidth=2)
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    plt.title("CNN + BiLSTM (No Attention) - Training Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(CNN_BILSTM_ACCURACY_PLOT, dpi=300)
    plt.close()
    print(f"[OUTPUT] Accuracy plot saved to: {CNN_BILSTM_ACCURACY_PLOT}")

    # Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss", color="#2ca02c", linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2)
    plt.title("CNN + BiLSTM (No Attention) - Training Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(CNN_BILSTM_LOSS_PLOT, dpi=300)
    plt.close()
    print(f"[OUTPUT] Loss plot saved to: {CNN_BILSTM_LOSS_PLOT}")


def train_ablation(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=INITIAL_LEARNING_RATE):
    """Train CNN + BiLSTM ablation model on identical splits and training setup."""
    ensure_directories()
    set_seed(RANDOM_SEED)

    print("=" * 70)
    print("ABLATION EXPERIMENT: CNN + BiLSTM (WITHOUT ATTENTION)")
    print("=" * 70)

    # 1. Load identical data splits
    train_ds, val_ds, test_ds, train_df, val_df, test_df = get_data_loaders(
        batch_size=batch_size,
        use_sparse=True,
    )
    class_weights = get_balanced_class_weights(train_df)

    print(f"Train samples : {len(train_df)}")
    print(f"Val samples   : {len(val_df)}")
    print(f"Test samples  : {len(test_df)}")

    # 2. Build and compile ablation model
    print("\nBuilding CNN + BiLSTM Ablation Model (No Attention)...")
    model = get_compiled_cnn_bilstm_ablation(learning_rate=lr)
    model.summary()

    # 3. Training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE_EARLY_STOP,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CNN_BILSTM_MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=FACTOR_REDUCE_LR,
            patience=PATIENCE_REDUCE_LR,
            min_lr=MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(CNN_BILSTM_HISTORY_CSV)),
        EpochProgressCallback(),
    ]

    print("\n" + "=" * 70)
    print(f"STARTING CNN + BiLSTM TRAINING ({epochs} Epochs, Batch Size: {batch_size}, LR: {lr})")
    print("=" * 70)

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=0,
    )
    train_duration = time.time() - start_time
    print(f"\nTraining completed in {train_duration / 60:.2f} minutes ({len(history.history['loss'])} epochs executed).")

    # Save plots
    save_training_plots(history)

    # Load best checkpoint
    print(f"\nLoading best checkpoint from {CNN_BILSTM_MODEL_PATH} for test evaluation...")
    best_model = tf.keras.models.load_model(str(CNN_BILSTM_MODEL_PATH))

    # Evaluate on held-out test set
    metrics = evaluate_ablation_model(best_model, test_df)

    print(f"\n[DONE] Ablation model trained and saved to: {CNN_BILSTM_MODEL_PATH}")
    return best_model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN+BiLSTM ablation model")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=INITIAL_LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    train_ablation(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
