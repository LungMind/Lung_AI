"""Training and evaluation script for the Hybrid CNN-BiLSTM-Attention Model."""

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
    HYBRID_MODEL_PATH,
    HYBRID_HISTORY_CSV,
    HYBRID_ACCURACY_PLOT,
    HYBRID_LOSS_PLOT,
    HYBRID_REPORT_TXT,
    HYBRID_METRICS_JSON,
    HYBRID_CM_PNG,
    HYBRID_TEST_PREDICTIONS_CSV,
    MISCLASSIFIED_SAMPLES_CSV,
    ensure_directories,
)
from src.preprocessing import load_and_preprocess_image
from src.dataset import get_data_loaders, get_balanced_class_weights
from src.attention import SequenceAttention
from src.model import build_hybrid_model, get_compiled_hybrid_model
from src.compare_models import compare_models


def set_seed(seed=RANDOM_SEED):
    """Set random seeds across Python, NumPy, and TensorFlow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def detect_device():
    """Detect and log TensorFlow GPU or CPU fallback."""
    print("=" * 65)
    print("HARDWARE ACCELERATION & ENVIRONMENT")
    print("=" * 65)
    print(f"TensorFlow Version : {tf.__version__}")

    gpus = tf.config.list_physical_devices("GPU")
    gpu_available = len(gpus) > 0
    print(f"GPU Available      : {gpu_available}")

    if gpu_available:
        for idx, gpu in enumerate(gpus):
            print(f"GPU [{idx}] Name       : {gpu.name}")
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"Memory growth note : {e}")
        device_label = f"GPU ({len(gpus)} device(s))"
    else:
        print("CPU Fallback       : GPU unavailable; using system CPU.")
        device_label = "CPU"
    print("=" * 65)
    return device_label


class EpochProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback printing required per-epoch metrics."""

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        tr_loss = logs.get("loss", 0.0)
        tr_acc = logs.get("accuracy", 0.0) * 100
        vl_loss = logs.get("val_loss", 0.0)
        vl_acc = logs.get("val_accuracy", 0.0) * 100

        # Extract current learning rate
        lr = float(self.model.optimizer.learning_rate)
        if hasattr(self.model.optimizer.learning_rate, "numpy"):
            lr = float(self.model.optimizer.learning_rate.numpy())

        print(
            f"Epoch {epoch + 1:2d}/{self.params['epochs']} | "
            f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:5.2f}% | "
            f"Val Loss: {vl_loss:.4f} | Val Acc: {vl_acc:5.2f}% | "
            f"LR: {lr:.6f}"
        )


def run_smoke_test(learning_rate=INITIAL_LEARNING_RATE):
    """Quick training smoke test to verify loading, forward pass, gradients, saving, and validation."""
    print("\n" + "=" * 65)
    print("RUNNING PRE-TRAINING SMOKE TEST")
    print("=" * 65)

    ensure_directories()
    set_seed(RANDOM_SEED)

    # 1. Test data loaders (sparse integer labels)
    print("[1/5] Loading sample batches from train and val splits...")
    train_ds, val_ds, _, train_df, val_df, _ = get_data_loaders(batch_size=4, use_sparse=True)
    images_tr, labels_tr = next(iter(train_ds))
    images_vl, labels_vl = next(iter(val_ds))
    print(f"      Train batch: images shape {images_tr.shape}, labels shape {labels_tr.shape}")
    print(f"      Val batch:   images shape {images_vl.shape}, labels shape {labels_vl.shape}")
    assert images_tr.shape == (4, 224, 224, 3), "Invalid train batch shape!"
    assert labels_tr.shape == (4,), "Labels should be 1D integer array for sparse crossentropy!"
    print("      [PASS] Data loading and sparse label alignment verified.")

    # 2. Build model with sparse categorical crossentropy
    print("\n[2/5] Initializing Hybrid CNN-BiLSTM-Attention Model...")
    model = get_compiled_hybrid_model(learning_rate=learning_rate, loss="sparse_categorical_crossentropy")
    print("      [PASS] Model initialized.")

    # 3. Test forward pass and loss finiteness
    print("\n[3/5] Testing forward pass and initial loss...")
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    initial_preds = model(images_tr, training=False)
    initial_loss = float(scce(labels_tr, initial_preds))
    print(f"      Initial batch loss: {initial_loss:.4f}")
    assert not np.isnan(initial_loss) and not np.isinf(initial_loss), "Initial loss is NaN or Inf!"
    print("      [PASS] Forward propagation and finite loss verified.")

    # 4. Test gradient update
    print("\n[4/5] Testing backpropagation and gradient update...")
    with tf.GradientTape() as tape:
        preds = model(images_tr, training=True)
        loss = scce(labels_tr, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

    post_update_preds = model(images_tr, training=False)
    post_update_loss = float(scce(labels_tr, post_update_preds))
    print(f"      Post-update batch loss: {post_update_loss:.4f}")
    assert not np.isnan(post_update_loss) and not np.isinf(post_update_loss), "Post-update loss is NaN or Inf!"
    print("      [PASS] Gradient backpropagation verified.")

    # 5. Test validation step and serialization
    print("\n[5/5] Testing validation step and model serialization...")
    val_preds = model(images_vl, training=False)
    val_loss = float(scce(labels_vl, val_preds))
    print(f"      Sample validation loss: {val_loss:.4f}")

    smoke_model_path = PROJECT_ROOT / "models" / "temp_smoke_hybrid.keras"
    model.save(str(smoke_model_path))
    loaded_smoke = tf.keras.models.load_model(
        str(smoke_model_path),
        custom_objects={"SequenceAttention": SequenceAttention},
    )
    smoke_preds = loaded_smoke(images_vl, training=False)
    assert np.allclose(val_preds.numpy(), smoke_preds.numpy(), atol=1e-5), "Saved and loaded predictions differ!"
    if smoke_model_path.exists():
        smoke_model_path.unlink()
    print("      [PASS] Validation step and model serialization verified.")

    print("\n" + "=" * 65)
    print("[SMOKE TEST SUCCESSFUL] Proceeding to full training.")
    print("=" * 65)
    return True


def save_plots(history):
    """Generate and save separate accuracy and loss plots."""
    # Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy", color="#1f77b4", linewidth=2)
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", linewidth=2)
    plt.title("Hybrid Model - Accuracy Across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Accuracy", fontsize=10)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(HYBRID_ACCURACY_PLOT, dpi=300)
    plt.close()
    print(f"[OUTPUT] Accuracy plot saved to: {HYBRID_ACCURACY_PLOT}")

    # Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss", color="#1f77b4", linewidth=2)
    plt.plot(history.history["val_loss"], label="Validation Loss", color="#ff7f0e", linewidth=2)
    plt.title("Hybrid Model - Cross-Entropy Loss Across Epochs", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(HYBRID_LOSS_PLOT, dpi=300)
    plt.close()
    print(f"[OUTPUT] Loss plot saved to: {HYBRID_LOSS_PLOT}")


def evaluate_hybrid_on_test(model, test_df):
    """Evaluate best saved hybrid model strictly on the held-out test set (165 images)."""
    print("\n" + "=" * 65)
    print("EVALUATION ON HELD-OUT TEST SET (165 images)")
    print("=" * 65)

    X_test = []
    y_true = test_df["label"].values.astype(int)

    for path in test_df["file_path"]:
        img = load_and_preprocess_image(path, target_size=IMAGE_SIZE)
        X_test.append(img)
    X_test = np.array(X_test, dtype=np.float32)

    # Predict class probabilities
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
        print(f"[WARNING] Multiclass ROC-AUC notice: {e}")
        roc_auc_macro = None
        roc_auc_weighted = None

    cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

    # Save Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Hybrid Model - Confusion Matrix (Test Set)", fontsize=12, fontweight="bold")
    plt.ylabel("Ground Truth", fontsize=10)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(HYBRID_CM_PNG, dpi=300)
    plt.close()
    print(f"[OUTPUT] Confusion matrix saved to: {HYBRID_CM_PNG}")

    # Save Classification Report Text
    with open(HYBRID_REPORT_TXT, "w") as f:
        f.write("HYBRID CNN-BiLSTM-ATTENTION TEST CLASSIFICATION REPORT\n")
        f.write("=" * 65 + "\n")
        f.write(f"Test samples: {len(test_df)}\n\n")
        f.write(clf_report)
    print(f"[OUTPUT] Classification report saved to: {HYBRID_REPORT_TXT}")

    # Save Metrics JSON
    metrics_data = {
        "model": "Hybrid_CNN_BiLSTM_Attention",
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
    with open(HYBRID_METRICS_JSON, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[OUTPUT] Metrics saved to: {HYBRID_METRICS_JSON}")

    # Save Test Predictions CSV
    predictions_df = pd.DataFrame(
        {
            "image_path": test_df["file_path"].values,
            "true_label": [CLASS_NAMES[i] for i in y_true],
            "predicted_label": [CLASS_NAMES[i] for i in y_pred],
            "prob_normal": y_probs[:, 0],
            "prob_benign": y_probs[:, 1],
            "prob_malignant": y_probs[:, 2],
        }
    )
    predictions_df.to_csv(HYBRID_TEST_PREDICTIONS_CSV, index=False)
    print(f"[OUTPUT] Test predictions CSV saved to: {HYBRID_TEST_PREDICTIONS_CSV}")

    # Save Misclassified Samples for Error Analysis
    misclassified_mask = y_pred != y_true
    misclassified_df = predictions_df[misclassified_mask].copy()
    misclassified_df.to_csv(MISCLASSIFIED_SAMPLES_CSV, index=False)
    print(f"[OUTPUT] Misclassified samples CSV ({len(misclassified_df)} samples) saved to: {MISCLASSIFIED_SAMPLES_CSV}")

    # Run Model Comparison against CNN Baseline
    try:
        compare_models()
    except Exception as e:
        print(f"[NOTE] Model comparison notice: {e}")

    # Console display
    print("\n" + "=" * 65)
    print("HYBRID MODEL TEST EVALUATION METRICS (MEASURED):")
    print("=" * 65)
    print(f"Accuracy:           {acc * 100:.2f}%")
    print(f"Precision (Macro):  {prec_macro:.4f}")
    print(f"Recall (Macro):     {rec_macro:.4f}")
    print(f"F1-Score (Macro):   {f1_macro:.4f}")
    if roc_auc_macro is not None:
        print(f"ROC-AUC (Macro):    {roc_auc_macro:.4f}")
    print("\nClassification Report:\n" + clf_report)
    print("=" * 65)
    return metrics_data, predictions_df, misclassified_df


def train_hybrid(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=INITIAL_LEARNING_RATE,
    skip_smoke=False,
):
    """Execute complete training pipeline for the Hybrid CNN-BiLSTM-Attention model."""
    ensure_directories()
    set_seed(RANDOM_SEED)
    device_used = detect_device()

    # Step 1: Pre-training Smoke Test
    if not skip_smoke:
        run_smoke_test(learning_rate=learning_rate)

    # Step 2: Load splits (reusing exact Step 2 manifests)
    print("\n" + "=" * 65)
    print("LOADING DATA SPLITS & COMPUTING CLASS WEIGHTS")
    print("=" * 65)
    train_ds, val_ds, test_ds, train_df, val_df, test_df = get_data_loaders(
        batch_size=batch_size,
        use_sparse=True,
    )

    print(f"Training set:   {len(train_df)} images (69.9%)")
    print(f"Validation set: {len(val_df)} images (15.0%)")
    print(f"Test set:       {len(test_df)} images (15.0%)")

    class_weights = get_balanced_class_weights(train_df)
    print("\nBalanced Class Weights (sklearn.utils.class_weight):")
    for cls_idx, weight in class_weights.items():
        print(f" - Class {cls_idx} ({CLASS_NAMES[cls_idx]:<10s}): {weight:.4f}")

    # Step 3: Build Model
    print("\n" + "=" * 65)
    print("INITIALIZING HYBRID MODEL")
    print("=" * 65)
    model = get_compiled_hybrid_model(
        learning_rate=learning_rate,
        loss="sparse_categorical_crossentropy",
    )
    model.summary()

    # Callbacks as specified by user
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE_EARLY_STOP,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=FACTOR_REDUCE_LR,
            patience=PATIENCE_REDUCE_LR,
            min_lr=MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(HYBRID_MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EpochProgressCallback(),
    ]

    print("\n" + "=" * 65)
    print(f"STARTING FULL TRAINING ({epochs} Epochs, Batch Size: {batch_size}, LR: {learning_rate})")
    print("=" * 65)

    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=0,  # Handled cleanly by EpochProgressCallback
    )
    total_training_time = time.time() - start_time
    epochs_completed = len(history.history["loss"])
    best_val_acc = max(history.history["val_accuracy"])

    print(f"\n[TRAINING FINISHED] Completed {epochs_completed} epochs in {total_training_time:.1f} seconds.")
    print(f"Best Validation Accuracy achieved: {best_val_acc * 100:.2f}%")

    # Save training history CSV
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(HYBRID_HISTORY_CSV, index_label="epoch")
    print(f"[OUTPUT] Training history saved: {HYBRID_HISTORY_CSV}")

    # Save Loss and Accuracy plots
    save_plots(history)

    # Step 4: Load BEST saved model and evaluate on TEST SET ONLY
    print(f"\n[EVALUATION] Loading best model checkpoint from {HYBRID_MODEL_PATH}...")
    best_model = tf.keras.models.load_model(
        str(HYBRID_MODEL_PATH),
        custom_objects={"SequenceAttention": SequenceAttention},
    )

    test_metrics, preds_df, misclass_df = evaluate_hybrid_on_test(best_model, test_df)

    # Final concise summary
    print("\n" + "=" * 65)
    print("FINAL CONCISE TRAINING & EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Best Validation Accuracy : {best_val_acc * 100:.2f}%")
    print(f"Test Accuracy            : {test_metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score           : {test_metrics['f1_score_macro']:.4f}")
    if test_metrics["roc_auc_macro"] is not None:
        print(f"Macro ROC-AUC            : {test_metrics['roc_auc_macro']:.4f}")
    else:
        print(f"Macro ROC-AUC            : N/A")
    print(f"Training Epochs          : {epochs_completed}")
    print(f"Training Time            : {total_training_time:.1f} seconds ({total_training_time/60:.2f} minutes)")
    print(f"Device Used              : {device_used}")
    print("=" * 65)

    return {
        "best_val_accuracy": best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "macro_f1": test_metrics["f1_score_macro"],
        "roc_auc_macro": test_metrics["roc_auc_macro"],
        "epochs_completed": epochs_completed,
        "training_time_seconds": total_training_time,
        "device_used": device_used,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hybrid CNN-BiLSTM-Attention Model for Lung Cancer CT")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=INITIAL_LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--smoke-test", action="store_true", help="Run only the pre-training smoke test")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the pre-training smoke test")

    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test(learning_rate=args.lr)
    else:
        train_hybrid(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            skip_smoke=args.skip_smoke,
        )
