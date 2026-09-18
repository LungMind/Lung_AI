"""Final Evaluation, Visualization, Explainability, and Error Analysis Script."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)

from src.config import (
    HYBRID_MODEL_PATH,
    CNN_BASELINE_MODEL_PATH,
    CLASS_NAMES,
    IMAGE_SIZE,
    TEST_MANIFEST_CSV,
    HYBRID_HISTORY_CSV,
    RESULTS_DIR,
    PLOTS_DIR,
    METRICS_DIR,
    CONFUSION_MATRIX_DIR,
    GRADCAM_DIR,
    ensure_directories,
)
from src.attention import SequenceAttention
from src.preprocessing import load_and_preprocess_image
from src.explainability import generate_gradcam_heatmap, overlay_gradcam


def run_final_evaluation():
    """Execute complete Step 6 evaluation, plotting, Grad-CAM, error analysis, and comparison."""
    ensure_directories()
    print("=" * 70)
    print("STEP 6: FINAL EVALUATION, VISUALIZATION & ERROR ANALYSIS")
    print("=" * 70)

    # 1. Load trained hybrid model
    if not HYBRID_MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained hybrid model not found at: {HYBRID_MODEL_PATH.resolve()}")

    print(f"\n[1] Loading trained Hybrid Model from: {HYBRID_MODEL_PATH.resolve()}...")
    model = tf.keras.models.load_model(
        str(HYBRID_MODEL_PATH),
        custom_objects={"SequenceAttention": SequenceAttention},
    )
    print("    [PASS] Hybrid model successfully loaded.")

    # 2. Load held-out test set
    if not TEST_MANIFEST_CSV.exists():
        raise FileNotFoundError(f"Test manifest not found at: {TEST_MANIFEST_CSV.resolve()}")

    print(f"\n[2] Loading held-out test manifest from: {TEST_MANIFEST_CSV.resolve()}...")
    test_df = pd.read_csv(TEST_MANIFEST_CSV)
    y_true = test_df["label"].values.astype(int)
    print(f"    Loaded {len(test_df)} test CT slices.")

    # Preprocess test images
    print("    Preprocessing test slices (224x224, float32, [0,1])...")
    X_test = []
    for path in test_df["file_path"]:
        img = load_and_preprocess_image(path, target_size=IMAGE_SIZE)
        X_test.append(img)
    X_test = np.array(X_test, dtype=np.float32)

    # 3. Model inference on test set
    print("\n[3] Running model inference on test set...")
    y_probs = model.predict(X_test, batch_size=16, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # 4. Calculate actual metrics
    print("\n[4] Calculating actual evaluation metrics...")
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # Multiclass ROC-AUC (One-vs-Rest)
    y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(CLASS_NAMES))
    try:
        roc_auc_macro = float(roc_auc_score(y_true_onehot, y_probs, multi_class="ovr", average="macro"))
        roc_auc_weighted = float(roc_auc_score(y_true_onehot, y_probs, multi_class="ovr", average="weighted"))
    except Exception as e:
        print(f"    [NOTE] ROC-AUC calculation note: {e}")
        roc_auc_macro = None
        roc_auc_weighted = None

    cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

    # Save metrics JSON
    final_metrics_json = METRICS_DIR / "final_metrics.json"
    metrics_data = {
        "model": "Hybrid_CNN_BiLSTM_Attention",
        "total_test_samples": len(test_df),
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "weighted_precision": prec_weighted,
        "weighted_recall": rec_weighted,
        "weighted_f1": f1_weighted,
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
    with open(final_metrics_json, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"    [OUTPUT] Saved: {final_metrics_json}")

    # Save classification report text
    final_report_txt = METRICS_DIR / "final_classification_report.txt"
    with open(final_report_txt, "w") as f:
        f.write("HYBRID CNN-BiLSTM-ATTENTION FINAL CLASSIFICATION REPORT\n")
        f.write("=" * 65 + "\n")
        f.write(f"Test Samples: {len(test_df)}\n\n")
        f.write(clf_report)
    print(f"    [OUTPUT] Saved: {final_report_txt}")

    # 5. Confusion Matrices (Raw and Normalized)
    print("\n[5] Generating Confusion Matrix Visualizations...")
    # Raw confusion matrix
    cm_raw_path = CONFUSION_MATRIX_DIR / "final_confusion_matrix.png"
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Hybrid Model - Confusion Matrix (Counts)", fontsize=12, fontweight="bold")
    plt.ylabel("Actual Ground Truth", fontsize=10)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.tight_layout()
    plt.savefig(cm_raw_path, dpi=300)
    plt.close()
    print(f"    [OUTPUT] Saved: {cm_raw_path}")

    # Normalized confusion matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm_path = CONFUSION_MATRIX_DIR / "final_confusion_matrix_normalized.png"
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Hybrid Model - Normalized Confusion Matrix (Proportions)", fontsize=12, fontweight="bold")
    plt.ylabel("Actual Ground Truth", fontsize=10)
    plt.xlabel("Predicted Class", fontsize=10)
    plt.tight_layout()
    plt.savefig(cm_norm_path, dpi=300)
    plt.close()
    print(f"    [OUTPUT] Saved: {cm_norm_path}")

    # 6. Multiclass ROC Curves (One-vs-Rest)
    print("\n[6] Generating Multiclass ROC Curves...")
    roc_plot_path = PLOTS_DIR / "final_roc_curve.png"
    plt.figure(figsize=(8, 6))
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    for i, class_name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_probs[:, i])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f"ROC {class_name} (AUC = {roc_auc_val:.3f})")

    # Micro-average ROC
    fpr_micro, tpr_micro, _ = roc_curve(y_true_onehot.ravel(), y_probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, color="navy", linestyle=":", lw=2.5, label=f"Micro-average ROC (AUC = {roc_auc_micro:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.title("Multiclass One-vs-Rest ROC Curves", fontsize=12, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"    [OUTPUT] Saved: {roc_plot_path}")

    # 7. Precision / Recall / F1 Bar Chart
    print("\n[7] Generating Classification Metrics Comparison Bar Chart...")
    metrics_bar_path = PLOTS_DIR / "classification_metrics.png"
    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    plt.figure(figsize=(9, 5.5))
    plt.bar(x - width, prec_per_class, width, label="Precision", color="#3498db")
    plt.bar(x, rec_per_class, width, label="Recall", color="#2ecc71")
    plt.bar(x + width, f1_per_class, width, label="F1-Score", color="#e74c3c")

    plt.ylabel("Score (0.0 - 1.0)", fontsize=10)
    plt.title("Per-Class Classification Metrics (Test Set)", fontsize=12, fontweight="bold")
    plt.xticks(x, CLASS_NAMES, fontsize=10)
    plt.ylim([0.0, 1.1])
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Annotate values on bars
    for i in range(len(CLASS_NAMES)):
        plt.text(i - width, prec_per_class[i] + 0.02, f"{prec_per_class[i]:.2f}", ha="center", fontsize=8)
        plt.text(i, rec_per_class[i] + 0.02, f"{rec_per_class[i]:.2f}", ha="center", fontsize=8)
        plt.text(i + width, f1_per_class[i] + 0.02, f"{f1_per_class[i]:.2f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(metrics_bar_path, dpi=300)
    plt.close()
    print(f"    [OUTPUT] Saved: {metrics_bar_path}")

    # 8. Training Curves from saved history
    print("\n[8] Generating Final Training Curves from History...")
    if HYBRID_HISTORY_CSV.exists():
        history_df = pd.read_csv(HYBRID_HISTORY_CSV)

        # Accuracy Plot
        train_acc_path = PLOTS_DIR / "final_training_accuracy.png"
        plt.figure(figsize=(8, 5))
        plt.plot(history_df["accuracy"], label="Training Accuracy", color="#1f77b4", lw=2)
        plt.plot(history_df["val_accuracy"], label="Validation Accuracy", color="#ff7f0e", lw=2)
        plt.title("Training Accuracy vs Validation Accuracy", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Accuracy", fontsize=10)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(train_acc_path, dpi=300)
        plt.close()
        print(f"    [OUTPUT] Saved: {train_acc_path}")

        # Loss Plot
        train_loss_path = PLOTS_DIR / "final_training_loss.png"
        plt.figure(figsize=(8, 5))
        plt.plot(history_df["loss"], label="Training Loss", color="#1f77b4", lw=2)
        plt.plot(history_df["val_loss"], label="Validation Loss", color="#ff7f0e", lw=2)
        plt.title("Training Loss vs Validation Loss", fontsize=12, fontweight="bold")
        plt.xlabel("Epoch", fontsize=10)
        plt.ylabel("Cross-Entropy Loss", fontsize=10)
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(train_loss_path, dpi=300)
        plt.close()
        print(f"    [OUTPUT] Saved: {train_loss_path}")
    else:
        print(f"    [NOTE] History file {HYBRID_HISTORY_CSV} not found yet; skipping curve generation.")

    # 9. Grad-CAM Generation for Sample Images
    print("\n[9] Generating Grad-CAM Heatmaps across Representative Classes...")
    # Select 3 samples from test set representing different ground truth / predicted classes
    sample_indices = []
    for c_idx in range(len(CLASS_NAMES)):
        class_samples = np.where(y_true == c_idx)[0]
        if len(class_samples) > 0:
            sample_indices.append(class_samples[0])

    gradcam_outputs = []
    for idx_num, sample_idx in enumerate(sample_indices, 1):
        sample_path = test_df["file_path"].iloc[sample_idx]
        true_cls = CLASS_NAMES[y_true[sample_idx]]
        pred_cls = CLASS_NAMES[y_pred[sample_idx]]
        conf = float(y_probs[sample_idx, y_pred[sample_idx]])

        raw_bgr = cv2.imread(sample_path)
        if raw_bgr is not None:
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            raw_resized = cv2.resize(raw_rgb, IMAGE_SIZE)
        else:
            raw_resized = np.uint8(255 * X_test[sample_idx])

        # Compute Grad-CAM
        heatmap = generate_gradcam_heatmap(model, np.expand_dims(X_test[sample_idx], axis=0), pred_index=y_pred[sample_idx])
        overlay = overlay_gradcam(raw_resized, heatmap, alpha=0.45)

        # Plot side-by-side: Original CT | Heatmap | Overlay
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(raw_resized)
        axes[0].set_title(f"Original CT Slice\nTrue: {true_cls}", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Activation Heatmap", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay\nPred: {pred_cls} ({conf*100:.1f}%)", fontsize=10)
        axes[2].axis("off")

        save_cam_path = GRADCAM_DIR / f"sample_0{idx_num}_gradcam.png"
        plt.tight_layout()
        plt.savefig(save_cam_path, dpi=300)
        plt.close()
        print(f"    [OUTPUT] Saved: {save_cam_path} (True: {true_cls}, Pred: {pred_cls})")
        gradcam_outputs.append(str(save_cam_path))

    # 10. Error Analysis & Predictions CSV
    print("\n[10] Performing Error Analysis on Test Predictions...")
    confidences = np.max(y_probs, axis=1)
    predictions_df = pd.DataFrame(
        {
            "image_path": test_df["file_path"].values,
            "true_label": [CLASS_NAMES[i] for i in y_true],
            "predicted_label": [CLASS_NAMES[i] for i in y_pred],
            "confidence": confidences,
            "prob_normal": y_probs[:, 0],
            "prob_benign": y_probs[:, 1],
            "prob_malignant": y_probs[:, 2],
        }
    )

    # Save test predictions CSV
    test_preds_csv = METRICS_DIR / "hybrid_test_predictions.csv"
    predictions_df.to_csv(test_preds_csv, index=False)
    print(f"    [OUTPUT] Saved: {test_preds_csv}")

    # Identify misclassified samples
    incorrect_mask = y_pred != y_true
    misclassified_df = predictions_df[incorrect_mask][["image_path", "true_label", "predicted_label", "confidence"]].copy()

    error_analysis_csv = METRICS_DIR / "error_analysis.csv"
    misclassified_df.to_csv(error_analysis_csv, index=False)
    print(f"    [OUTPUT] Saved: {error_analysis_csv}")

    total_samples = len(test_df)
    correct_samples = int((y_pred == y_true).sum())
    incorrect_samples = int((y_pred != y_true).sum())
    error_rate = incorrect_samples / total_samples

    print(f"\n    Error Analysis Summary:")
    print(f"    - Total Test Samples:       {total_samples}")
    print(f"    - Correctly Classified:     {correct_samples} ({correct_samples/total_samples*100:.2f}%)")
    print(f"    - Incorrectly Classified:   {incorrect_samples} ({error_rate*100:.2f}%)")
    print(f"    - Error Rate:               {error_rate:.4f}")

    # 11. Final Model Comparison
    print("\n[11] Generating Final Model Comparison CSV...")
    baseline_metrics_path = METRICS_DIR / "cnn_baseline_metrics.json"
    baseline_metrics = None
    if baseline_metrics_path.exists():
        with open(baseline_metrics_path, "r") as f:
            baseline_metrics = json.load(f)

    comparison_records = []
    if baseline_metrics:
        comparison_records.append(
            {
                "Model": "CNN Baseline",
                "Accuracy": f"{baseline_metrics.get('accuracy', 0.0) * 100:.2f}%",
                "Precision": f"{baseline_metrics.get('precision_macro', 0.0):.4f}",
                "Recall": f"{baseline_metrics.get('recall_macro', 0.0):.4f}",
                "F1": f"{baseline_metrics.get('f1_score_macro', 0.0):.4f}",
                "ROC-AUC": (
                    f"{baseline_metrics['roc_auc_macro']:.4f}"
                    if baseline_metrics.get("roc_auc_macro") is not None
                    else "N/A"
                ),
            }
        )
    else:
        comparison_records.append(
            {
                "Model": "CNN Baseline",
                "Accuracy": "Not evaluated",
                "Precision": "N/A",
                "Recall": "N/A",
                "F1": "N/A",
                "ROC-AUC": "N/A",
            }
        )

    comparison_records.append(
        {
            "Model": "Hybrid CNN-BiLSTM-Attention",
            "Accuracy": f"{acc * 100:.2f}%",
            "Precision": f"{prec_macro:.4f}",
            "Recall": f"{rec_macro:.4f}",
            "F1": f"{f1_macro:.4f}",
            "ROC-AUC": f"{roc_auc_macro:.4f}" if roc_auc_macro is not None else "N/A",
        }
    )

    final_comp_df = pd.DataFrame(comparison_records)
    final_comp_csv = METRICS_DIR / "final_model_comparison.csv"
    final_comp_df.to_csv(final_comp_csv, index=False)
    print(f"    [OUTPUT] Saved: {final_comp_csv}")

    print("\n" + "=" * 70)
    print("FINAL TEST METRICS (ACTUAL MEASURED):")
    print("=" * 70)
    print(f"Accuracy:           {acc * 100:.2f}%")
    print(f"Precision (Macro):  {prec_macro:.4f}")
    print(f"Recall (Macro):     {rec_macro:.4f}")
    print(f"F1-Score (Macro):   {f1_macro:.4f}")
    if roc_auc_macro is not None:
        print(f"ROC-AUC (Macro):    {roc_auc_macro:.4f}")
    print("\nClassification Report:\n" + clf_report)
    print("=" * 70)

    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "roc_auc_macro": roc_auc_macro,
        "total_samples": total_samples,
        "correct": correct_samples,
        "incorrect": incorrect_samples,
        "error_rate": error_rate,
        "gradcam_outputs": gradcam_outputs,
    }


if __name__ == "__main__":
    run_final_evaluation()
