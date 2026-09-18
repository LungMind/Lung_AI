import sys
from pathlib import Path

# Add project root to sys.path so both 'python src/evaluate.py' and 'python -m src.evaluate' work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
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
    CNN_BASELINE_MODEL_PATH,
    CLASS_NAMES,
    IMAGE_SIZE,
    TEST_MANIFEST_CSV,
    CNN_BASELINE_REPORT_TXT,
    CNN_BASELINE_METRICS_JSON,
    CNN_BASELINE_CM_PNG,
    ensure_directories,
)
from src.preprocessing import load_and_preprocess_image


def evaluate_baseline(model_path=CNN_BASELINE_MODEL_PATH, manifest_path=TEST_MANIFEST_CSV):
    """Run full evaluation on the holdout test set."""
    ensure_directories()
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found at: {Path(model_path).resolve()}.\n"
            "Run training first: python src/train.py"
        )

    if not Path(manifest_path).exists():
        raise FileNotFoundError(
            f"Test manifest not found at: {Path(manifest_path).resolve()}.\n"
            "Run dataset preparation first: python src/inspect_dataset.py"
        )

    print(f"[EVAL] Loading model from: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    print(f"[EVAL] Loading test manifest from: {manifest_path}")
    test_df = pd.read_csv(manifest_path)
    print(f"[EVAL] Preprocessing {len(test_df)} test images...")

    X_test = []
    y_true = test_df["label"].values

    for path in test_df["file_path"]:
        img = load_and_preprocess_image(path, target_size=IMAGE_SIZE)
        X_test.append(img)
    X_test = np.array(X_test, dtype=np.float32)

    # Predictions
    print("[EVAL] Running inference...")
    y_probs = model.predict(X_test, batch_size=16, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # Metrics
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    try:
        y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(CLASS_NAMES))
        roc_auc_macro = float(roc_auc_score(y_true_onehot, y_probs, multi_class="ovr", average="macro"))
    except Exception as e:
        print(f"[WARNING] ROC-AUC calculation notice: {e}")
        roc_auc_macro = None

    cm = confusion_matrix(y_true, y_pred)
    clf_report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)

    # Save Confusion Matrix Heatmap
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("CNN Baseline - Confusion Matrix", fontsize=12, fontweight="bold")
    plt.ylabel("Ground Truth", fontsize=10)
    plt.xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()
    plt.savefig(CNN_BASELINE_CM_PNG, dpi=300)
    plt.close()
    print(f"[OUTPUT] Confusion matrix saved to: {CNN_BASELINE_CM_PNG}")

    # Save Classification Report
    with open(CNN_BASELINE_REPORT_TXT, "w") as f:
        f.write("CNN BASELINE TEST CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test samples: {len(test_df)}\n\n")
        f.write(clf_report)
    print(f"[OUTPUT] Classification report saved to: {CNN_BASELINE_REPORT_TXT}")

    # Save JSON metrics
    metrics_data = {
        "model": "CNN_Baseline",
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "precision_weighted": prec_weighted,
        "recall_weighted": rec_weighted,
        "f1_weighted": f1_weighted,
        "roc_auc_macro": roc_auc_macro,
        "total_test_samples": len(test_df),
        "confusion_matrix": cm.tolist(),
    }
    with open(CNN_BASELINE_METRICS_JSON, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"[OUTPUT] Metrics saved to: {CNN_BASELINE_METRICS_JSON}")

    # Console display
    print("\n" + "=" * 50)
    print("CNN BASELINE EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:           {acc * 100:.2f}%")
    print(f"Precision (Macro):  {prec_macro:.4f}")
    print(f"Recall (Macro):     {rec_macro:.4f}")
    print(f"F1-Score (Macro):   {f1_macro:.4f}")
    if roc_auc_macro is not None:
        print(f"ROC-AUC (Macro):    {roc_auc_macro:.4f}")
    print("=" * 50)
    print(clf_report)
    return metrics_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained CNN baseline model")
    parser.add_argument("--model-path", type=str, default=str(CNN_BASELINE_MODEL_PATH))
    parser.add_argument("--manifest", type=str, default=str(TEST_MANIFEST_CSV))
    args = parser.parse_args()
    evaluate_baseline(model_path=args.model_path, manifest_path=args.manifest)
