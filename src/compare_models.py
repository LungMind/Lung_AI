"""Model Comparison and Ablation Study Script.

Compares measured metrics across:
1. CNN (Baseline)
2. CNN + BiLSTM (Ablation without Attention)
3. CNN + BiLSTM + Attention (Full Hybrid Architecture)

Outputs:
- results/metrics/ablation_comparison.csv
- results/metrics/model_comparison.csv
- results/plots/ablation_comparison.png
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    CNN_BASELINE_METRICS_JSON,
    CNN_BILSTM_METRICS_JSON,
    HYBRID_METRICS_JSON,
    MODEL_COMPARISON_CSV,
    ABLATION_COMPARISON_CSV,
    ABLATION_COMPARISON_PLOT,
    ensure_directories,
)


def load_metrics_file(filepath):
    """Load JSON metrics if file exists and is valid."""
    path = Path(filepath)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load {filepath}: {e}")
        return None


def generate_ablation_comparison():
    """Build comparison table and visualization for CNN, CNN+BiLSTM, and CNN+BiLSTM+Attention."""
    ensure_directories()

    cnn_metrics = load_metrics_file(CNN_BASELINE_METRICS_JSON)
    bilstm_metrics = load_metrics_file(CNN_BILSTM_METRICS_JSON)
    hybrid_metrics = load_metrics_file(HYBRID_METRICS_JSON)

    models_info = [
        ("CNN", cnn_metrics),
        ("CNN + BiLSTM", bilstm_metrics),
        ("CNN + BiLSTM + Attention", hybrid_metrics),
    ]

    records = []
    plot_data = []

    for name, m in models_info:
        if m:
            acc = m.get("accuracy", 0.0)
            prec = m.get("precision_macro", 0.0)
            rec = m.get("recall_macro", 0.0)
            f1 = m.get("f1_score_macro", 0.0)
            auc = m.get("roc_auc_macro", None)

            records.append(
                {
                    "Model": name,
                    "Accuracy": f"{acc * 100:.2f}%",
                    "Precision (Macro)": f"{prec:.4f}",
                    "Recall (Macro)": f"{rec:.4f}",
                    "F1-Score (Macro)": f"{f1:.4f}",
                    "ROC-AUC (Macro)": f"{auc:.4f}" if auc is not None else "N/A",
                }
            )

            plot_data.append(
                {
                    "Model": name,
                    "Accuracy": acc * 100.0,
                    "Precision (Macro)": prec,
                    "Recall (Macro)": rec,
                    "F1-Score (Macro)": f1,
                    "ROC-AUC (Macro)": auc if auc is not None else 0.0,
                }
            )
        else:
            records.append(
                {
                    "Model": name,
                    "Accuracy": "Not evaluated yet",
                    "Precision (Macro)": "N/A",
                    "Recall (Macro)": "N/A",
                    "F1-Score (Macro)": "N/A",
                    "ROC-AUC (Macro)": "N/A",
                }
            )

    comparison_df = pd.DataFrame(records)

    # Save CSVs
    comparison_df.to_csv(ABLATION_COMPARISON_CSV, index=False)
    comparison_df.to_csv(MODEL_COMPARISON_CSV, index=False)
    print(f"[OUTPUT] Ablation comparison table saved to: {ABLATION_COMPARISON_CSV}")
    print(f"[OUTPUT] Model comparison table saved to: {MODEL_COMPARISON_CSV}")

    # Print Table
    print("\n" + "=" * 88)
    print("ABLATION & MODEL COMPARISON TABLE (Held-out Test Set - 165 CT Slices)")
    print("=" * 88)
    print(comparison_df.to_string(index=False))
    print("=" * 88)

    # Create visualization if at least one model has measured metrics
    if plot_data:
        create_ablation_plot(plot_data)

    return comparison_df


def create_ablation_plot(plot_data):
    """Create a publication-quality grouped bar chart for the ablation study."""
    models = [d["Model"] for d in plot_data]
    n_models = len(models)

    # Metrics to display on separate subplots or grouped
    metrics_keys = [
        ("Accuracy", "Accuracy (%)", "%"),
        ("F1-Score (Macro)", "Macro F1-Score", ""),
        ("Precision (Macro)", "Macro Precision", ""),
        ("Recall (Macro)", "Macro Recall", ""),
        ("ROC-AUC (Macro)", "Macro ROC-AUC", ""),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 5))
    colors = ["#2b5c8f", "#2a9d8f", "#e76f51"]  # Distinct professional palette

    for i, (key, title, unit) in enumerate(metrics_keys):
        ax = axes[i]
        vals = [d[key] for d in plot_data]
        bars = ax.bar(models, vals, color=colors[:n_models], width=0.55, edgecolor="black", linewidth=0.8)

        # Annotate bars with measured values
        for bar in bars:
            height = bar.get_height()
            if unit == "%":
                text = f"{height:.2f}%"
            else:
                text = f"{height:.3f}"
            ax.annotate(
                text,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        if unit == "%":
            ax.set_ylim(0, 105)
        else:
            ax.set_ylim(0, 1.05)

        ax.set_xticks(range(n_models))
        ax.set_xticklabels(models, rotation=25, ha="right", fontsize=9)

    plt.suptitle(
        "Ablation Experiment: Architecture Comparison on Held-Out Test Set",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(ABLATION_COMPARISON_PLOT, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OUTPUT] Ablation comparison plot saved to: {ABLATION_COMPARISON_PLOT}")


if __name__ == "__main__":
    generate_ablation_comparison()
