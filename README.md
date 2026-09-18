# Hybrid CNN-BiLSTM-Attention Framework for Lung Cancer Classification from CT Images

An academic AI/ML project adapting a hybrid **CNN + BiLSTM + Attention** deep learning framework for 3-class lung cancer CT image classification (`Normal`, `Benign`, `Malignant`), supported by a benchmark **CNN Baseline**, Grad-CAM visual explainability, and an interactive Streamlit clinical dashboard.

---

## 1. Project Objective & Methodology
Early characterization of pulmonary nodules from Computed Tomography (CT) scans is crucial for clinical triage. This project implements a rigorous two-tier comparative methodology:

1. **Tier 1 (CNN Baseline)**: A pure convolutional network establishing performance benchmarks.
2. **Tier 2 (Hybrid Architecture)**: Adapts a hybrid CNN-BiLSTM-Attention architecture (originally proposed in medical narrative / sequential text literature) to spatial visual feature maps.

> [!IMPORTANT]
> **Architecture Adaptation Note**: The reference research paper used CNN + BiLSTM + Attention for clinical-note text sequences. Here, we adapt the framework to visual CT imaging by treating spatial feature grid locations as an ordered sequence of feature tokens. This is an explicit adaptation for visual spatial modeling, not an exact clinical text reproduction.

---

## 2. Dataset: IQ-OTH/NCCD
Evaluated on the benchmark **IQ-OTH/NCCD** (Iraq-Oncology Teaching Hospital / National Center for Cancer Diseases) dataset.

- **Classes & Sample Distribution (Empirically Verified)**:
  1. `Normal`: 416 images (37.9%) — Healthy pulmonary tissue
  2. `Benign`: 120 images (10.9%) — Non-cancerous nodules/scars
  3. `Malignant`: 561 images (51.1%) — Malignant pulmonary neoplasms
  - **Total Images**: **1,097 images** (100% valid and readable, 0 corrupted)
- **Data Splitting**: 70% Train (767 images), 15% Val (165 images), 15% Test (165 images).
- **Class Imbalance**: Handled using `sklearn.utils.class_weight` (Normal: 0.8786, Benign: 3.0437, Malignant: 0.6522).
- **Data Leakage Limitation**: The public dataset release indexes individual slices sequentially without a patient-to-slice grouping key (110 true patients across 1,097 slices). Hence, a reproducible stratified split (`RANDOM_SEED = 42`) is enforced and saved in `results/metrics/`.

---

## 3. Architecture Deep-Dive (B.Tech Viva Guide)

### Conceptual Flow:
```text
               Input CT Image (224 × 224 × 3)
                            ↓
      ┌─────────────────────────────────────────┐
      │  STAGE 1: CNN Feature Extractor         │
      │  Conv Blocks: 32 → 64 → 128 filters     │
      │  Extracts local spatial representations │
      └─────────────────────────────────────────┘
                            ↓  Feature Map: (7 × 7 × 128)
      ┌─────────────────────────────────────────┐
      │  STAGE 2: Spatial-to-Sequence Reshape   │
      │  Flattens grid into 49 spatial patches  │
      └─────────────────────────────────────────┘
                            ↓  Sequence: (49 timesteps × 128 features)
      ┌─────────────────────────────────────────┐
      │  STAGE 3: Bidirectional LSTM (BiLSTM)   │
      │  Scans sequence forward and backward    │
      │  Captures contextual dependencies       │
      └─────────────────────────────────────────┘
                            ↓  Output: (49 timesteps × 128 hidden)
      ┌─────────────────────────────────────────┐
      │  STAGE 4: Custom Additive Attention     │
      │  Calculates importance weights alpha_t  │
      │  Outputs condensed context vector c     │
      └─────────────────────────────────────────┘
                            ↓  Context Vector: (128,)
      ┌─────────────────────────────────────────┐
      │  STAGE 5: Dense Classifier Head         │
      │  Dense(64, ReLU) + Dropout(0.3)         │
      │  Dense(3) + Softmax Output              │
      └─────────────────────────────────────────┘
                            ↓
            [ Normal (0) | Benign (1) | Malignant (2) ]
```

### Viva Explanation of Key Components:
1. **What the CNN does**: Extracts hierarchical local visual features (edges, parenchymal textures, and nodule margins).
2. **Why convert to a sequence**: High-level convolutional feature maps preserve spatial layout $(H, W)$. Reshaping $(7 \times 7, 128) \rightarrow (49, 128)$ converts 2D spatial patches into a sequence of feature vectors, allowing sequential and relational models to operate across anatomical zones.
3. **What the BiLSTM does**: Standard CNNs treat spatial features independently in global pooling. The BiLSTM scans the spatial patches in both forward and reverse directions, capturing contextual dependencies and transitions between healthy lung tissue and suspicious nodule areas.
4. **What Attention does**: Not all 49 patches contain pathology; most contain background air or normal parenchyma. The custom `SequenceAttention` layer assigns a normalized importance weight $\alpha_t \in [0, 1]$ to each spatial patch ($\sum \alpha_t = 1$), allowing the model to focus primarily on the nodular/tumor region.
5. **Why this differs from a standard CNN**: A standard CNN uses Global Average Pooling (GAP), which treats all spatial locations equally. The Hybrid model uses sequential recurrent modeling (BiLSTM) and adaptive dynamic weighting (Attention) to focus on focal lesion areas.

---

## 4. Environment & Installation

```powershell
# 1. Create and activate virtual environment (Python 3.11 recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify environment
python -c "import tensorflow as tf; print('TensorFlow Version:', tf.__version__)"
```

---

## 5. Execution Workflow

### Step A: Dataset Inspection & Manifest Generation
```powershell
python src/inspect_dataset.py
```

### Step B: CNN Baseline
```powershell
# Instant sanity check:
python src/train.py --sanity-check

# Full training and evaluation:
python src/train.py --epochs 30 --batch-size 16 --lr 0.0001
```

### Step C: CNN + BiLSTM Ablation (Without Attention)
```powershell
# Train and evaluate ablation model:
python src/train_ablation.py --epochs 30 --batch-size 16 --lr 0.0001
```

### Step D: Final Hybrid CNN-BiLSTM-Attention Model
```powershell
# Instant sanity check (verifies build, shapes, loss, gradient step, and saving):
python src/train_hybrid.py --sanity-check

# Full hybrid training and evaluation:
python src/train_hybrid.py --epochs 30 --batch-size 16 --lr 0.0001
```

### Step E: Objective Model Comparison & Ablation Study
```powershell
python src/compare_models.py
```
*Generates an objective side-by-side comparison table from measured test metrics and saves `results/metrics/ablation_comparison.csv` and `results/plots/ablation_comparison.png`.*

### Step F: Streamlit Clinical AI Dashboard
```powershell
streamlit run app/app.py
```

---

## 6. Output Artifacts Directory Layout

```text
results/
├── plots/
│   ├── cnn_baseline_accuracy.png
│   ├── cnn_baseline_loss.png
│   ├── hybrid_accuracy.png
│   └── hybrid_loss.png
├── confusion_matrix/
│   ├── cnn_baseline_confusion_matrix.png
│   └── hybrid_confusion_matrix.png
├── metrics/
│   ├── train_manifest.csv
│   ├── val_manifest.csv
│   ├── test_manifest.csv
│   ├── cnn_baseline_history.csv
│   ├── cnn_baseline_metrics.json
│   ├── cnn_baseline_classification_report.txt
│   ├── hybrid_history.csv
│   ├── hybrid_metrics.json
│   ├── hybrid_classification_report.txt
│   └── model_comparison.csv
└── gradcam/
    └── (visual heatmaps)
```

---

## 7. Limitations & Ethical Disclaimer

> [!WARNING]
> **Medical & Regulatory Disclaimer**: This software is strictly an academic demonstration and research system developed for college coursework evaluation. It is **NOT** a certified medical diagnostic device, does not diagnose cancer, and must never replace consultation with certified radiologists and medical professionals.