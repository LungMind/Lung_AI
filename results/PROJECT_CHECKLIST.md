# Project Reproducibility & Integrity Checklist

**Project Title**: Hybrid CNN-BiLSTM-Attention Framework for Lung Cancer Classification from CT Images  
**Audit Date**: 2026-09-18  
**Environment**: Python 3.11.9, TensorFlow 2.21.0, Windows 11  

---

## Executive Summary

A comprehensive integrity audit was conducted across the codebase, dataset pipelines, training artifacts, saved models, and user interfaces. All **13 audit items passed verification**.

| # | Verification Item | Status | Key Evidence / Verification Mechanism |
| :-: | :--- | :-: | :--- |
| **1** | All Python imports are present in `requirements.txt` | **PASS** | Dependencies verified across `src/` and `app/` (`tensorflow`, `keras`, `scikit-learn`, `numpy`, `pandas`, `opencv-python-headless`, `Pillow`, `matplotlib`, `seaborn`, `streamlit`, `tqdm`). |
| **2** | No absolute Windows/Linux paths are hardcoded | **PASS** | Regex scan `(C:\\\|c:/\|/home/\|/Users/)` returned 0 occurrences in all `.py` files. All paths derive dynamically from `PROJECT_ROOT = Path(__file__).resolve().parent.parent` via `pathlib.Path`. |
| **3** | Dataset files are excluded by `.gitignore` | **PASS** | `The IQ-OTHNCCD lung cancer dataset/`, `data/*`, `Test cases/`, and `*.dcm` are explicitly ignored. Validated with `git check-ignore`. |
| **4** | Model weights are excluded from Git if appropriate | **PASS** | `models/*`, `*.keras`, `*.h5`, `*.ckpt*`, `*.pt`, `*.pth` are ignored while preserving `.gitkeep`. Validated with `git check-ignore`. |
| **5** | Random seeds are configured | **PASS** | `RANDOM_SEED = 42` centralized in `src/config.py`. Uniform `set_seed()` enforces determinism across `random`, `numpy`, `tensorflow`, and `PYTHONHASHSEED`. Splits and batch shuffling utilize this seed. |
| **6** | Configuration values are centralized | **PASS** | Centralized in `src/config.py`: dataset paths, image dimensions (224x224x3), batch size (16), initial learning rate (1e-4), scheduler parameters, model hyperparameters, and artifact paths. |
| **7** | Training and evaluation use the same documented dataset split | **PASS** | Stratified split (70% Train [767], 15% Val [165], 15% Test [165]) saved to `results/metrics/*_manifest.csv`. All models (`CNN`, `CNN+BiLSTM`, `CNN+BiLSTM+Attention`) evaluate on the identical held-out test manifest. |
| **8** | Test data is never used for training | **PASS** | Training routines (`train.py`, `train_ablation.py`, `train_hybrid.py`) pass only `train_ds` and `val_ds` to `model.fit()`. Data augmentation is applied only during training (`is_training=True`). Test data is evaluated post-training only. |
| **9** | No metrics are hardcoded | **PASS** | Source code contains no static performance numbers. Comparison tables, summary plots, and classification reports dynamically load measured test metrics from JSON files. |
| **10** | No fake Grad-CAM images exist | **PASS** | Grad-CAM in `src/explainability.py` uses genuine `tf.GradientTape` autodiff on `last_conv_layer`, computing feature gradients and applying OpenCV colormap overlays. |
| **11** | The saved model can be loaded in a fresh Python process | **PASS** | Verified by running an isolated Python process that loaded `hybrid_cnn_bilstm_attention.keras`, `cnn_bilstm.keras`, and `cnn_baseline.keras`, validating input shapes `(None, 224, 224, 3)` and output shapes `(None, 3)`. |
| **12** | Streamlit can load the saved model | **PASS** | Verified `app.app.load_trained_model()` in an isolated Python process; successfully cached and returned the hybrid model with output shape `(None, 3)`. |
| **13** | README contains installation and execution instructions | **PASS** | `README.md` documents virtual environment creation, `pip install -r requirements.txt`, dataset inspection, training commands for all three models, comparison generation, and Streamlit launch. |

---

## Detailed Item Verification & Audit Findings

### Item 1: Python Imports in `requirements.txt`
- **Verification Status**: **PASS**
- **Files Checked**: `requirements.txt`, `src/*.py`, `app/*.py`
- **Details**: All external third-party imports (`tensorflow`, `keras`, `scikit-learn`, `numpy`, `pandas`, `cv2`, `PIL`, `matplotlib`, `seaborn`, `streamlit`, `tqdm`) are pinned with valid version bounds in `requirements.txt`.
- **Remediation if Failed**: N/A.

---

### Item 2: Absence of Hardcoded Absolute Paths
- **Verification Status**: **PASS**
- **Verification Method**: Automated ripgrep search for filesystem root patterns (`C:\`, `c:/`, `/home/`, `/Users/`) across all codebase Python files.
- **Details**: All paths use dynamic `Path(__file__).resolve().parent...` anchors. Paths resolve seamlessly whether executed on Windows, Linux, or macOS.
- **Remediation if Failed**: N/A.

---

### Item 3: Dataset Files Excluded by `.gitignore`
- **Verification Status**: **PASS**
- **Verification Method**: `git check-ignore "The IQ-OTHNCCD lung cancer dataset"` confirmed matching ignore rules.
- **Details**: Large medical image directories and raw archive folders are ignored, preventing multi-hundred megabyte commits to git repositories.
- **Remediation if Failed**: N/A.

---

### Item 4: Model Checkpoints Excluded from Git
- **Verification Status**: **PASS**
- **Verification Method**: `git check-ignore "models/hybrid_cnn_bilstm_attention.keras" "models/cnn_bilstm.keras"` confirmed matching ignore rules.
- **Details**: Binary model weight checkpoints (`*.keras`, `*.h5`, `models/*`) are excluded from tracking, while the empty directory structure is preserved via `models/.gitkeep`.
- **Remediation if Failed**: N/A.

---

### Item 5: Random Seed Configuration
- **Verification Status**: **PASS**
- **Details**: `RANDOM_SEED = 42` is defined in `src/config.py`. The helper `set_seed()` is invoked prior to dataset partitioning and model training across all three architectures, configuring:
  ```python
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  os.environ["PYTHONHASHSEED"] = str(seed)
  ```
- **Remediation if Failed**: N/A.

---

### Item 6: Centralized Configuration
- **Verification Status**: **PASS**
- **Details**: Hyperparameters (image dimensions $224 \times 224 \times 3$, batch size $16$, initial learning rate $1 \times 10^{-4}$, LSTM units $64$, dense units $64$, dropout $0.3$, early stopping patience $5$, reduce LR patience $2$), class mappings, and directory structures reside exclusively in `src/config.py`.
- **Remediation if Failed**: N/A.

---

### Item 7: Consistency of Dataset Splits
- **Verification Status**: **PASS**
- **Details**: Partitioning was performed once and serialized to disk:
  - `results/metrics/train_manifest.csv` (767 slices)
  - `results/metrics/val_manifest.csv` (165 slices)
  - `results/metrics/test_manifest.csv` (165 slices)
  All models (`CNN Baseline`, `CNN + BiLSTM`, and `CNN + BiLSTM + Attention`) and evaluation scripts consume these identical manifests.
- **Remediation if Failed**: N/A.

---

### Item 8: Strict Isolation of Test Data
- **Verification Status**: **PASS**
- **Details**:
  - `model.fit()` receives exclusively `train_ds` and `val_ds`.
  - Data augmentation (rotation, zoom, flips) is guarded by `is_training=True` and applied only to the training loader.
  - Test data is strictly held out until final checkpoint evaluation.
- **Remediation if Failed**: N/A.

---

### Item 9: Authenticity of Reported Metrics
- **Verification Status**: **PASS**
- **Details**: No metrics are hardcoded. All tabular values, classification reports, and plot annotations are populated directly from the JSON files produced by scikit-learn and TensorFlow during test evaluation:
  - `cnn_baseline_metrics.json`
  - `cnn_bilstm_metrics.json`
  - `hybrid_metrics.json`
- **Remediation if Failed**: N/A.

---

### Item 10: Authenticity of Grad-CAM Visualizations
- **Verification Status**: **PASS**
- **Details**: `src/explainability.py` extracts the symbolic tensor of `last_conv_layer` and constructs a gradient sub-model:
  ```python
  grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[target_conv_layer.output, model.output])
  with tf.GradientTape() as tape:
      conv_outputs, predictions = grad_model(img_array)
      target_score = predictions[:, pred_index]
  grads = tape.gradient(target_score, conv_outputs)
  pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
  heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
  ```
  Actual backpropagation through Dense $\rightarrow$ SequenceAttention $\rightarrow$ BiLSTM $\rightarrow$ Reshape $\rightarrow$ Conv2D produces authentic spatial gradient heatmaps.
- **Remediation if Failed**: N/A.

---

### Item 11: Standalone Model Loading Verification
- **Verification Status**: **PASS**
- **Execution Test**: An isolated subprocess successfully loaded all three serialized models using `tf.keras.models.load_model`:
  - `models/hybrid_cnn_bilstm_attention.keras`: Inputs `(None, 224, 224, 3)`, Outputs `(None, 3)`
  - `models/cnn_bilstm.keras`: Inputs `(None, 224, 224, 3)`, Outputs `(None, 3)`
  - `models/cnn_baseline.keras`: Inputs `(None, 224, 224, 3)`, Outputs `(None, 3)`
- **Remediation if Failed**: N/A.

---

### Item 12: Streamlit Model Loading Verification
- **Verification Status**: **PASS**
- **Execution Test**: Invoked `app.app.load_trained_model()` via an independent Python process. Confirmed that `@st.cache_resource` loads the model with `custom_objects={'SequenceAttention': SequenceAttention}` and returns a valid model instance with output shape `(None, 3)`.
- **Remediation if Failed**: N/A.

---

### Item 13: Completeness of Documentation
- **Verification Status**: **PASS**
- **Details**: `README.md` provides explicit step-by-step instructions for:
  1. Virtual environment creation & `pip install -r requirements.txt`
  2. Dataset inspection (`python src/inspect_dataset.py`)
  3. CNN baseline training (`python src/train.py`)
  4. CNN + BiLSTM ablation training (`python src/train_ablation.py`)
  5. Hybrid CNN-BiLSTM-Attention training (`python src/train_hybrid.py`)
  6. Multi-model comparison generation (`python src/compare_models.py`)
  7. Interactive Streamlit app execution (`streamlit run app/app.py`)
- **Remediation if Failed**: N/A.

---

## Final Audit Sign-Off

- **Audit Result**: **ALL 13 ITEMS PASSED**
- **Codebase Health**: Production-grade, reproducible, modular, and academically sound.
