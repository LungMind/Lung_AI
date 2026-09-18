# Dataset Instructions

## 1. Primary Dataset: IQ-OTH/NCCD
This project uses the **IQ-OTH/NCCD (Iraq-Oncology Teaching Hospital/National Center for Cancer Diseases)** lung cancer CT dataset for 3-class classification:
1. `Normal` (Healthy lung tissue)
2. `Benign` (Non-cancerous lesions / nodules)
3. `Malignant` (Cancerous tumors / lesions)

## 2. Expected Directory Structure
Place the extracted dataset inside this `data/` folder (or configure `DATA_DIR` in `src/config.py`):

```text
data/
└── lung_cancer/
    ├── Normal/             (or "Normal cases")
    │   ├── normal_001.png
    │   └── ...
    ├── Benign/             (or "Benign cases")
    │   ├── benign_001.png
    │   └── ...
    └── Malignant/          (or "Malignant cases")
        ├── malignant_001.png
        └── ...
```

> **Robust Loader Support**: The dataset loader in `src/dataset.py` automatically normalizes folder names, handling variations like `Normal cases`, `Bengin cases`, `Benign`, `Malignant`, etc.

## 3. Data Leakage Prevention (Slice vs Patient Splitting)
- **Clinical Rule**: Multiple CT slices originating from the same patient must **never** be split across both training and test sets. Doing so leads to severe data leakage and artificially inflated accuracy.
- **Identifier Handling**:
  - If patient/case IDs are present in filenames (e.g. `case_102_slice_3.jpg`), the dataset pipeline groups slices by patient ID using `GroupShuffleSplit`.
  - If patient IDs are not explicitly provided by the downloaded dataset release, the loader documents this limitation and falls back to stratified splitting with a recorded warning.

## 4. Datasets NOT Used in This Project
- **COVIDx CT**: Reserved strictly for COVID-19 pulmonary imaging; **excluded** from this lung cancer project.
- **MIMIC-IV**: Access is currently under PhysioNet credentialing review. It will be incorporated in a future phase for multi-modal clinical notes integration.
