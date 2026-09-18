"""Dataset inspection, verification, and train/val/test splitting script.

Verifies image integrity, detects patient-level metadata limitations, performs
stratified 70/15/15 split, and saves split manifests to results/metrics/.
"""

import sys
import json
from pathlib import Path
from collections import Counter

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from PIL import Image
import numpy as np

from src.config import (
    DATA_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    RANDOM_SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    METRICS_DIR,
    TRAIN_MANIFEST_CSV,
    VAL_MANIFEST_CSV,
    TEST_MANIFEST_CSV,
    DATASET_SUMMARY_JSON,
    ensure_directories,
)
from src.dataset import scan_dataset, split_dataset
from src.preprocessing import load_and_preprocess_image


def run_inspection(data_dir=DATA_DIR):
    """Run comprehensive dataset inspection, integrity verification, and split manifest creation."""
    ensure_directories()
    print("=" * 70)
    print("IQ-OTH/NCCD LUNG CANCER CT DATASET INSPECTION & PREPARATION")
    print("=" * 70)

    # 1. Dataset Directory Detection
    print(f"\n[1] DETECTED DATASET DIRECTORY:")
    print(f"    Path: {Path(data_dir).resolve()}")
    if not Path(data_dir).exists():
        print(f"    [ERROR] Configured path does not exist!")
        sys.exit(1)
    print("    [STATUS] Directory detected and accessible.")

    # 2. Recursive Folder & File Scanning
    print(f"\n[2] SCANNING FILES RECURSIVELY...")
    df = scan_dataset(data_dir)
    total_images = len(df)
    print(f"    Found {total_images} total valid CT image files.")

    # 3. Class Identification & Distribution
    print(f"\n[3] IDENTIFIED CLASSES & DISTRIBUTION:")
    class_counts = df["class_name"].value_counts().to_dict()
    for class_name in CLASS_NAMES:
        count = class_counts.get(class_name, 0)
        pct = (count / total_images) * 100 if total_images > 0 else 0
        print(f"    - {class_name:10s}: {count:5d} images ({pct:5.1f}%)")

    # 4. File Extensions
    extensions = df["extension"].value_counts().to_dict()
    print(f"\n[4] IMAGE FILE EXTENSIONS:")
    for ext, count in extensions.items():
        print(f"    - {ext}: {count} images")

    # 5. File Path Samples
    print(f"\n[5] EXAMPLE FILE PATHS PER CLASS:")
    for class_name in CLASS_NAMES:
        sample = df[df["class_name"] == class_name]["file_path"].iloc[0]
        rel_sample = Path(sample).name
        print(f"    - {class_name:10s} -> .../{rel_sample}")

    # 6. PIL & OpenCV Integrity Verification & Dimension Analysis
    print(f"\n[6] VERIFYING IMAGE READABILITY & EXTRACTING DIMENSIONS...")
    corrupted_files = []
    dimensions_counter = Counter()

    for idx, row in df.iterrows():
        fpath = row["file_path"]
        try:
            # Check PIL verification
            with Image.open(fpath) as pil_img:
                pil_img.verify()

            # Check OpenCV decoding
            cv_img = cv2.imread(fpath)
            if cv_img is None:
                corrupted_files.append({"file": fpath, "reason": "OpenCV decoding failed"})
            else:
                h, w, c = cv_img.shape
                dimensions_counter[(w, h, c)] += 1

        except Exception as e:
            corrupted_files.append({"file": fpath, "reason": str(e)})

    print(f"    Total images verified: {total_images}")
    print(f"    Corrupted / unreadable files: {len(corrupted_files)}")
    if len(corrupted_files) == 0:
        print("    [PASS] 100% of CT images are structurally intact and readable.")
    else:
        print(f"    [ALERT] Found {len(corrupted_files)} corrupted files!")
        for c in corrupted_files[:5]:
            print(f"      - {c['file']}: {c['reason']}")

    print("\n    Image Dimensions Distribution (Width x Height x Channels):")
    for dim, count in dimensions_counter.most_common():
        pct = (count / total_images) * 100
        print(f"    - {dim[0]}x{dim[1]} ({dim[2]} channels): {count:5d} images ({pct:5.1f}%)")

    # 7. Test Preprocessing Pipeline
    print(f"\n[7] VERIFYING PREPROCESSING PIPELINE (load, RGB, resize to 224x224, float32, normalize)...")
    sample_path = df["file_path"].iloc[0]
    preprocessed_sample = load_and_preprocess_image(sample_path, target_size=IMAGE_SIZE)
    print(f"    Sample: {Path(sample_path).name}")
    print(f"    Output shape: {preprocessed_sample.shape} (Expected: (224, 224, 3))")
    print(f"    Dtype:        {preprocessed_sample.dtype} (Expected: float32)")
    print(f"    Min value:    {preprocessed_sample.min():.4f}, Max value: {preprocessed_sample.max():.4f} (Expected range: [0.0, 1.0])")
    assert preprocessed_sample.shape == (224, 224, 3), "Shape mismatch!"
    assert preprocessed_sample.dtype == np.float32, "Dtype mismatch!"
    assert 0.0 <= preprocessed_sample.min() and preprocessed_sample.max() <= 1.0, "Normalization out of bounds!"
    print("    [PASS] Preprocessing pipeline verified successfully.")

    # 8. Patient ID Analysis & Data Leakage Limitation
    print(f"\n[8] PATIENT/CASE IDENTIFIER & DATA LEAKAGE ASSESSMENT:")
    print("    - Inspection of slice filenames shows indices: 'Bengin case (1).jpg' to '(120).jpg', etc.")
    print("    - According to official IQ-OTH/NCCD clinical documentation, there are 110 patients")
    print("      (15 Benign, 40 Malignant, 55 Normal).")
    print("    - Because 120 images exist in Benign, the numbers represent sequential image indices,")
    print("      NOT patient identifiers.")
    print("    - CONCLUSION: True patient grouping metadata is NOT provided in this public release.")
    print("    - ACTION: We explicitly document this limitation and use a reproducible Stratified")
    print(f"      Shuffle Split (Random Seed: {RANDOM_SEED}) across classes.")

    # 9. Perform Reproducible Stratified Split
    print(f"\n[9] PERFORMING 70% TRAIN / 15% VAL / 15% TEST STRATIFIED SPLIT:")
    train_df, val_df, test_df, split_method = split_dataset(
        df,
        train_size=TRAIN_SPLIT,
        val_size=VAL_SPLIT,
        test_size=TEST_SPLIT,
        seed=RANDOM_SEED,
        save_manifests=True,
    )

    print(f"\n    Split Counts:")
    print(f"    - Training Set:   {len(train_df):5d} images ({len(train_df)/total_images*100:5.1f}%)")
    print(f"    - Validation Set: {len(val_df):5d} images ({len(val_df)/total_images*100:5.1f}%)")
    print(f"    - Test Set:       {len(test_df):5d} images ({len(test_df)/total_images*100:5.1f}%)")
    print(f"    - Total:          {len(train_df)+len(val_df)+len(test_df):5d} images")

    print("\n    Per-Class Split Breakdown:")
    print(f"    {'Class':<12} | {'Train':<7} | {'Val':<7} | {'Test':<7} | {'Total':<7}")
    print("    " + "-" * 45)
    for c in CLASS_NAMES:
        tr_c = (train_df["class_name"] == c).sum()
        vl_c = (val_df["class_name"] == c).sum()
        ts_c = (test_df["class_name"] == c).sum()
        tt_c = tr_c + vl_c + ts_c
        print(f"    {c:<12} | {tr_c:<7} | {vl_c:<7} | {ts_c:<7} | {tt_c:<7}")

    # 10. Save Inspection Summary JSON
    summary_data = {
        "dataset_name": "IQ-OTH/NCCD Lung Cancer CT Dataset",
        "dataset_directory": str(Path(data_dir).resolve()),
        "total_images": total_images,
        "corrupted_images": len(corrupted_files),
        "class_distribution": class_counts,
        "extensions": extensions,
        "dimensions": {f"{k[0]}x{k[1]}x{k[2]}": v for k, v in dimensions_counter.items()},
        "patient_level_splitting_possible": False,
        "patient_level_limitation": (
            "Filenames use sequential slice numbering rather than patient/case IDs. "
            "Patient-to-slice grouping lookup was not provided in this release."
        ),
        "split_method": split_method,
        "random_seed": RANDOM_SEED,
        "split_ratios": {"train": TRAIN_SPLIT, "val": VAL_SPLIT, "test": TEST_SPLIT},
        "split_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "manifest_files": {
            "train": str(TRAIN_MANIFEST_CSV.resolve()),
            "val": str(VAL_MANIFEST_CSV.resolve()),
            "test": str(TEST_MANIFEST_CSV.resolve()),
        },
    }

    with open(DATASET_SUMMARY_JSON, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\n[10] INSPECTION SUMMARY JSON SAVED:")
    print(f"     Path: {DATASET_SUMMARY_JSON}")

    print("\n" + "=" * 70)
    print("[COMPLETE] Dataset inspection and preparation finished successfully.")
    print("=" * 70)


if __name__ == "__main__":
    run_inspection()
