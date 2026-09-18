"""Dataset discovery, inspection, stratified splitting, and tf.data loading pipelines."""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    DATA_DIR,
    CLASS_NAMES,
    CLASS_TO_IDX,
    IMAGE_SIZE,
    BATCH_SIZE,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    RANDOM_SEED,
    TRAIN_MANIFEST_CSV,
    VAL_MANIFEST_CSV,
    TEST_MANIFEST_CSV,
    ensure_directories,
)
from src.preprocessing import load_and_preprocess_image, get_data_augmentation

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def normalize_class_name(name):
    """Normalize folder names to canonical class: Normal, Benign, Malignant."""
    cleaned = name.lower().replace("_", " ").strip()
    if "normal" in cleaned:
        return "Normal"
    if "benign" in cleaned or "bengin" in cleaned:
        return "Benign"
    if "malignant" in cleaned or "cancer" in cleaned:
        return "Malignant"
    return None


def inspect_patient_identifiers(df):
    """Determine whether filenames contain true patient/case grouping identifiers."""
    sample_names = df["file_name"].head(10).tolist()
    genuine_group_pattern = re.compile(r"(patient|pt|p)[\s_\-]*\d+[\s_\-]*(slice|scan)", re.IGNORECASE)
    matches = [name for name in df["file_name"] if genuine_group_pattern.search(name)]
    return len(matches) > (0.5 * len(df))


def scan_dataset(data_dir=DATA_DIR):
    """Scan dataset recursively, extract metadata, and identify classes."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_path.resolve()}")

    records = []
    for file_path in data_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
            canonical_class = None
            for parent in [file_path.parent.name, file_path.parent.parent.name]:
                norm = normalize_class_name(parent)
                if norm is not None:
                    canonical_class = norm
                    break

            if canonical_class is None:
                canonical_class = normalize_class_name(file_path.stem)

            if canonical_class is not None:
                records.append(
                    {
                        "file_path": str(file_path.resolve()),
                        "file_name": file_path.name,
                        "class_name": canonical_class,
                        "label": CLASS_TO_IDX[canonical_class],
                        "extension": file_path.suffix.lower(),
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(
            f"No valid CT images found in {data_path.resolve()}. "
            f"Expected subdirectories for {CLASS_NAMES}."
        )

    df = df.sort_values(by=["class_name", "file_name"]).reset_index(drop=True)
    return df


def split_dataset(
    df,
    train_size=TRAIN_SPLIT,
    val_size=VAL_SPLIT,
    test_size=TEST_SPLIT,
    seed=RANDOM_SEED,
    save_manifests=True,
):
    """Split dataset into 70% train, 15% validation, and 15% test."""
    ensure_directories()
    total = train_size + val_size + test_size
    train_ratio = train_size / total
    val_ratio = val_size / (val_size + test_size)

    has_patient_groups = inspect_patient_identifiers(df)

    if has_patient_groups:
        print("[SPLIT] Patient/case groups detected. Performing GroupShuffleSplit...")
        gss_train = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
        train_idx, temp_idx = next(gss_train.split(df, groups=df["patient_id"]))
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)

        gss_val = GroupShuffleSplit(n_splits=1, train_size=val_ratio, random_state=seed)
        val_idx, test_idx = next(gss_val.split(temp_df, groups=temp_df["patient_id"]))
        val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
        test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)
        split_method = "Patient-Level GroupShuffleSplit"
    else:
        print("[SPLIT] Using reproducible StratifiedShuffleSplit (70% Train, 15% Val, 15% Test)...")
        sss_train = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio, random_state=seed)
        train_idx, temp_idx = next(sss_train.split(df, df["label"]))
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        temp_df = df.iloc[temp_idx].copy().reset_index(drop=True)

        sss_val = StratifiedShuffleSplit(n_splits=1, train_size=val_ratio, random_state=seed)
        val_idx, test_idx = next(sss_val.split(temp_df, temp_df["label"]))
        val_df = temp_df.iloc[val_idx].copy().reset_index(drop=True)
        test_df = temp_df.iloc[test_idx].copy().reset_index(drop=True)
        split_method = "StratifiedShuffleSplit (Slice-Level)"

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    if save_manifests:
        train_df.to_csv(TRAIN_MANIFEST_CSV, index=False)
        val_df.to_csv(VAL_MANIFEST_CSV, index=False)
        test_df.to_csv(TEST_MANIFEST_CSV, index=False)

    return train_df, val_df, test_df, split_method


def load_or_create_splits(data_dir=DATA_DIR, seed=RANDOM_SEED):
    """Load existing manifest CSVs if present; otherwise scan and split."""
    if TRAIN_MANIFEST_CSV.exists() and VAL_MANIFEST_CSV.exists() and TEST_MANIFEST_CSV.exists():
        print(f"[DATA] Loading existing split manifests from {TRAIN_MANIFEST_CSV.parent}...")
        train_df = pd.read_csv(TRAIN_MANIFEST_CSV)
        val_df = pd.read_csv(VAL_MANIFEST_CSV)
        test_df = pd.read_csv(TEST_MANIFEST_CSV)
    else:
        print("[DATA] Manifests not found. Scanning dataset and creating splits...")
        df = scan_dataset(data_dir)
        train_df, val_df, test_df, _ = split_dataset(df, seed=seed)

    return train_df, val_df, test_df


def get_balanced_class_weights(train_df):
    """Compute balanced class weights using sklearn.utils.class_weight.compute_class_weight."""
    unique_classes = np.sort(train_df["label"].unique())
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=train_df["label"].values,
    )
    class_weights_dict = {cls: float(w) for cls, w in zip(unique_classes, weights)}
    return class_weights_dict


def create_tf_dataset(df, is_training=False, batch_size=BATCH_SIZE, use_sparse=True):
    """Build a tf.data.Dataset pipeline.

    - Loads and normalizes images using load_and_preprocess_image
    - Supports sparse integer labels (for SparseCategoricalCrossentropy) or one-hot vectors
    - Applies augmentation only if is_training=True
    - Validation and test datasets are NEVER augmented
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required to build tf.data.Dataset pipeline.")

    file_paths = df["file_path"].values
    labels = df["label"].values

    def _load_np(path_bytes):
        path_str = path_bytes.decode("utf-8")
        return load_and_preprocess_image(path_str, target_size=IMAGE_SIZE)

    def _map_fn(path, lbl):
        img = tf.numpy_function(_load_np, [path], tf.float32)
        img.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
        if use_sparse:
            label_out = tf.cast(lbl, tf.int32)
        else:
            label_out = tf.one_hot(lbl, depth=len(CLASS_NAMES))
        return img, label_out

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if is_training:
        ds = ds.shuffle(buffer_size=len(df), seed=RANDOM_SEED)

    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)

    # Augmentation applied strictly for training data
    if is_training:
        aug = get_data_augmentation()
        if aug is not None:
            ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def get_data_loaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE, use_sparse=True):
    """High-level function to load splits and return tf.data loaders."""
    train_df, val_df, test_df = load_or_create_splits(data_dir=data_dir)

    train_ds = create_tf_dataset(train_df, is_training=True, batch_size=batch_size, use_sparse=use_sparse)
    val_ds = create_tf_dataset(val_df, is_training=False, batch_size=batch_size, use_sparse=use_sparse)
    test_ds = create_tf_dataset(test_df, is_training=False, batch_size=batch_size, use_sparse=use_sparse)

    return train_ds, val_ds, test_ds, train_df, val_df, test_df
