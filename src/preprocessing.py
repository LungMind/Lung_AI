"""Image loading, conversion to RGB, resizing to 224x224, and float32 normalization."""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from src.config import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_SIZE


def load_and_preprocess_image(image_path, target_size=IMAGE_SIZE):
    """Load a CT slice, convert to RGB, resize to 224x224, and normalize pixel values to [0.0, 1.0].

    Steps:
        1. Reads image from disk using OpenCV (or PIL fallback)
        2. Converts color space to 3-channel RGB
        3. Resizes to target dimension (224, 224)
        4. Converts dtype to float32
        5. Normalizes pixel intensities to [0.0, 1.0]
    """
    path_str = str(image_path)
    image = cv2.imread(path_str)

    if image is None:
        with Image.open(path_str) as pil_img:
            rgb_img = pil_img.convert("RGB")
            image = np.array(rgb_img)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return image


def preprocess_pil_image(pil_img, target_size=IMAGE_SIZE):
    """Preprocess an in-memory PIL image (e.g. from file upload)."""
    rgb_img = pil_img.convert("RGB")
    image_np = np.array(rgb_img)
    resized = cv2.resize(image_np, target_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def get_data_augmentation():
    """Create data augmentation pipeline applied strictly to training data only.

    Includes:
    - Small rotation
    - Small zoom
    - Horizontal flip (medically valid for bilateral lung symmetry)
    - Small translation

    Validation and test sets MUST NOT be augmented.
    """
    try:
        import tensorflow as tf

        return tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.04),
                tf.keras.layers.RandomZoom(height_factor=(-0.04, 0.04)),
                tf.keras.layers.RandomTranslation(height_factor=0.04, width_factor=0.04),
            ],
            name="training_data_augmentation",
        )
    except ImportError:
        return None
