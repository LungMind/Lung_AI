"""Prediction script for single CT slice using trained Hybrid CNN-BiLSTM-Attention model."""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import tensorflow as tf

from src.config import (
    HYBRID_MODEL_PATH,
    CNN_BASELINE_MODEL_PATH,
    CLASS_NAMES,
    IMAGE_SIZE,
    GRADCAM_DIR,
    ensure_directories,
)
from src.attention import SequenceAttention
from src.preprocessing import load_and_preprocess_image
from src.explainability import generate_gradcam_heatmap, overlay_gradcam


def load_model(model_path=HYBRID_MODEL_PATH):
    """Load model with custom SequenceAttention layer registered."""
    path = Path(model_path)
    if not path.exists():
        # Fallback to baseline if hybrid does not exist
        if CNN_BASELINE_MODEL_PATH.exists():
            path = CNN_BASELINE_MODEL_PATH
        else:
            raise FileNotFoundError(f"Model checkpoint not found at: {path.resolve()}")

    model = tf.keras.models.load_model(
        str(path),
        custom_objects={"SequenceAttention": SequenceAttention},
    )
    return model, path.name


def predict_image(image_path, model_path=HYBRID_MODEL_PATH, save_gradcam=None):
    """Run inference on a single CT slice, print formatted probabilities, and optionally save Grad-CAM."""
    ensure_directories()
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Image not found at: {img_p.resolve()}")

    model, model_name = load_model(model_path)

    # Preprocess image
    preprocessed = load_and_preprocess_image(str(img_p), target_size=IMAGE_SIZE)
    input_tensor = np.expand_dims(preprocessed, axis=0)  # (1, 224, 224, 3)

    # Inference
    probabilities = model.predict(input_tensor, verbose=0)[0]
    predicted_idx = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_idx]

    # Required output format
    print(f"\nImage: {img_p.resolve()}")
    print(f"Model: {model_name}")
    print(f"Prediction: {predicted_class}")
    for idx, class_name in enumerate(CLASS_NAMES):
        print(f"{class_name}: {probabilities[idx]:.4f}")

    # Optional Grad-CAM generation
    if save_gradcam:
        cam_save_path = Path(save_gradcam)
        heatmap = generate_gradcam_heatmap(model, input_tensor, pred_index=predicted_idx)
        raw_bgr = cv2.imread(str(img_p))
        if raw_bgr is not None:
            raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            raw_resized = cv2.resize(raw_rgb, IMAGE_SIZE)
        else:
            raw_resized = np.uint8(255 * preprocessed)

        overlay = overlay_gradcam(raw_resized, heatmap, alpha=0.45)
        # Convert RGB to BGR for cv2.imwrite
        cv2.imwrite(str(cam_save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"\n[EXPLAINABILITY] Grad-CAM overlay saved to: {cam_save_path.resolve()}")

    return {
        "image": str(img_p),
        "prediction": predicted_class,
        "probabilities": {name: float(probabilities[i]) for i, name in enumerate(CLASS_NAMES)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict lung cancer class from single CT slice")
    parser.add_argument("--image", type=str, required=True, help="Path to CT image slice")
    parser.add_argument("--model", type=str, default=str(HYBRID_MODEL_PATH), help="Path to model checkpoint")
    parser.add_argument("--save-cam", type=str, default=None, help="Optional output path to save Grad-CAM heatmap")

    args = parser.parse_args()
    predict_image(args.image, model_path=args.model, save_gradcam=args.save_cam)
