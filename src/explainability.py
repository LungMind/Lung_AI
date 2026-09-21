"""Grad-CAM (Gradient-weighted Class Activation Mapping) for CNN and Hybrid Models.

Technical Implementation Note:
-----------------------------
In the Hybrid CNN-BiLSTM-Attention architecture, gradients flow backward from the
final class logit/probability through:
1. Softmax and Dense Classifier layers
2. The Custom SequenceAttention Layer (differentiable weighted sum)
3. The Bidirectional LSTM (differentiable recurrent sequence)
4. The Reshape Layer
5. Directly into the last convolutional feature map (last_conv_layer).

Because all operations in the hybrid model are fully differentiable under TensorFlow's
autodiff engine, Grad-CAM accurately computes channel-wise gradient importance weights
alpha_k = mean(d y_c / d A_k) over the convolutional feature map, producing true visual
heatmaps of the salient lung CT regions.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.config import GRADCAM_DIR, CLASS_NAMES, IMAGE_SIZE, ensure_directories
from src.attention import SequenceAttention


def find_target_conv_layer(model):
    """Locate the target convolutional layer for Grad-CAM inspection."""
    if hasattr(model, "last_conv_name") and model.last_conv_name:
        try:
            return model.get_layer(model.last_conv_name)
        except ValueError:
            pass

    # Search backwards for last Conv2D layer
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("No Conv2D layer found in the model for Grad-CAM computation.")


def get_grad_model(model, target_conv_layer=None):
    """Construct or return reusable multi-output model for Grad-CAM."""
    if target_conv_layer is None:
        target_conv_layer = find_target_conv_layer(model)
    return tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[target_conv_layer.output, model.output],
    )


def generate_gradcam_heatmap(model, img_array, pred_index=None, target_conv_layer=None, grad_model=None):
    """Compute Grad-CAM activation heatmap for a single preprocessed CT slice.

    Args:
        model: Trained Keras model (Baseline CNN or Hybrid).
        img_array: Preprocessed numpy array of shape (1, 224, 224, 3).
        pred_index: Target class index to explain (None = model top prediction).
        target_conv_layer: Specific Conv2D layer (None = automatically detected).
        grad_model: Optional pre-built Grad-CAM model to avoid graph recreation overhead.

    Returns:
        np.ndarray: 2D heatmap normalized to [0, 1] of shape (224, 224).
    """
    if grad_model is None:
        grad_model = get_grad_model(model, target_conv_layer)


    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        target_score = predictions[:, pred_index]

    # Compute gradient of target class score with respect to convolutional feature maps
    grads = tape.gradient(target_score, conv_outputs)

    # Global average pooling of gradients gives importance weight per feature channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight convolutional feature channels by pooled gradients
    conv_outputs_single = conv_outputs[0]
    heatmap = conv_outputs_single @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU: only positive influence is considered
    heatmap = tf.maximum(heatmap, 0.0)

    # Normalize between 0.0 and 1.0
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    heatmap_np = heatmap.numpy()

    # Resize heatmap to match input CT image dimensions
    heatmap_resized = cv2.resize(heatmap_np, (img_array.shape[2], img_array.shape[1]))
    return heatmap_resized


def overlay_gradcam(original_img_rgb, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """Blend the Grad-CAM heatmap over the original CT scan."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    colored_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

    if original_img_rgb.dtype != np.uint8:
        if original_img_rgb.max() <= 1.0:
            original_img_rgb = np.uint8(255 * original_img_rgb)
        else:
            original_img_rgb = np.uint8(original_img_rgb)

    blended = cv2.addWeighted(original_img_rgb, 1 - alpha, colored_rgb, alpha, 0)
    return blended


def explain_prediction(model, preprocessed_img, original_img=None, save_filename=None):
    """High-level function to run prediction, Grad-CAM heatmap, and overlay."""
    ensure_directories()
    preds = model.predict(preprocessed_img, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = CLASS_NAMES[pred_idx]
    pred_confidence = float(preds[pred_idx])

    heatmap = generate_gradcam_heatmap(model, preprocessed_img, pred_index=pred_idx)

    if original_img is None:
        base_img = np.uint8(255 * preprocessed_img[0])
    else:
        base_img = cv2.resize(original_img, IMAGE_SIZE)
        if base_img.dtype != np.uint8:
            base_img = np.uint8(255 * base_img)

    overlay = overlay_gradcam(base_img, heatmap)

    if save_filename:
        save_path = GRADCAM_DIR / save_filename
        plt.imsave(str(save_path), overlay)
        print(f"[EXPLAINABILITY] Grad-CAM saved to: {save_path}")

    return {
        "predicted_class": pred_class,
        "confidence": pred_confidence,
        "probabilities": {name: float(preds[i]) for i, name in enumerate(CLASS_NAMES)},
        "heatmap": heatmap,
        "overlay": overlay,
    }
