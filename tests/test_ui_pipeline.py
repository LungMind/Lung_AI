"""Automated test suite verifying the Streamlit UI pipeline, model inference, and Grad-CAM."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import io
import numpy as np
import pandas as pd
from PIL import Image

from app.app import load_cached_model
from src.config import TEST_MANIFEST_CSV, IMAGE_SIZE, CLASS_NAMES
from src.preprocessing import preprocess_pil_image
from src.explainability import generate_gradcam_heatmap, overlay_gradcam


def run_test_suite():
    print("=" * 70)
    print("RUNNING STREAMLIT UI PIPELINE VALIDATION TEST SUITE")
    print("=" * 70)

    # 1. Test model loading
    print("\n[TEST 1] Loading cached model...")
    model, grad_model = load_cached_model()
    assert model is not None, "Failed to load cached hybrid model!"
    assert grad_model is not None, "Failed to load Grad-CAM model!"
    print(f"  [PASS] Model loaded successfully: Input shape {model.input_shape}, Output shape {model.output_shape}")

    # 2. Test predictions on actual test set images for all 3 classes
    test_df = pd.read_csv(TEST_MANIFEST_CSV)
    classes_to_test = ["Normal", "Benign", "Malignant"]

    for cname in classes_to_test:
        sample_row = test_df[test_df["class_name"] == cname].iloc[0]
        fpath = sample_row["file_path"]
        print(f"\n[TEST 2] Testing {cname} CT image: {Path(fpath).name}...")

        # Load with PIL
        pil_img = Image.open(fpath)
        assert pil_img is not None, f"Could not load image: {fpath}"

        # Preprocess
        tensor = preprocess_pil_image(pil_img, target_size=IMAGE_SIZE)
        assert tensor.shape == (1, 224, 224, 3), f"Invalid preprocessed shape: {tensor.shape}"

        # Inference
        probs = model.predict(tensor, verbose=0)[0]
        prob_sum = float(np.sum(probs))
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        conf = float(probs[pred_idx]) * 100.0

        print(f"  True Class:       {cname}")
        print(f"  Predicted Class:  {pred_class}")
        print(f"  Confidence:       {conf:.2f}%")
        print(f"  Probabilities:    Normal={probs[0]*100:.1f}%, Benign={probs[1]*100:.1f}%, Malignant={probs[2]*100:.1f}%")
        print(f"  Probability Sum:  {prob_sum:.4f}")

        assert np.isclose(prob_sum, 1.0, atol=1e-4), f"Probabilities do not sum to 1.0! Sum: {prob_sum}"
        print("  [PASS] Probabilities calibrated and sum to ~100%.")

        # Grad-CAM
        print("  Generating Grad-CAM attribution...")
        heatmap = generate_gradcam_heatmap(model, tensor, pred_index=pred_idx)
        assert heatmap.shape == (224, 224), f"Invalid heatmap shape: {heatmap.shape}"
        assert not np.isnan(heatmap).any(), "Heatmap contains NaN!"

        rgb_np = np.array(pil_img.convert("RGB"))
        resized_rgb = Image.fromarray(rgb_np).resize(IMAGE_SIZE)
        overlay = overlay_gradcam(np.array(resized_rgb), heatmap, alpha=0.45)
        assert overlay.shape == (224, 224, 3), f"Invalid overlay shape: {overlay.shape}"
        print("  [PASS] Grad-CAM heatmap and overlay generated cleanly.")

    # 3. Test WebP format support
    print("\n[TEST 3] Testing WebP image format compatibility...")
    synthetic_arr = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    webp_buf = io.BytesIO()
    Image.fromarray(synthetic_arr).save(webp_buf, format="WEBP")
    webp_buf.seek(0)

    webp_img = Image.open(webp_buf)
    webp_tensor = preprocess_pil_image(webp_img, target_size=IMAGE_SIZE)
    webp_probs = model.predict(webp_tensor, verbose=0)[0]
    assert len(webp_probs) == 3, "WebP prediction failed!"
    print(f"  [PASS] WebP successfully ingested and predicted. Top class: {CLASS_NAMES[np.argmax(webp_probs)]}")

    # 4. Test corrupted / invalid image handling
    print("\n[TEST 4] Testing invalid image handling...")
    bad_bytes = b"not an image file content"
    try:
        corrupted_img = Image.open(io.BytesIO(bad_bytes))
        corrupted_img.verify()
        passed_invalid = False
    except Exception as e:
        passed_invalid = True
        print(f"  [PASS] Correctly caught corrupted file exception: {type(e).__name__}")
    assert passed_invalid, "Failed to reject corrupted image!"

    print("\n" + "=" * 70)
    print("ALL TEST SUITE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_test_suite()
