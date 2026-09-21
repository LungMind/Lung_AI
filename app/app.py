"""LUNG AI - Streamlit Clinical AI Research Dashboard.

Hybrid CNN-BiLSTM-Attention Framework for Lung Cancer Classification from CT Images.
Evaluated on the IQ-OTH/NCCD dataset (Normal, Benign, Malignant).
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

from src.config import (
    HYBRID_MODEL_PATH,
    CLASS_NAMES,
    IMAGE_SIZE,
    HYBRID_METRICS_JSON,
)
from src.attention import SequenceAttention
from src.preprocessing import preprocess_pil_image
from src.explainability import generate_gradcam_heatmap, overlay_gradcam, get_grad_model

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="LUNG AI | Hybrid CNN-BiLSTM-Attention",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. Futuristic Dark Medical AI Theme (Custom CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global dark canvas */
    .stApp {
        background-color: #070b14;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Top header styling */
    .header-container {
        padding: 1.2rem 0 0.8rem 0;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 1rem;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #f8fafc;
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
    }
    .header-badge {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 700;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        background: rgba(6, 182, 212, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        vertical-align: middle;
        margin-left: 0.6rem;
    }
    .header-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-top: 0.2rem;
        font-weight: 500;
    }
    .header-tags {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 0.35rem;
        letter-spacing: 0.02em;
    }

    /* Medical disclaimer */
    .disclaimer-banner {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #334155;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 0.65rem 1rem;
        margin-bottom: 1.4rem;
        font-size: 0.85rem;
        line-height: 1.45;
        color: #cbd5e1;
    }
    .disclaimer-banner strong {
        color: #fbbf24;
    }

    /* UI Cards */
    .dashboard-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Metadata pills */
    .meta-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.6rem;
    }
    .meta-pill {
        background: #1e293b;
        color: #94a3b8;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        border: 1px solid #334155;
    }

    /* Empty state card */
    .empty-state {
        background: #0f172a;
        border: 1px dashed #334155;
        border-radius: 10px;
        padding: 3rem 1.5rem;
        text-align: center;
        color: #64748b;
    }
    .empty-state-icon {
        font-size: 2.4rem;
        margin-bottom: 0.6rem;
        opacity: 0.7;
    }
    .empty-state-text {
        font-size: 0.95rem;
        color: #94a3b8;
        font-weight: 500;
    }
    .empty-state-subtext {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Prediction hero card */
    .pred-hero {
        border-radius: 10px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
        text-align: center;
    }
    .pred-hero-normal {
        background: linear-gradient(180deg, rgba(16, 185, 129, 0.12) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .pred-hero-benign {
        background: linear-gradient(180deg, rgba(245, 158, 11, 0.12) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .pred-hero-malignant {
        background: linear-gradient(180deg, rgba(239, 68, 68, 0.14) 0%, rgba(15, 23, 42, 0.7) 100%);
        border: 1px solid rgba(239, 68, 68, 0.45);
    }

    .pred-hero-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        font-weight: 600;
    }
    .pred-hero-value {
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: 0.02em;
        margin: 0.2rem 0;
    }
    .pred-hero-val-normal { color: #10b981; }
    .pred-hero-val-benign { color: #f59e0b; }
    .pred-hero-val-malignant { color: #ef4444; }

    .pred-hero-conf {
        font-size: 1rem;
        font-weight: 600;
        color: #e2e8f0;
    }

    /* Probability meter cards */
    .prob-row {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .prob-row-selected {
        border-width: 1.5px;
    }
    .prob-row-normal.prob-row-selected {
        border-color: #10b981;
        background: rgba(16, 185, 129, 0.08);
    }
    .prob-row-benign.prob-row-selected {
        border-color: #f59e0b;
        background: rgba(245, 158, 11, 0.08);
    }
    .prob-row-malignant.prob-row-selected {
        border-color: #ef4444;
        background: rgba(239, 68, 68, 0.08);
    }

    .prob-name {
        font-size: 0.9rem;
        font-weight: 600;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .prob-pct {
        font-size: 0.92rem;
        font-weight: 700;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }

    /* Summary table */
    .summary-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.6rem;
        margin-top: 0.6rem;
    }
    .summary-item {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 0.55rem 0.75rem;
    }
    .summary-key {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
    }
    .summary-val {
        font-size: 0.9rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 0.15rem;
    }

    /* Sidebar tweaks */
    [data-testid="stSidebar"] {
        background-color: #0b1120;
        border-right: 1px solid #1e293b;
    }
    .sidebar-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #f8fafc;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sidebar-section {
        margin-top: 1.2rem;
        padding-top: 1rem;
        border-top: 1px solid #1e293b;
    }
    .sidebar-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .sidebar-value {
        font-size: 0.85rem;
        color: #cbd5e1;
        font-weight: 500;
        line-height: 1.4;
    }

    /* Hide Streamlit default branding watermark */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 3. Cached Model & Metadata Loading
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="⚡ Initializing LUNG AI Neural Engine...")
def load_cached_model():
    """Load the trained Hybrid Keras model once into cache with pre-warmed execution graphs."""
    if not HYBRID_MODEL_PATH.exists():
        return None, None
    try:
        model = tf.keras.models.load_model(
            str(HYBRID_MODEL_PATH),
            custom_objects={"SequenceAttention": SequenceAttention},
            compile=False,
        )
        grad_model = get_grad_model(model)
        # Pre-warm execution graphs with a dummy tensor so first user query is instant
        dummy = tf.zeros((1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=tf.float32)
        _ = model(dummy, training=False)
        with tf.GradientTape() as tape:
            c_out, preds = grad_model(dummy)
            _ = tape.gradient(preds[:, 0], c_out)

        return model, grad_model
    except Exception as e:
        st.session_state["model_load_error"] = str(e)
        return None, None


@st.cache_data(show_spinner=False)
def load_metrics_metadata():
    """Load measured model performance metrics from metrics JSON if present."""
    if HYBRID_METRICS_JSON.exists():
        try:
            with open(HYBRID_METRICS_JSON, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


model, grad_model = load_cached_model()
metrics_data = load_metrics_metadata()


# -----------------------------------------------------------------------------
# 4. Sidebar Navigation & Metadata
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-title">
            <span>🫁</span> LUNG AI
        </div>
        <div style="font-size: 0.8rem; color: #38bdf8; font-weight: 600; margin-top: 0.2rem;">
            Research Prototype Dashboard
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_choice = st.radio(
        "Navigation",
        options=["CT Analysis", "Model Architecture", "About Project"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">IQ-OTH/NCCD Thoracic CT Cohort</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label" style="margin-top: 0.7rem;">Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">CNN + BiLSTM + Attention</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label" style="margin-top: 0.7rem;">Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">3 Classes (Normal, Benign, Malignant)</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label" style="margin-top: 0.7rem;">Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-value">Grad-CAM (Conv Feature Attribution)</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if metrics_data:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-label">Verified Test Benchmarks</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size: 0.8rem; line-height: 1.6; color: #94a3b8;">
                • Test Accuracy: <strong style="color: #f8fafc;">{metrics_data.get('accuracy', 0)*100:.2f}%</strong><br>
                • Macro ROC-AUC: <strong style="color: #f8fafc;">{metrics_data.get('roc_auc_macro', 0):.4f}</strong><br>
                • Macro F1-Score: <strong style="color: #f8fafc;">{metrics_data.get('f1_score_macro', 0):.4f}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="margin-top: 2rem; padding: 0.6rem 0.8rem; background: #070b14; border-radius: 6px; border: 1px solid #1e293b; font-size: 0.72rem; color: #64748b; line-height: 1.4;">
            ⚖️ <strong>Educational & Research Use Only</strong><br>
            Not certified for medical diagnosis or clinical decision making.
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 5. Top Header & Medical Disclaimer
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="header-container">
        <div class="header-title">
            LUNG AI <span class="header-badge">Research Prototype</span>
        </div>
        <div class="header-subtitle">
            Hybrid CNN-BiLSTM-Attention Lung Cancer Classification
        </div>
        <div class="header-tags">
            CT Image Classification &bull; 3 Classes (Normal, Benign, Malignant) &bull; Grad-CAM Explainability
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer-banner">
        <strong>⚠️ Medical & Regulatory Notice:</strong>
        Educational and research use only. This system is not a medical diagnostic tool and should not replace professional medical advice, radiological assessment, or clinical judgment.
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# 6. Main View Routing
# -----------------------------------------------------------------------------

# =============================================================================
# VIEW A: CT Analysis (Main Workflow)
# =============================================================================
if nav_choice == "CT Analysis":
    # Model Availability Alert
    if model is None:
        st.warning(
            "⚠️ **Trained Model Checkpoint Not Detected**\n\n"
            "The model file `models/hybrid_cnn_bilstm_attention.keras` is not available in the workspace. "
            "Please run the training pipeline to generate the model checkpoint."
        )

    # Main Two-Column Layout
    col_upload, col_prediction = st.columns([1, 1], gap="large")

    # -------------------------------------------------------------------------
    # LEFT COLUMN: Upload CT Scan
    # -------------------------------------------------------------------------
    with col_upload:
        st.markdown(
            """
            <div class="card-title">
                <span>📁</span> Upload CT Scan
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload an axial chest CT slice",
            type=["png", "jpg", "jpeg", "bmp", "webp", "tiff", "tif"],
            help="Supported formats: PNG, JPG, JPEG, BMP, WEBP, TIFF.",
        )

        pil_image = None
        if uploaded_file is not None:
            try:
                pil_image = Image.open(uploaded_file)
                # Verify image can be parsed
                pil_image.verify()
                # Re-open after verify() closes it
                uploaded_file.seek(0)
                pil_image = Image.open(uploaded_file)

                # Modern Streamlit API: use_container_width replaces deprecated use_column_width
                st.image(
                    pil_image,
                    caption=f"Uploaded Slice: {uploaded_file.name}",
                    use_container_width=True,
                )

                # Image metadata pills
                file_size_kb = uploaded_file.size / 1024.0 if hasattr(uploaded_file, "size") else 0
                st.markdown(
                    f"""
                    <div class="meta-row">
                        <span class="meta-pill">Dims: {pil_image.size[0]} &times; {pil_image.size[1]}</span>
                        <span class="meta-pill">Mode: {pil_image.mode}</span>
                        <span class="meta-pill">Format: {pil_image.format or 'Image'}</span>
                        <span class="meta-pill">Size: {file_size_kb:.1f} KB</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
                predict_clicked = st.button(
                    "🔍 Analyze CT Scan",
                    type="primary",
                    use_container_width=True,
                    disabled=(model is None),
                )

                if predict_clicked:
                    st.session_state["active_file"] = uploaded_file.name

            except Exception as e:
                st.error(
                    "Unable to read the uploaded file as a valid image. "
                    "Please ensure the file is an uncorrupted axial CT slice in a supported format."
                )
                pil_image = None
        else:
            # Clear stored prediction and Grad-CAM state when file is removed
            st.session_state.pop("prediction_result", None)
            st.session_state.pop("active_file", None)
            st.session_state.pop("gradcam_result", None)
            st.session_state.pop("gradcam_for_file", None)

            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-icon">🩻</div>
                    <div class="empty-state-text">No CT scan selected</div>
                    <div class="empty-state-subtext">Drag and drop an axial thoracic CT slice to begin analysis.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # RIGHT COLUMN: AI Prediction
    # -------------------------------------------------------------------------
    with col_prediction:
        st.markdown(
            """
            <div class="card-title">
                <span>⚡</span> AI Prediction
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Trigger inference if user clicked Analyze or state is current
        if pil_image is not None and st.session_state.get("active_file") == getattr(uploaded_file, "name", None):
            # Check if prediction already computed
            if "prediction_result" not in st.session_state or st.session_state.get("pred_for_file") != uploaded_file.name:
                if model is not None:
                    try:
                        with st.spinner("Processing CT slice through CNN-BiLSTM-Attention..."):
                            preprocessed_tensor = preprocess_pil_image(pil_image, target_size=IMAGE_SIZE)
                            # Direct tensor call is 3-5x faster than model.predict for single-slice inference
                            probabilities = model(preprocessed_tensor, training=False).numpy()[0]
                            pred_idx = int(np.argmax(probabilities))
                            pred_class = CLASS_NAMES[pred_idx]
                            confidence = float(probabilities[pred_idx]) * 100.0

                            st.session_state["prediction_result"] = {
                                "pred_class": pred_class,
                                "pred_idx": pred_idx,
                                "confidence": confidence,
                                "probabilities": [float(p) for p in probabilities],
                                "tensor": preprocessed_tensor,
                            }
                            st.session_state["pred_for_file"] = uploaded_file.name
                    except Exception as e:
                        st.error(
                            "An unexpected error occurred during model inference. "
                            "Please verify that the image dimensions and format are valid."
                        )

        # Render prediction results or empty state
        if "prediction_result" in st.session_state and pil_image is not None:
            res = st.session_state["prediction_result"]
            pred_class = res["pred_class"]
            confidence = res["confidence"]
            probs = res["probabilities"]

            class_lower = pred_class.lower()
            hero_cls = f"pred-hero-{class_lower}"
            val_cls = f"pred-hero-val-{class_lower}"

            # Hero Card
            st.markdown(
                f"""
                <div class="pred-hero {hero_cls}">
                    <div class="pred-hero-label">Model Prediction</div>
                    <div class="pred-hero-value {val_cls}">{pred_class.upper()}</div>
                    <div class="pred-hero-conf">
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Class Probability Cards
            st.markdown(
                "<div style='font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;'>Class Probabilities</div>",
                unsafe_allow_html=True,
            )

            icons = {"Normal": "🟢", "Benign": "🟡", "Malignant": "🔴"}
            for idx, cname in enumerate(CLASS_NAMES):
                c_prob = probs[idx] * 100.0
                is_sel = (idx == res["pred_idx"])
                sel_cls = "prob-row-selected" if is_sel else ""
                row_cls = f"prob-row-{cname.lower()}"

                st.markdown(
                    f"""
                    <div class="prob-row {row_cls} {sel_cls}">
                        <span class="prob-name">{icons.get(cname, '')} {cname}</span>
                        <span class="prob-pct">{c_prob:.1f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Native Streamlit progress bar
                st.progress(float(probs[idx]))

            # Prediction Summary Grid
            st.markdown(
                """
                <div style='margin-top: 1.2rem; font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;'>
                    Prediction Summary
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="summary-grid">
                    <div class="summary-item">
                        <div class="summary-key">Predicted Class</div>
                        <div class="summary-val">{pred_class}</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-key">Confidence</div>
                        <div class="summary-val">{confidence:.1f}%</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-key">Model</div>
                        <div class="summary-val">Hybrid CNN-BiLSTM-Attention</div>
                    </div>
                    <div class="summary-item">
                        <div class="summary-key">Input Type</div>
                        <div class="summary-val">CT image (224 &times; 224)</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            # Clean empty state before inference
            st.markdown(
                """
                <div class="empty-state">
                    <div class="empty-state-icon">⚡</div>
                    <div class="empty-state-text">Upload a CT image to begin analysis.</div>
                    <div class="empty-state-subtext">Click 'Analyze CT Scan' after selecting a slice to run predictive inference.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------------
    # GRAD-CAM EXPLAINABILITY SECTION
    # -------------------------------------------------------------------------
    if "prediction_result" in st.session_state and pil_image is not None and model is not None:
        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div class="dashboard-card">
                <div class="card-title">
                    <span>🔬</span> Model Explainability
                </div>
                <div style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 1rem;">
                    Grad-CAM visualization of regions contributing to the CNN feature representation.
                </div>
            """,
            unsafe_allow_html=True,
        )

        res = st.session_state["prediction_result"]
        try:
            # Cache Grad-CAM in session state to avoid re-running on every Streamlit rerun
            if st.session_state.get("gradcam_for_file") != uploaded_file.name or "gradcam_result" not in st.session_state:
                with st.spinner("Generating Grad-CAM feature attribution..."):
                    heatmap = generate_gradcam_heatmap(
                        model,
                        res["tensor"],
                        pred_index=res["pred_idx"],
                        grad_model=grad_model,
                    )

                    rgb_np = np.array(pil_image.convert("RGB"))
                    resized_rgb = tf.image.resize(rgb_np, IMAGE_SIZE).numpy().astype(np.uint8)
                    overlay = overlay_gradcam(resized_rgb, heatmap, alpha=0.45)
                    st.session_state["gradcam_result"] = {
                        "resized_rgb": resized_rgb,
                        "overlay": overlay,
                    }
                    st.session_state["gradcam_for_file"] = uploaded_file.name

            gc_res = st.session_state["gradcam_result"]
            col_orig, col_gradcam = st.columns(2, gap="medium")
            with col_orig:
                st.image(
                    gc_res["resized_rgb"],
                    caption="Original (Preprocessed CT Slice)",
                    use_container_width=True,
                )
            with col_gradcam:
                st.image(
                    gc_res["overlay"],
                    caption="Grad-CAM (Attention & Activation Overlay)",
                    use_container_width=True,
                )

            st.markdown(
                """
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 0.8rem; line-height: 1.45; border-top: 1px solid #1e293b; padding-top: 0.6rem;">
                    ℹ️ <strong>Interpretation Note:</strong> Highlighted regions indicate image areas that contributed more strongly to the model's prediction. This visualization does not establish clinical causality or diagnosis.
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.warning(
                "Grad-CAM is currently unavailable for this prediction. "
                "The classification prediction remains valid."
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # EXPANDABLE SECTIONS: How the Model Works & Model Information
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    with st.expander("🧠 How the Model Works", expanded=False):
        st.markdown(
            """
            ```
            CT Image
               ↓
            CNN Feature Extraction
               ↓
            Spatial Feature Sequence (49 patches)
               ↓
            BiLSTM (Bidirectional Sequence Modeling)
               ↓
            Attention (Dynamic Spatial Saliency Weighting)
               ↓
            Dense Layer (Feature Projection + Dropout)
               ↓
            Softmax (Calibrated Probabilities)
               ↓
            Normal / Benign / Malignant
            ```

            #### Core Architectural Stages:
            - **CNN (Convolutional Neural Network)**: Extracts hierarchical visual features (parenchymal textures, nodule margins, opacities) from the $224 \\times 224$ CT slice, outputting a $7 \\times 7 \\times 128$ feature map.
            - **Spatial-to-Sequence Reshape**: Converts the $7 \\times 7$ grid into an ordered sequence of 49 spatial patches, each representing a localized anatomical zone.
            - **BiLSTM (Bidirectional LSTM)**: Processes the sequence of CNN feature representations in both forward and reverse directions, capturing contextual transitions between normal parenchyma and lesions.
            - **Attention Mechanism**: Assigns different importance weights ($\alpha_t$) to each spatial patch in the sequence, allowing the model to focus on focal nodule regions while down-weighting non-lesion background tissue.
            - **Softmax Classifier**: Dense classification head producing calibrated probability distributions across Normal, Benign, and Malignant categories.

            *Note: The model performs computational pattern recognition and does not perform clinical reasoning.*
            """
        )

    with st.expander("📊 Model Information", expanded=False):
        st.markdown(
            """
            - **Architecture**: Hybrid CNN + BiLSTM + Attention
            - **Input Modality**: $224 \\times 224 \\times 3$ RGB axial chest CT slice
            - **Target Classes**: `Normal` (Class 0), `Benign` (Class 1), `Malignant` (Class 2)
            - **Explainability Engine**: Grad-CAM on the final convolutional feature layer (`last_conv_layer`)
            - **Purpose**: Academic and educational prototype for pulmonary nodule analysis
            """
        )
        if metrics_data:
            st.markdown("#### Measured Held-out Test Set Benchmarks (165 Slices):")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Accuracy", f"{metrics_data.get('accuracy', 0)*100:.2f}%")
            m_col2.metric("Macro ROC-AUC", f"{metrics_data.get('roc_auc_macro', 0):.4f}")
            m_col3.metric("Macro F1-Score", f"{metrics_data.get('f1_score_macro', 0):.4f}")
            m_col4.metric("Malignant Recall", f"{metrics_data.get('per_class', {}).get('Malignant', {}).get('recall', 0)*100:.1f}%")


# =============================================================================
# VIEW B: Model Architecture
# =============================================================================
elif nav_choice == "Model Architecture":
    st.markdown(
        """
        <div class="card-title" style="font-size: 1.4rem;">
            <span>🧠</span> Model Architecture Deep-Dive
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Visual Architecture Pipeline
        ```
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
                    [ Normal | Benign | Malignant ]
        ```

        ### Mathematical Formulation

        1. **Spatial-to-Sequence Mapping**:
           $$\mathbf{S} = \\text{Reshape}_{(49, 128)}(\\mathbf{X}_{7 \\times 7 \\times 128})$$
           Each timestep $t \\in \\{1, \\dots, 49\\}$ represents a localized $32 \\times 32$ pixel receptive field patch.

        2. **Bidirectional Recurrent Context**:
           $$\\overrightarrow{\\mathbf{h}}_t = \\text{LSTM}_{\\text{fwd}}(\\mathbf{s}_t, \\overrightarrow{\\mathbf{h}}_{t-1}), \\quad \\overleftarrow{\\mathbf{h}}_t = \\text{LSTM}_{\\text{bwd}}(\\mathbf{s}_t, \\overleftarrow{\\mathbf{h}}_{t+1})$$
           $$\\mathbf{h}_t = [\\overrightarrow{\\mathbf{h}}_t \\,\\|\\, \\overleftarrow{\\mathbf{h}}_t] \\in \\mathbb{R}^{128}$$

        3. **Additive Attention Weights**:
           $$u_t = \\tanh(\\mathbf{W}_a \\mathbf{h}_t + \\mathbf{b}_a), \\quad e_t = \\mathbf{v}_a^T u_t$$
           $$\\alpha_t = \\frac{\\exp(e_t)}{\\sum_{k=1}^{49} \\exp(e_k)}, \\quad \\mathbf{c} = \\sum_{t=1}^{49} \\alpha_t \\mathbf{h}_t$$

        4. **Softmax Probabilities**:
           $$P(y = j \\mid \\mathbf{x}) = \\frac{\\exp(z_j)}{\\sum_{k=0}^{2} \\exp(z_k)}, \\quad j \\in \\{0, 1, 2\\}$$
        """
    )


# =============================================================================
# VIEW C: About Project
# =============================================================================
elif nav_choice == "About Project":
    st.markdown(
        """
        <div class="card-title" style="font-size: 1.4rem;">
            <span>ℹ️</span> About This Project
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        ### Academic Background
        This project implements a hybrid **CNN-BiLSTM-Attention** architecture for classification of lung CT images into **Normal**, **Benign**, and **Malignant** categories using the **IQ-OTH/NCCD** dataset.

        ### Architectural Adaptation Note
        The reference research work originally designed a CNN + BiLSTM + Attention framework for processing unstructured **clinical text notes** (from the MIMIC-IV database). 
        
        For this project, the core concept has been **adapted to visual spatial sequences** from thoracic CT images:
        - Treating spatial patches across the 2D feature map as an ordered token sequence.
        - Using bidirectional recurrent modeling to learn contextual dependencies between adjacent lung parenchyma zones.
        - Utilizing additive attention to dynamically focus on suspicious nodular lesions.

        > **Important Scientific Note**: This project does not claim an exact reproduction of the reference paper, because the original research operated on clinical narrative notes whereas this implementation evaluates thoracic Computed Tomography (CT) images.

        ### Dataset: IQ-OTH/NCCD
        - **Total Images**: 1,097 axial CT slices
        - **Classes**: Normal (416), Benign (120), Malignant (561)
        - **Stratified Partition**: 70% Train (767 slices), 15% Validation (165 slices), 15% Held-out Test (165 slices)
        - **Class Balancing**: Balanced inverse-frequency class weighting to counteract benign sample scarcity.

        ### Regulatory & Clinical Disclaimer
        This application is developed strictly as an educational and research prototype for college coursework evaluation. It is **not** a certified medical diagnostic device and must not be used as a substitute for professional radiological consultation or clinical diagnostic procedures.
        """
    )
