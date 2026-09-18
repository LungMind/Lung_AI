# Streamlit Demonstration Application: Hybrid CNN-BiLSTM-Attention Lung Cancer Classifier

An interactive clinical AI dashboard demonstrating 3-class lung cancer classification on axial CT slices (`Normal`, `Benign`, `Malignant`) with real-time Grad-CAM explainability.

---

## 1. How to Run the Application

From the project root directory, execute:

```bash
streamlit run app/app.py
```

Once launched, access the dashboard at:
`http://localhost:8501`

---

## 2. Key Features

1. **Model Caching**: The trained `models/hybrid_cnn_bilstm_attention.keras` model is loaded only once into memory using `@st.cache_resource`, ensuring fast and efficient inference without model reloading or retraining.
2. **Image Upload**: Accepts standard image slice formats (PNG, JPG, JPEG, BMP, WEBP, TIFF).
3. **Interactive Predict Button**: Triggers the forward inference pipeline on demand.
4. **Class Probabilities**: Displays calibrated confidence scores and visual progress bars for `Normal`, `Benign`, and `Malignant`.
5. **Explainable AI (Grad-CAM)**: Generates and displays the activation heatmap side-by-side with the overlaid CT slice, highlighting the spatial regions that influenced the model's classification.
6. **Model Information Sidebar**: Outlines the CNN feature extraction, BiLSTM sequence modeling, attention mechanism, and 3-class classification head.
7. **Medical Disclaimer**: Emphasizes the educational/research scope of the system.

---

## 3. Important Safety & Usage Notes

- **Educational Purpose**: This application is strictly an educational and academic machine learning prototype. It is **NOT** a certified medical diagnostic instrument and must never replace consultation with certified radiologists and oncologists.
- **Pre-trained Weights Required**: Ensure `models/hybrid_cnn_bilstm_attention.keras` exists (trained via `python src/train_hybrid.py`).
