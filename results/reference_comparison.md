# Comparison with Reference Work

## 1. Architectural & Methodological Context

The reference research literature introduced the **Hybrid CNN-BiLSTM-Attention architecture** specifically for **clinical text processing and medical narrative classification** (evaluating de-identified clinical notes from MIMIC-IV). 

In our project, we have adapted this hybrid paradigm to **computer vision and volumetric radiological imaging**, classifying 2D axial Computed Tomography (CT) slices from the **IQ-OTH/NCCD** dataset into `Normal`, `Benign`, and `Malignant` categories.

> [!IMPORTANT]
> **Methodological Statement**:
> The reference work performs lung-cancer classification from clinical medical notes using MIMIC-IV, whereas our current implementation applies the hybrid CNN-BiLSTM-Attention concept to CT images from IQ-OTH/NCCD.
> Direct numerical comparisons (e.g. comparing test accuracy or F1 score directly against the reference paper's reported numbers) are **not scientifically valid** because the input modalities (unstructured clinical text vs. radiological CT pixel arrays) and benchmark datasets are fundamentally different.

---

## 2. Comparative Matrix: Reference Work vs. Our Implementation

| Aspect | Reference Work | Our Implementation |
| :--- | :--- | :--- |
| **Dataset** | MIMIC-IV (PhysioNet clinical notes repository) | IQ-OTH/NCCD (Iraq-Oncology Teaching Hospital / National Center for Cancer Diseases) |
| **Input Modality** | Unstructured clinical notes / medical text narratives | 2D Axial Computed Tomography (CT) image slices |
| **Input Representation** | Word/subword token embeddings (1D text sequence) | $224 \times 224 \times 3$ normalized pixel arrays |
| **Task** | Text-based clinical report classification | Medical image classification with visual explainability |
| **Number of Classes** | Cancer phenotype / diagnostic text categories | 3 Classes (`Normal`, `Benign`, `Malignant`) |
| **Overall Architecture** | Hybrid CNN + BiLSTM + Attention | Hybrid CNN + BiLSTM + Attention (Adapted for visual features) |
| **CNN Component** | 1D Convolutions extracting local n-gram text patterns | 2D Convolutional Blocks (32 $\rightarrow$ 64 $\rightarrow$ 128 filters) extracting local spatial textures, parenchymal densities, and nodule margins |
| **Sequence Conversion** | Embedded word tokens ordered sequentially | Reshaping 2D spatial feature map ($7 \times 7 \times 128$) into 49 sequential spatial patches ($49 \times 128$) |
| **BiLSTM Component** | Bidirectional LSTM capturing contextual sentence dependencies | Bidirectional LSTM ($64$ units) scanning the 49 spatial patches forward and backward across the chest slice |
| **Attention Mechanism** | Word/phrase-level attention weighting key clinical diagnostic terms | Custom additive `SequenceAttention` weighting key spatial CT patches (focusing on nodular lesions over background parenchyma) |
| **Model Output** | Dense softmax classification layer | Dense ($64$, ReLU) + Dropout ($0.3$) + Dense ($3$, Softmax) |
| **Evaluation Metrics** | Precision, Recall, F1-Score, Accuracy | Accuracy, Macro/Weighted Precision, Recall, F1-Score, Multiclass One-vs-Rest ROC-AUC, Confusion Matrix |
| **Explainability** | Text token importance weights | Grad-CAM feature activation heatmaps overlaid on CT slices |

---

## 3. Scientific Justification for the Visual Adaptation

In clinical narrative classification, words in a medical note have spatial proximity and order-dependent meaning. In radiological CT imaging:
1. **Spatial Continuity**: Pulmonary nodules do not exist in isolation; their diagnostic interpretation depends on surrounding vascular structures, bronchioles, and parenchymal boundaries.
2. **Patch Sequence Formulation**: Converting the high-level convolutional feature map into a sequence allows the BiLSTM to model directional transitions across spatial quadrants.
3. **Adaptive Focal Attention**: While a chest CT slice covers the entire thoracic cavity, a cancerous nodule typically occupies a small focal subregion. The Attention mechanism dynamically assigns higher weights $\alpha_t$ to the nodule-containing patches while down-weighting non-informative background regions.

---

## 4. Future Multi-Modal Synthesis (Post PhysioNet Credentialing)

Upon approval of MIMIC-IV PhysioNet access, this project's modular design enables a unified multi-modal fusion architecture:
- **Visual Branch**: The current CNN-BiLSTM-Attention image feature extractor for chest CT scans.
- **Textual Branch**: A clinical note embedding branch utilizing the reference paper's text pipeline.
- **Fusion Layer**: Joint latent representation vector combining radiologic imaging with clinical oncological notes for multi-modal decision support.
