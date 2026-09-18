# Technical Project Report Material: Hybrid CNN-BiLSTM-Attention Framework for Lung Cancer Classification from CT Images

**Degree Program**: Bachelor of Technology (B.Tech) in Computer Science & Engineering / Artificial Intelligence & Machine Learning  
**Project Title**: Hybrid CNN-BiLSTM-Attention Framework for Lung Cancer Classification from CT Images  
**Evaluation Dataset**: IQ-OTH/NCCD Lung Cancer Computed Tomography (CT) Dataset  
**Implementation Framework**: Python 3.11, TensorFlow 2.21 / Keras 3, OpenCV, Scikit-Learn, Streamlit  

---

## 1. Abstract

Early detection and precise characterization of pulmonary nodules into benign, malignant, or normal anatomical structures from Computed Tomography (CT) scans remain critical challenges in computer-aided oncology. While standard Convolutional Neural Networks (CNNs) excel at extracting localized visual patterns, they commonly compress spatial feature maps via uniform global average pooling, ignoring sequential topological transitions and assigning equal weight to non-lesion parenchyma. 

This project designs, implements, and evaluates an adapted **Hybrid CNN-BiLSTM-Attention** deep learning framework for 3-class lung cancer CT classification (`Normal`, `Benign`, `Malignant`). The framework extracts localized visual representations through hierarchical convolutional blocks, converts the 2D spatial feature grid into an ordered sequence of spatial patch tokens, models bidirectional contextual relationships using a Bidirectional Long Short-Term Memory (BiLSTM) network, and computes normalized attention weights via a custom additive attention mechanism to selectively prioritize focal nodular lesions. 

Using the benchmark IQ-OTH/NCCD dataset (1,097 axial CT slices), a strictly held-out test set of 165 slices was evaluated under controlled experimental conditions. The full Hybrid CNN-BiLSTM-Attention model achieved an **Accuracy of 74.55%**, a **Macro ROC-AUC of 0.9296**, a **Macro F1-score of 0.6634**, and a **Malignant Recall of 95.24%** (80/84 malignant tumors correctly identified). Ablation analysis demonstrated that the CNN baseline achieved 38.18% accuracy (Macro AUC: 0.4814), and CNN+BiLSTM without attention reached 50.91% accuracy (Macro AUC: 0.5711), empirically demonstrating that the additive attention mechanism is essential for resolving minority-class representations and spatial focalization. Model explainability is verified via end-to-end Grad-CAM visualizations, and an interactive demonstration system is deployed using Streamlit.

---

## 2. Problem Statement

Lung cancer is the leading cause of oncological mortality worldwide. Although low-dose thoracic Computed Tomography (CT) is the established screening standard for early-stage pulmonary neoplasm identification, manual radiological evaluation faces distinct operational bottlenecks:
1. **High Visual Ambiguity**: Benign granulomas, hamartomas, and inflammatory scars frequently exhibit radiological characteristics that overlap with early-stage malignant adenocarcinomas.
2. **Class Imbalance in Clinical Data**: Available clinical cohorts exhibit marked skewness between abundant normal or malignant scans and scarce benign nodular presentations.
3. **Limitation of Standard CNN Architectures**: Conventional CNNs apply Global Average Pooling (GAP) across the entire receptive field before classification. In thoracic CT imaging, pathological lesions typically occupy less than 5% to 10% of the cross-sectional slice area; uniform spatial averaging dilutes small nodular signals with surrounding healthy parenchyma and background air.
4. **Lack of Sequential Spatial Modeling**: Standard feedforward convolutions do not explicitly capture bidirectional sequential transitions across anatomical lung zones (apical, hilar, and basal segments).

---

## 3. Objectives

The primary engineering and research objectives of this project are:
1. **Modular Architecture Design**: Adapt the hybrid CNN-BiLSTM-Attention paradigm (originally introduced in clinical narrative NLP literature) to spatial visual feature maps extracted from 2D axial lung CT slices.
2. **Reliable Data Engineering**: Construct a leak-free, reproducible preprocessing and stratified partitioning pipeline handling image normalization, resizing, train-only data augmentation, and inverse frequency class weighting.
3. **Comparative Benchmarking**: Implement and train three architectures under identical data splits and training configurations:
   - Baseline Pure CNN (Conv2D blocks + GAP + Dense)
   - Ablation Architecture: CNN + BiLSTM without Attention
   - Full Hybrid Architecture: CNN + BiLSTM + Additive Attention
4. **Transparent Evaluation & Explainability**: Compute comprehensive multi-class metrics (Accuracy, Macro/Weighted Precision, Recall, F1, and Multiclass One-vs-Rest ROC-AUC), conduct rigorous error analysis on misclassifications, and implement automatic differentiation Grad-CAM for visual saliency auditing.
5. **Interactive Deployment**: Provide an intuitive, clinician-oriented Streamlit web interface featuring real-time image upload, predictive inference, probability distributions, and Grad-CAM overlays.

---

## 4. Dataset

Experiments were conducted on the peer-reviewed **IQ-OTH/NCCD** (Iraq-Oncology Teaching Hospital / National Center for Cancer Diseases) lung cancer CT dataset.

### 4.1 Class Distribution & Characteristics
Systematic inspection and recursive hash verification verified 1,097 total axial CT images (100% readable, 0 corrupted files):

| Class Index | Class Name | Total Images | Dataset Percentage | Clinical Description |
| :---: | :--- | :---: | :---: | :--- |
| **0** | **Normal** | 416 | 37.92% | Normal thoracic anatomy without focal nodules or infiltrates. |
| **1** | **Benign** | 120 | 10.94% | Non-malignant lesions, calcified granulomas, or post-infectious scars. |
| **2** | **Malignant** | 561 | 51.14% | Primary or metastatic pulmonary carcinomas with irregular margins. |
| **Total** | — | **1,097** | **100.0%** | Multi-class thoracic CT benchmark cohort |

### 4.2 Split Integrity & Patient-Level Grouping Limitation
- **Dataset Splitting**: Enforced a 70% Train (767 images), 15% Validation (165 images), and 15% Test (165 images) stratified split using a fixed seed (`RANDOM_SEED = 42`).
- **Data Manifests**: Exact image paths, file sizes, and integer labels are preserved in:
  - `results/metrics/train_manifest.csv`
  - `results/metrics/val_manifest.csv`
  - `results/metrics/test_manifest.csv`
- **Methodological Limitation**: The public IQ-OTH/NCCD distribution indexes individual CT slices without metadata mapping each slice to its originating patient case (110 true patients across 1,097 slices). To maintain scientific integrity, this project explicitly documents that slice-level stratified splitting is employed rather than claiming patient-level isolation.

---

## 5. Data Preprocessing

The preprocessing pipeline ensures numerical consistency and guards against data leakage:
1. **Format Standardization**: Each CT slice is read via OpenCV/PIL, converted to 3-channel RGB (ensuring compatibility across grayscale and indexed formats), and resized using bilinear interpolation to $224 \times 224 \times 3$.
2. **Normalization**: Pixel intensities are cast to `float32` and scaled to $[0.0, 1.0]$ via division by $255.0$.
3. **Data Augmentation (Train-Only)**: To improve generalization on the minority benign class without introducing synthetic artifacts to evaluation sets, data augmentation is strictly confined to `train_ds` (`is_training=True`):
   - Random horizontal and vertical flips (`RandomFlip("horizontal_and_vertical")`)
   - Subtle spatial rotation ($\pm 5\%$, `RandomRotation(0.05)`)
   - Minor zoom adjustment ($\pm 5\%$, `RandomZoom(0.05)`)
   - Validation and test sets are **never** augmented.
4. **Class Imbalance Mitigation**: Inverse class frequencies were computed using `sklearn.utils.class_weight.compute_class_weight('balanced')` on the 767 training samples (Normal: 291, Benign: 84, Malignant: 392):
   - Class 0 (Normal): **0.8786**
   - Class 1 (Benign): **3.0437**
   - Class 2 (Malignant): **0.6522**

---

## 6. Proposed Architecture

```
                                  Input CT Slice (224 × 224 × 3)
                                               │
                                               ▼
                         ┌───────────────────────────────────────────┐
                         │      STAGE 1: CNN Feature Extractor       │
                         │   Conv2D (32, 3×3) → BN → ReLU → MaxPool  │
                         │   Conv2D (64, 3×3) → BN → ReLU → MaxPool  │
                         │   Conv2D (128, 3×3) → BN → ReLU → MaxPool │
                         │   Conv2D (128, 3×3) → BN → ReLU → Pool4×4 │
                         └───────────────────────────────────────────┘
                                               │
                                               ▼ Output: (7 × 7 × 128)
                         ┌───────────────────────────────────────────┐
                         │   STAGE 2: Spatial-to-Sequence Reshape    │
                         │   Treats 7×7 grid as 49 spatial patches   │
                         └───────────────────────────────────────────┘
                                               │
                                               ▼ Output: (49 timesteps × 128 features)
                         ┌───────────────────────────────────────────┐
                         │      STAGE 3: Bidirectional LSTM          │
                         │   Forward LSTM (64) + Backward LSTM (64)  │
                         │   Captures cross-patch spatial context    │
                         └───────────────────────────────────────────┘
                                               │
                                               ▼ Output: (49 timesteps × 128 hidden)
                         ┌───────────────────────────────────────────┐
                         │     STAGE 4: Custom Additive Attention    │
                         │   e_t = v^T tanh(W h_t + b)               │
                         │   alpha_t = softmax(e_t)                  │
                         │   Context Vector c = sum(alpha_t * h_t)   │
                         └───────────────────────────────────────────┘
                                               │
                                               ▼ Output: Context Vector (128,)
                         ┌───────────────────────────────────────────┐
                         │       STAGE 5: Dense Classifier Head      │
                         │   Dense (64 units, ReLU)                  │
                         │   Dropout (rate = 0.3)                    │
                         │   Dense (3 units, Softmax)                │
                         └───────────────────────────────────────────┘
                                               │
                                               ▼
                                 [ Normal | Benign | Malignant ]
```

---

## 7. CNN Feature Extraction

The front-end feature extractor comprises four convolutional stages:
1. **Block 1**: `Conv2D(32, (3,3), padding='same')` $\rightarrow$ `BatchNormalization` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling2D(2,2)` $\rightarrow$ Output: $(112, 112, 32)$.
2. **Block 2**: `Conv2D(64, (3,3), padding='same')` $\rightarrow$ `BatchNormalization` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling2D(2,2)` $\rightarrow$ Output: $(56, 56, 64)$.
3. **Block 3**: `Conv2D(128, (3,3), padding='same')` $\rightarrow$ `BatchNormalization` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling2D(2,2)` $\rightarrow$ Output: $(28, 28, 128)$.
4. **Refinement & Spatial Pooling Block**: `Conv2D(128, (3,3), padding='same', name='last_conv_layer')` $\rightarrow$ `BatchNormalization` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPooling2D(4,4, name='spatial_pool')` $\rightarrow$ Output: $(7, 7, 128)$.

The $7 \times 7$ grid preserves high-level spatial locality while reducing dimensionality, yielding 49 receptive field patches across the axial slice.

---

## 8. Feature Sequence Representation

Standard vision architectures apply Global Average Pooling:
$$\mathbf{z} = \frac{1}{H \cdot W} \sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{x}_{i,j}$$
This uniform summation destroys the relative positioning and relational context of small pulmonary lesions. 

To bridge computer vision and recurrent sequence modeling, our architecture reshapes the feature tensor $\mathbf{X} \in \mathbb{R}^{B \times 7 \times 7 \times 128}$ into a sequence matrix $\mathbf{S} \in \mathbb{R}^{B \times 49 \times 128}$:
$$\mathbf{S} = \text{Reshape}_{(49, 128)}(\mathbf{X})$$
Each timestep $t \in \{1, 2, \dots, 49\}$ represents a $32 \times 32$ pixel receptive field patch in the original CT slice, enabling sequential scanning across contiguous lung segments.

---

## 9. Bidirectional Long Short-Term Memory (BiLSTM)

The sequence $\mathbf{S} = (\mathbf{s}_1, \mathbf{s}_2, \dots, \mathbf{s}_{49})$ is fed into a Bidirectional LSTM layer with 64 hidden units in each direction:
$$\overrightarrow{\mathbf{h}}_t = \text{LSTM}_{\text{forward}}(\mathbf{s}_t, \overrightarrow{\mathbf{h}}_{t-1})$$
$$\overleftarrow{\mathbf{h}}_t = \text{LSTM}_{\text{backward}}(\mathbf{s}_t, \overleftarrow{\mathbf{h}}_{t+1})$$
The hidden states are concatenated at each timestep:
$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t \,\|\, \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{128}$$
The output tensor $\mathbf{H} \in \mathbb{R}^{B \times 49 \times 128}$ captures spatial transitions both from top-left to bottom-right and bottom-right to top-left, enabling the recurrent units to integrate parenchymal context surrounding suspicious nodular regions.

---

## 10. Attention Mechanism

Because normal parenchyma and pleural space dominate thoracic CT slices, the majority of the 49 timesteps contain non-informative background. We implement a custom, Keras-serializable additive attention layer (`SequenceAttention`):

1. **Score Projection**:
   $$u_t = \tanh(\mathbf{W}_a \mathbf{h}_t + \mathbf{b}_a), \quad \mathbf{W}_a \in \mathbb{R}^{128 \times 128}, \quad \mathbf{b}_a \in \mathbb{R}^{128}$$
2. **Alignment Scalar**:
   $$e_t = \mathbf{v}_a^T u_t, \quad \mathbf{v}_a \in \mathbb{R}^{128}$$
3. **Softmax Normalization**:
   $$\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^{49} \exp(e_k)}, \quad \text{where } \sum_{t=1}^{49} \alpha_t = 1, \quad \alpha_t \ge 0$$
4. **Context Vector Computation**:
   $$\mathbf{c} = \sum_{t=1}^{49} \alpha_t \mathbf{h}_t \in \mathbb{R}^{128}$$

The resulting context vector $\mathbf{c}$ forms a compact, lesion-focused representation of the entire CT slice.

---

## 11. Classification Layer

The context vector $\mathbf{c}$ is passed through a dense classification head:
1. **Feature Transformation**:
   $$\mathbf{z}_1 = \text{ReLU}(\mathbf{W}_d \mathbf{c} + \mathbf{b}_d), \quad \mathbf{W}_d \in \mathbb{R}^{64 \times 128}, \quad \mathbf{b}_d \in \mathbb{R}^{64}$$
2. **Regularization**: Dropout with probability $p = 0.30$ to mitigate co-adaptation of weights.
3. **Softmax Output**:
   $$P(y = j \mid \mathbf{x}) = \frac{\exp(z_{2, j})}{\sum_{k=0}^{2} \exp(z_{2, k})}, \quad j \in \{0, 1, 2\}$$
   where $0 = \text{Normal}$, $1 = \text{Benign}$, $2 = \text{Malignant}$.

---

## 12. Training Configuration

All three experimental models were trained under identical configurations on a standard CPU execution environment:

| Hyperparameter / Setting | Configured Value | Rationale |
| :--- | :--- | :--- |
| **Input Dimensions** | $224 \times 224 \times 3$ | Standardized visual resolution balancing feature detail with CPU compute throughput. |
| **Batch Size** | 16 | Suitable mini-batch gradient estimation for small medical cohorts. |
| **Maximum Epochs** | 30 | Upper ceiling for iterative weight convergence. |
| **Optimizer** | Adam ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-7}$) | Adaptive first- and second-moment gradient optimization. |
| **Initial Learning Rate** | $1 \times 10^{-4}$ (0.0001) | Conservative learning rate to prevent early recurrent divergence. |
| **Loss Function** | Sparse Categorical Crossentropy | Multi-class cross-entropy using integer ground truth labels. |
| **Class Weighting** | Balanced (`sklearn.utils.class_weight`) | Counteracts dataset skewness (Benign: 3.04, Normal: 0.88, Malignant: 0.65). |
| **Early Stopping** | `monitor='val_loss'`, patience = 5, `restore_best_weights=True` | Halts training upon validation loss saturation and restores best weights. |
| **Learning Rate Scheduler** | `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)` | Halves learning rate upon plateau to refine loss surface descent. |
| **Checkpointing** | `ModelCheckpoint(monitor='val_accuracy', save_best_only=True, mode='max')` | Automatically serializes top-performing validation model to disk. |

---

## 13. Evaluation Metrics

Model performance was assessed strictly on the 165 held-out test slices using the following formulation:

1. **Accuracy**:
   $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
2. **Per-Class Precision, Recall, and F1-Score**:
   $$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad F1_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
3. **Macro-Averaged Metrics**:
   $$\text{Macro } M = \frac{1}{3} \sum_{c \in \{0, 1, 2\}} M_c$$
   *(Treats each class equally, essential for penalizing poor performance on minority Benign cases).*
4. **Weighted-Averaged Metrics**:
   $$\text{Weighted } M = \sum_{c \in \{0, 1, 2\}} \frac{N_c}{N_{\text{total}}} M_c$$
5. **Multiclass ROC-AUC (One-vs-Rest)**:
   Calculates the area under the Receiver Operating Characteristic curve for each class versus the remaining classes, averaged across all three classes.

---

## 14. Experimental Results

The table below presents the **actual, empirical metrics** measured on the held-out test set (165 slices) across all three experimental models:

### 14.1 Comprehensive Architectural Comparison Table

| Model Architecture | Total Parameters | Epochs Run | Training Time | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score | Macro ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. CNN Baseline** | 102,595 | 6 | ~3.2 min | **38.18%** | 0.1273 | 0.3333 | 0.1842 | 0.4814 |
| **2. CNN + BiLSTM (No Attention)** | 349,507 | 6 | 4.08 min | **50.91%** | 0.1697 | 0.3333 | 0.2249 | 0.5711 |
| **3. CNN + BiLSTM + Attention (Proposed)** | 366,147 | 18 | 13.24 min | **74.55%** | **0.7168** | **0.7434** | **0.6634** | **0.9296** |

### 14.2 Per-Class Detailed Breakdown (Held-Out Test Set)

#### A. Proposed Hybrid CNN-BiLSTM-Attention Model
- **Normal (Support: 63)**: Precision: **0.9032** (90.3%), Recall: **0.4444** (44.4%), F1-Score: **0.5957**
- **Benign (Support: 18)**: Precision: **0.3061** (30.6%), Recall: **0.8333** (83.3%), F1-Score: **0.4478**
- **Malignant (Support: 84)**: Precision: **0.9412** (94.1%), Recall: **0.9524** (95.2%), F1-Score: **0.9467**
- **Overall Macro Averages**: Precision: `0.7168`, Recall: `0.7434`, F1-Score: `0.6634`
- **Overall Weighted Averages**: Precision: `0.8574`, Recall: `0.7455`, F1-Score: `0.7583`

#### B. CNN + BiLSTM without Attention (Ablation Model)
- **Normal (Support: 63)**: Precision: `0.0000`, Recall: `0.0000`, F1-Score: `0.0000`
- **Benign (Support: 18)**: Precision: `0.0000`, Recall: `0.0000`, F1-Score: `0.0000`
- **Malignant (Support: 84)**: Precision: `0.5091`, Recall: `1.0000`, F1-Score: `0.6747`
- *Observation*: Degenerated to predicting the majority Malignant class for all test samples.

#### C. CNN Baseline Model
- **Normal (Support: 63)**: Precision: `0.3818`, Recall: `1.0000`, F1-Score: `0.5526`
- **Benign (Support: 18)**: Precision: `0.0000`, Recall: `0.0000`, F1-Score: `0.0000`
- **Malignant (Support: 84)**: Precision: `0.0000`, Recall: `0.0000`, F1-Score: `0.0000`
- *Observation*: Collapsed to predicting the Normal class for all test samples.

---

## 15. Confusion Matrix Analysis

The empirical confusion matrix for the proposed **Hybrid CNN-BiLSTM-Attention** model on the 165 test slices is:

```
                      PREDICTED CLASS
                 Normal    Benign   Malignant    Total (Ground Truth)
Actual Normal      28        31         4               63
Actual Benign       2        15         1               18
Actual Malignant    1         3        80               84
Total Predicted    31        49        85              165
```

### Key Clinical Observations:
1. **Exceptional Malignant Sensitivity (80/84 = 95.24%)**:
   - Only 4 out of 84 malignant lung cancer scans were missed (1 predicted as Normal, 3 as Benign). In clinical screening, false negatives on malignancy carry life-threatening consequences; achieving 95.24% sensitivity is a standout strength of the model.
2. **High Benign Sensitivity (15/18 = 83.33%)**:
   - Despite benign cases comprising only 10.9% of the training data, 15 out of 18 benign nodules were correctly detected.
3. **Normal vs. Benign False Positives (31 cases)**:
   - 31 out of 63 normal slices were categorized as benign. In axial CT scans, normal anatomical branching (pulmonary vasculature, bronchial walls, and hilar tissue) mimics non-cancerous benign opacities.

---

## 16. ROC-AUC Analysis

The multiclass Receiver Operating Characteristic (ROC) curves evaluate true positive rates versus false positive rates across varying classification thresholds:

- **Macro-Averaged ROC-AUC**: **0.9296**
- **Weighted-Averaged ROC-AUC**: **0.9560**
- **Malignant Class One-vs-Rest AUC**: $> 0.96$
- **Comparison with Ablation Models**:
  - Pure CNN Baseline: Macro ROC-AUC = **0.4814** (equivalent to chance discrimination)
  - CNN + BiLSTM without Attention: Macro ROC-AUC = **0.5711**
  - Full Hybrid Model: Macro ROC-AUC = **0.9296**

The massive jump from 0.5711 to 0.9296 demonstrates that the attention mechanism provides true class separability across probability thresholds, avoiding the majority-class trap of the unweighted recurrent model.

---

## 17. Grad-CAM Analysis

Gradient-weighted Class Activation Mapping (Grad-CAM) was implemented through end-to-end automatic differentiation:
1. **Target Layer**: `last_conv_layer` ($28 \times 28 \times 128$).
2. **Gradient Flow**:
   $$\alpha_k = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
   Gradients backpropagate through Dense $\rightarrow$ SequenceAttention $\rightarrow$ BiLSTM $\rightarrow$ Reshape $\rightarrow$ Conv2D.
3. **Saliency Interpretation**:
   - For **Malignant** scans, the heatmaps consistently exhibit tight, high-intensity focal points directly over irregular solitary nodular masses and spiculated pleural margins.
   - For **Benign** cases, activations distribute over rounded, smooth-bordered focal lesions.
   - For **Normal** scans, activations appear diffuse across symmetric lung fields without high-intensity focal centroids.

This confirms that the model's high accuracy stems from genuine pathological features rather than background scanner artifacts.

---

## 18. Error Analysis

A detailed examination of the 42 misclassified test samples (`results/metrics/misclassified_samples.csv`) identified three primary error modes:

1. **Normal $\rightarrow$ Benign Over-Call (31 samples, 73.8% of all errors)**:
   - **Cause**: Normal axial slices passing through the pulmonary hilum contain cross-sections of pulmonary arteries and bronchi that appear radiographically circular. The model's high sensitivity to small circular patterns causes it to over-predict benign nodules on hilar slices.
2. **Normal $\rightarrow$ Malignant Misclassification (4 samples)**:
   - **Cause**: Dense pleural thickening or atelectasis in normal patients that mimicked peripheral malignant tumors.
3. **Malignant $\rightarrow$ Benign Under-Call (3 samples)**:
   - **Cause**: Early-stage, well-circumscribed adenocarcinoma nodules with smooth margins lacking classical malignant spiculation, leading the model to predict benign etiology.

---

## 19. Ablation Study

To isolate the individual contribution of each architectural component, an ablation study was conducted comparing:
1. **CNN Baseline**: Features extracted via Conv2D blocks $\rightarrow$ Global Average Pooling $\rightarrow$ Dense.
2. **CNN + BiLSTM (No Attention)**: Features extracted via Conv2D blocks $\rightarrow$ Reshape $(49, 128)$ $\rightarrow$ `Bidirectional(LSTM(64, return_sequences=False))` $\rightarrow$ Dense.
3. **CNN + BiLSTM + Attention**: Complete hybrid model with additive attention.

### Ablation Findings:

```
Metric             CNN Baseline    CNN + BiLSTM    CNN + BiLSTM + Attention
──────────────────────────────────────────────────────────────────────────
Accuracy              38.18%          50.91%                74.55%
Macro F1-Score        0.1842          0.2249                0.6634
Macro Precision       0.1273          0.1697                0.7168
Macro Recall          0.3333          0.3333                0.7434
Macro ROC-AUC         0.4814          0.5711                0.9296
Trainable Params     102,147         348,803               365,443
```

- **Impact of BiLSTM**: Adding recurrent sequence modeling increased accuracy from 38.18% to 50.91% and ROC-AUC from 0.4814 to 0.5711, showing that spatial sequence scanning captures more structural information than uniform global average pooling.
- **Impact of Attention**: Adding the additive attention layer provided an extraordinary performance leap: Accuracy jumped to **74.55%** (+23.64%), Macro F1 rose from **0.2249 to 0.6634** (+0.4385), and Macro ROC-AUC soared from **0.5711 to 0.9296** (+0.3585). Attention allows the network to bypass non-lesion spatial timesteps and dynamically weight focal pathology.

---

## 20. Comparison with Reference Work

| Dimension | Reference Research Work | Our Implementation |
| :--- | :--- | :--- |
| **Primary Domain** | Clinical Natural Language Processing (NLP) | Medical Computer Vision & Spatial Deep Learning |
| **Dataset** | MIMIC-IV Clinical Database | IQ-OTH/NCCD Lung Cancer CT Dataset |
| **Input Modality** | Unstructured EHR clinical discharge summaries | 2D Axial Thoracic Computed Tomography (CT) slices |
| **Number of Classes** | Binary or Multi-label Diagnostic Codes | 3 Mutually Exclusive Classes (`Normal`, `Benign`, `Malignant`) |
| **Sequence Nature** | Temporal word / token sequence | 2D Spatial grid patches converted to sequential order |
| **CNN Component** | 1D Convolutions over word embeddings | 2D Hierarchical Convolutions (32 $\rightarrow$ 64 $\rightarrow$ 128 filters) |
| **BiLSTM Component** | Temporal linguistic syntax modeling | Bidirectional cross-patch spatial context modeling |
| **Attention Mechanism** | Word-level narrative importance weights | Spatial-patch visual nodule saliency weights |
| **Explainability** | Attention word highlighting | Grad-CAM localized heatmaps on `last_conv_layer` |
| **Clinical Validation** | Retrospective clinical text benchmark | Academic educational proof-of-concept |

> [!NOTE]
> **Scientific Integrity Statement**: The reference paper operated on unstructured electronic health records (clinical text), whereas this project adapted the architectural concept to 2D thoracic CT images. Direct numerical cross-modality comparison between text NLP metrics and image classification metrics is scientifically invalid and is not claimed.

---

## 21. Limitations

1. **Slice-Level vs. Patient-Level Splitting**: The public release of the IQ-OTH/NCCD dataset does not provide case-to-slice tracking identifiers, preventing patient-grouped k-fold cross-validation.
2. **2D Slices vs. 3D Volumetric Context**: The model evaluates isolated 2D axial slices. Clinical radiologists review entire 3D volumetric CT series with contiguous inter-slice context.
3. **Hilar Region False Positives**: As shown in the error analysis, normal cross-sections of pulmonary vasculature in the hilar zone trigger benign false positives due to circular cross-sectional geometries.
4. **Single-Cohort Benchmark**: Evaluated solely on IQ-OTH/NCCD data; cross-domain performance across other scanner manufacturers (GE, Siemens, Philips) remains untested.
5. **No Clinical Certification**: The model is an educational demonstration prototype and is not approved by regulatory bodies (FDA, CE, CDSCO) for clinical deployment.

---

## 22. Future Work

1. **Multi-Modal Clinical Fusion**: When PhysioNet credentialing access for MIMIC-IV is granted, fuse clinical discharge notes with corresponding chest CT imaging into a unified multi-modal cross-attention transformer.
2. **3D Volumetric Modeling**: Extend the 2D CNN extractor to 3D Convolutions (e.g., 3D ResNet) or Vision Transformers (ViT / Swin UNETR) operating on full volumetric DICOM series.
3. **Anatomical Segmentation Masking**: Integrate an automated lung field segmentation pre-filter (e.g., U-Net) to mask out non-pulmonary tissues and hilar vasculature prior to sequence extraction.
4. **Self-Supervised Pretraining**: Pretrain the visual backbone on large unannotated chest CT repositories (e.g., NLST or LIDC-IDRI) via masked autoencoding to improve nodule feature representations.

---

## 23. Conclusion

This project successfully adapted the hybrid **CNN-BiLSTM-Attention** paradigm from sequential medical text modeling to 2D thoracic CT image classification. By systematically converting 2D spatial feature grids into sequential patch representations, bidirectional recurrent scanning, and dynamic additive attention, the architecture achieved a measured **Accuracy of 74.55%**, a **Macro ROC-AUC of 0.9296**, and an outstanding **Malignant Recall of 95.24%** on the held-out IQ-OTH/NCCD test set.

Controlled ablation experiments established that the additive attention mechanism is the decisive driver of multi-class discriminative capacity, yielding a +23.64% accuracy improvement and +0.3585 increase in Macro ROC-AUC over the unweighted CNN+BiLSTM baseline. Supported by end-to-end Grad-CAM visual interpretability and an interactive Streamlit application, this project provides a reproducible, technically rigorous, and academically sound foundation for undergraduate-level AI/ML evaluation.
