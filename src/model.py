"""Model architectures for Lung Cancer CT Classification.

Includes:
1. Baseline CNN: Conv2D blocks -> GlobalAveragePooling -> Dense -> Softmax
2. Final Hybrid: CNN Feature Extractor -> Reshape to Spatial Sequence -> BiLSTM -> Attention -> Dense -> Softmax
"""

import sys
from pathlib import Path

# Add project root to sys.path so both direct and module executions work
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    IMAGE_CHANNELS,
    NUM_CLASSES,
    LEARNING_RATE,
    LSTM_UNITS,
    DENSE_UNITS,
    DROPOUT_RATE,
)
from src.attention import SequenceAttention


# ==========================================================================
# 1. Baseline CNN Model
# ==========================================================================
def build_cnn_baseline(
    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
    num_classes=NUM_CLASSES,
    dense_units=DENSE_UNITS,
    dropout_rate=DROPOUT_RATE,
):
    """Construct standard CNN baseline for 3-class lung cancer CT classification."""
    inputs = layers.Input(shape=input_shape, name="ct_input_224")

    # Block 1: 32 filters (output: 112x112x32)
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2: 64 filters (output: 56x56x64)
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3: 128 filters (output: 28x28x128)
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Global Average Pooling Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout_gap")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_features")(x)
    x = layers.Dropout(dropout_rate, name="dropout_dense")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_Baseline")
    return model


def get_compiled_cnn_baseline(learning_rate=LEARNING_RATE, loss="sparse_categorical_crossentropy"):
    """Compile baseline CNN with Adam optimizer and sparse categorical crossentropy."""
    model = build_cnn_baseline()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )
    return model


# ==========================================================================
# 2. Final Hybrid CNN-BiLSTM-Attention Model
# ==========================================================================
def build_hybrid_model(
    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
    num_classes=NUM_CLASSES,
    lstm_units=LSTM_UNITS,
    dense_units=DENSE_UNITS,
    dropout_rate=DROPOUT_RATE,
):
    """Construct the Hybrid CNN-BiLSTM-Attention Model.

    Adaptation Note:
    ----------------
    The reference paper proposed CNN + BiLSTM + Attention for clinical medical-note sequences.
    Here we adapt this paradigm to CT imaging:
    1. CNN Feature Extractor: Learns 2D local spatial representations from axial CT slices.
    2. Sequence Reshape: Treats spatial grid patches (height x width) as an ordered sequence of feature tokens.
    3. BiLSTM: Scans the spatial feature sequence bidirectionally to model spatial relationships.
    4. Attention: Calculates normalized attention weights to dynamically focus on lesion/nodule regions.
    5. Dense Classifier: Categorizes into Normal, Benign, or Malignant.
    """
    inputs = layers.Input(shape=input_shape, name="ct_image_input")

    # -------------------------------------------------------------
    # STAGE 1: CNN Feature Extractor (reusing baseline feature blocks)
    # -------------------------------------------------------------
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)  # 112x112x32

    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)  # 56x56x64

    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)  # 28x28x128

    # Final feature refinement layer (accessible for Grad-CAM)
    x = layers.Conv2D(128, (3, 3), padding="same", name="last_conv_layer")(x)
    x = layers.BatchNormalization(name="bn_last")(x)
    x = layers.Activation("relu", name="relu_last")(x)
    x = layers.MaxPooling2D((4, 4), name="spatial_pool")(x)  # 7x7x128 -> 49 spatial patches

    # -------------------------------------------------------------
    # STAGE 2: Spatial-to-Sequence Conversion
    # (batch, 7, 7, 128) -> (batch, 49, 128)
    # Each spatial location (patch) becomes one timestep in the sequence.
    # -------------------------------------------------------------
    b, h, w, c = x.shape
    seq_len = int(h * w)
    feature_sequence = layers.Reshape(target_shape=(seq_len, int(c)), name="spatial_to_sequence")(x)

    # -------------------------------------------------------------
    # STAGE 3: Bidirectional LSTM Sequence Processing
    # Scans the 49 spatial patches forward and backward
    # Output: (batch, 49, 2 * lstm_units) = (batch, 49, 128)
    # -------------------------------------------------------------
    bilstm_output = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=True),
        name="bidirectional_lstm",
    )(feature_sequence)

    # -------------------------------------------------------------
    # STAGE 4: Attention Layer
    # Computes importance weights alpha_t and returns context vector c
    # Output: (batch, 128)
    # -------------------------------------------------------------
    context_vector = SequenceAttention(name="attention_layer")(bilstm_output)

    # -------------------------------------------------------------
    # STAGE 5: Fully Connected Classifier Head
    # -------------------------------------------------------------
    dense = layers.Dense(dense_units, activation="relu", name="classifier_dense")(context_vector)
    dense = layers.Dropout(dropout_rate, name="classifier_dropout")(dense)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax_output")(dense)

    model = models.Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_BiLSTM_Attention")
    model.last_conv_name = "last_conv_layer"
    return model


def get_compiled_hybrid_model(learning_rate=LEARNING_RATE, loss="sparse_categorical_crossentropy"):
    """Compile hybrid model with Adam optimizer and sparse categorical crossentropy."""
    model = build_hybrid_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )
    return model


# ==========================================================================
# 3. Ablation Model: CNN + BiLSTM (WITHOUT Attention)
# ==========================================================================
def build_cnn_bilstm_ablation(
    input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
    num_classes=NUM_CLASSES,
    lstm_units=LSTM_UNITS,
    dense_units=DENSE_UNITS,
    dropout_rate=DROPOUT_RATE,
):
    """Construct CNN + BiLSTM ablation model WITHOUT attention mechanism.

    Ablation Concept:
    Instead of calculating dynamic attention weights across the spatial patch sequence,
    the BiLSTM returns its final forward and backward recurrent states directly.
    """
    inputs = layers.Input(shape=input_shape, name="ct_image_input")

    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation("relu", name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.Activation("relu", name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.Activation("relu", name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    x = layers.Conv2D(128, (3, 3), padding="same", name="last_conv_layer")(x)
    x = layers.BatchNormalization(name="bn_last")(x)
    x = layers.Activation("relu", name="relu_last")(x)
    x = layers.MaxPooling2D((4, 4), name="spatial_pool")(x)  # 7x7x128

    b, h, w, c = x.shape
    seq_len = int(h * w)
    feature_sequence = layers.Reshape(target_shape=(seq_len, int(c)), name="spatial_to_sequence")(x)

    # BiLSTM without Attention (return_sequences=False -> outputs final states of shape (batch, 2*units))
    bilstm_output = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False),
        name="bidirectional_lstm",
    )(feature_sequence)

    dense = layers.Dense(dense_units, activation="relu", name="classifier_dense")(bilstm_output)
    dense = layers.Dropout(dropout_rate, name="classifier_dropout")(dense)
    outputs = layers.Dense(num_classes, activation="softmax", name="softmax_output")(dense)

    model = models.Model(inputs=inputs, outputs=outputs, name="CNN_BiLSTM_Ablation")
    model.last_conv_name = "last_conv_layer"
    return model


def get_compiled_cnn_bilstm_ablation(learning_rate=LEARNING_RATE, loss="sparse_categorical_crossentropy"):
    """Compile CNN + BiLSTM ablation model."""
    model = build_cnn_bilstm_ablation()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    baseline = build_cnn_baseline()
    print("Baseline CNN summary:")
    baseline.summary()

    hybrid = build_hybrid_model()
    print("\nHybrid CNN-BiLSTM-Attention summary:")
    hybrid.summary()
