"""Custom Additive Attention Mechanism Layer for Sequence Weighting."""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="Custom")
class SequenceAttention(layers.Layer):
    """Custom Additive Attention Layer (Bahdanau-style) for weighting BiLSTM sequence outputs.

    Explanation for B.Tech Viva / Concept:
    --------------------------------------
    Given the BiLSTM sequence of hidden states H = [h_1, h_2, ..., h_T] representing spatial patches:
    1. Projection: Computes hidden alignment score candidate u_t = tanh(W * h_t + b)
    2. Alignment Score: Measures importance relative to trainable context vector v: e_t = u_t * v
    3. Softmax Normalization: Converts raw scores into probability distribution alpha_t = softmax(e_t),
       where sum(alpha_t) = 1. High alpha_t indicates key pathological features (e.g. nodule/lesion).
    4. Context Vector: Computes weighted sum c = sum(alpha_t * h_t), condensing the sequence into a
       single informative feature vector.
    """

    def __init__(self, return_attention_weights=False, **kwargs):
        super(SequenceAttention, self).__init__(**kwargs)
        self.return_attention_weights = return_attention_weights

    def build(self, input_shape):
        # input_shape: (batch_size, sequence_length, feature_dim)
        feature_dim = int(input_shape[-1])

        # Projection weight matrix W
        self.W = self.add_weight(
            name="attention_weight_matrix",
            shape=(feature_dim, feature_dim),
            initializer="glorot_uniform",
            trainable=True,
        )

        # Projection bias b
        self.b = self.add_weight(
            name="attention_bias",
            shape=(feature_dim,),
            initializer="zeros",
            trainable=True,
        )

        # Learnable context vector v
        self.v = self.add_weight(
            name="attention_context_vector",
            shape=(feature_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(SequenceAttention, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, feature_dim)
        # 1. Project sequence: u_t = tanh(H * W + b)
        u = tf.tanh(tf.tensordot(inputs, self.W, axes=[[2], [0]]) + self.b)

        # 2. Score alignment with context vector: e_t = u_t * v -> shape: (batch_size, seq_len, 1)
        scores = tf.tensordot(u, self.v, axes=[[2], [0]])

        # 3. Softmax over the sequence dimension (axis=1) to obtain attention weights
        weights = tf.nn.softmax(scores, axis=1)

        # 4. Compute weighted sum context vector: c = sum(alpha_t * h_t) -> shape: (batch_size, feature_dim)
        context = tf.reduce_sum(inputs * weights, axis=1)

        if self.return_attention_weights:
            return context, weights
        return context

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        feature_dim = input_shape[-1]
        if self.return_attention_weights:
            return (batch_size, feature_dim), (batch_size, input_shape[1], 1)
        return (batch_size, feature_dim)

    def get_config(self):
        config = super(SequenceAttention, self).get_config()
        config.update({"return_attention_weights": self.return_attention_weights})
        return config
