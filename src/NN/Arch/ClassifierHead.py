import tensorflow as tf
from tensorflow.keras import layers as L

def build_classifier_head(num_classes: int, channels: int, dropout: float = 0.2) -> tf.keras.Model:
    x_in = tf.keras.Input(shape=(None, channels))   # (B,T,C)
    x = L.GlobalAveragePooling1D()(x_in)            # (B,C)
    x = L.Dropout(dropout)(x)
    logits = L.Dense(num_classes, name="logits")(x) # params ~= C*num_classes + num_classes
    return tf.keras.Model(x_in, logits, name="classifier_head")

