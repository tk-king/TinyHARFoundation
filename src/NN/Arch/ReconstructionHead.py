import tensorflow as tf
from tensorflow.keras import layers as L


def build_reconstruction_head(
    out_channels: int = 9,
    hidden: int | None = None,
    dropout: float = 0.1,
    name: str = "reconstruction_head",
) -> tf.keras.Model:
    """
    Head that maps per-timestep features (B, T, C) -> reconstructed signal (B, T, out_channels).

    This is intentionally shape-flexible: it infers `C` on first call, so you can use:
        multi_head.add_head("reconstruction", build_reconstruction_head())
    """
    layers: list[tf.keras.layers.Layer] = [L.LayerNormalization(name="ln")]
    if hidden is not None:
        layers += [
            L.Dense(hidden, activation="swish", name="dense_hidden"),
            L.Dropout(dropout, name="dropout"),
        ]
    layers += [L.Dense(out_channels, name="reconstruction")]
    return tf.keras.Sequential(layers, name=name)


def build_construction_head(*args, **kwargs) -> tf.keras.Model:
    # Backwards/typo-friendly alias.
    return build_reconstruction_head(*args, **kwargs)

