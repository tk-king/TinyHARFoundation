import tensorflow as tf
from tensorflow.keras import layers as L


class ContrastiveHead(L.Layer):
    def __init__(
        self,
        proj_dim: int = 128,
        hidden: int | None = 256,
        dropout: float = 0.1,
        name: str = "contrastive_head",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.proj_dim = int(proj_dim)
        self.hidden = None if hidden is None else int(hidden)
        self.dropout = float(dropout)

        self.ln = L.LayerNormalization(name="ln")
        self.dense_hidden = L.Dense(self.hidden, activation="swish", name="dense_hidden") if self.hidden else None
        self.do = L.Dropout(self.dropout, name="dropout") if self.hidden else None
        self.proj = L.Dense(self.proj_dim, name="proj")
        self.norm = L.UnitNormalization(axis=-1, name="unit_norm")

    def call(self, inputs, training=False):
        x = inputs  # (B,2,T,C)
        b = tf.shape(x)[0]
        t = tf.shape(x)[2]
        c = tf.shape(x)[3]

        x = tf.reshape(x, (b * 2, t, c))
        x = tf.reduce_mean(x, axis=1)

        x = self.ln(x, training=training)
        if self.dense_hidden is not None:
            x = self.dense_hidden(x, training=training)
            x = self.do(x, training=training)

        x = self.proj(x, training=training)
        x = self.norm(x, training=training)
        x = tf.reshape(x, (b, 2, self.proj_dim))
        return x


def build_contrastive_head(
    proj_dim: int = 128,
    hidden: int | None = 256,
    dropout: float = 0.1,
    name: str = "contrastive_head",
) -> tf.keras.layers.Layer:
    return ContrastiveHead(proj_dim=proj_dim, hidden=hidden, dropout=dropout, name=name)

