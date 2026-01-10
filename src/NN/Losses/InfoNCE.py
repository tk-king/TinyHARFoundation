import tensorflow as tf


class InfoNCELoss(tf.keras.losses.Loss):
    def __init__(self, temperature: float = 0.1, name: str = "info_nce"):
        super().__init__(name=name)
        self.temperature = float(temperature)

    def call(self, y_true, y_pred):
        """
        InfoNCE / NT-Xent loss for two views per sample.

        Accepts:
          - y_pred: (B, 2, D) where y_pred[:,0] and y_pred[:,1] are paired views
          - y_pred: (2B, D) where first B are view0 and next B are view1
        """
        z = y_pred
        z = tf.convert_to_tensor(z)

        if z.shape.rank == 3:
            z1 = z[:, 0, :]
            z2 = z[:, 1, :]
            z = tf.concat([z1, z2], axis=0)  # (2B, D) with correct ordering
        elif z.shape.rank == 2:
            pass
        else:
            raise ValueError(f"Expected y_pred rank 2 or 3, got shape {z.shape}")

        z = tf.math.l2_normalize(z, axis=-1)

        n = tf.shape(z)[0]          # 2B
        half = n // 2               # B
        logits = tf.matmul(z, z, transpose_b=True) / self.temperature  # (2B,2B)

        big_neg = tf.cast(-1e9, logits.dtype)
        logits = tf.where(tf.eye(n, dtype=tf.bool), big_neg, logits)

        idx = tf.range(n)
        pos = tf.where(idx < half, idx + half, idx - half)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pos, logits=logits)
        return tf.reduce_mean(loss)
