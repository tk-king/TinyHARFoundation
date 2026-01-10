import tensorflow as tf


class SupConLoss(tf.keras.losses.Loss):
    """
    Supervised Contrastive Loss (Khosla et al., 2020), compatible with SimCLR when labels are absent.

    Expects y_pred as either:
      - (B, V, D) where V is number of views (this repo uses V=2)
      - (B*V, D) in the order [view0 batch; view1 batch; ...]

    Expects y_true as:
      - (B,) integer class ids (supervised), or
      - None (unsupervised; uses identity mask)
    """

    def __init__(
        self,
        temperature: float = 0.07,
        contrast_mode: str = "all",
        base_temperature: float | None = None,
        name: str = "supcon",
    ):
        super().__init__(name=name)
        self.temperature = float(temperature)
        self.contrast_mode = str(contrast_mode)
        self.base_temperature = float(base_temperature if base_temperature is not None else temperature)

    def call(self, y_true, y_pred):
        features = tf.convert_to_tensor(y_pred)

        if features.shape.rank == 2:
            n = tf.shape(features)[0]
            tf.debugging.assert_equal(
                tf.math.floormod(n, 2),
                0,
                message="Expected even first dimension for (B*V,D) with V=2.",
            )
            b = n // 2
            v = 2
            features = tf.stack([features[:b], features[b:]], axis=1)  # (B,2,D)
        elif features.shape.rank == 3:
            pass
        else:
            raise ValueError(f"`features` must be rank 2 or 3, got shape {features.shape}")

        batch_size = tf.shape(features)[0]
        contrast_count = tf.shape(features)[1]

        labels = None
        if y_true is not None:
            labels = tf.reshape(tf.cast(y_true, tf.int64), (-1, 1))
            tf.debugging.assert_equal(
                tf.shape(labels)[0],
                batch_size,
                message="Num of labels does not match num of features",
            )

        if labels is None:
            mask = tf.eye(batch_size, dtype=tf.float32)
        else:
            mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)

        contrast_feature = tf.concat(tf.unstack(features, axis=1), axis=0)  # (B*V, D)

        if self.contrast_mode == "one":
            anchor_feature = features[:, 0, :]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        logits = tf.matmul(anchor_feature, contrast_feature, transpose_b=True) / self.temperature
        logits_max = tf.reduce_max(logits, axis=1, keepdims=True)
        logits = logits - tf.stop_gradient(logits_max)

        mask = tf.tile(mask, (anchor_count, contrast_count))  # (B*anchor_count, B*contrast_count)

        logits_mask = tf.ones_like(mask)
        diag = tf.range(batch_size * anchor_count)
        indices = tf.stack([diag, diag], axis=1)  # (B*anchor_count, 2)
        updates = tf.zeros((batch_size * anchor_count,), dtype=logits_mask.dtype)
        logits_mask = tf.tensor_scatter_nd_update(logits_mask, indices=indices, updates=updates)
        mask = mask * logits_mask

        exp_logits = tf.exp(logits) * logits_mask
        exp_sum = tf.reduce_sum(exp_logits, axis=1, keepdims=True)
        exp_sum = tf.maximum(exp_sum, tf.cast(1e-12, exp_sum.dtype))
        log_prob = logits - tf.math.log(exp_sum)

        mask_pos_pairs = tf.reduce_sum(mask, axis=1)
        mask_pos_pairs = tf.where(mask_pos_pairs < 1e-6, tf.ones_like(mask_pos_pairs), mask_pos_pairs)
        mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / mask_pos_pairs

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = tf.reshape(loss, (anchor_count, batch_size))
        return tf.reduce_mean(loss)
