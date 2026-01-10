import tensorflow as tf

def make_masked_ds(x_np, batch_size=64, mask_ratio=0.2):
    ds = tf.data.Dataset.from_tensor_slices(x_np.astype("float32"))

    def _map(x):  # x: (T, 9)
        # mask per-timestep (broadcast over 9 channels)
        t = tf.shape(x)[0]
        mask_t = tf.random.uniform([t, 1]) < mask_ratio          # (T, 1) bool
        mask = tf.cast(mask_t, x.dtype)                          # (T, 1) float
        x_masked = x * (1.0 - mask)                              # zero-out masked steps
        return (x_masked, mask), x                               # inputs, target

    return ds.shuffle(10_000).map(_map, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_masked_reconstruction_ds(
    x_np,
    batch_size: int = 64,
    mask_ratio: float = 0.2,
    shuffle: int = 10_000,
):
    """
    Returns a dataset suitable for masked reconstruction pretraining.

    Output elements are (x, y, sample_weight):
      - x: (B, T, C_in) masked input
      - y: {"reconstruction": (B, T, C_in)} original signal
      - sample_weight: {"reconstruction": (B, T)} 1.0 on masked steps, 0.0 otherwise
    """
    ds = tf.data.Dataset.from_tensor_slices(x_np.astype("float32"))

    def _map(x):  # x: (T, C_in)
        t = tf.shape(x)[0]
        mask_t = tf.random.uniform([t]) < mask_ratio  # (T,) bool
        mask = tf.cast(mask_t, x.dtype)               # (T,) float
        x_masked = x * (1.0 - mask[:, None])          # (T, C_in)
        y = {"reconstruction": x}
        sw = {"reconstruction": mask}
        return x_masked, y, sw

    if shuffle:
        ds = ds.shuffle(shuffle)
    return (
        ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def make_contrastive_ds(
    x_np,
    *,
    batch_size: int = 64,
    proj_dim: int = 128,
    shuffle: int = 10_000,
    augmenter=None,
    independent_views: bool = True,
):
    """
    Returns a dataset suitable for 2-view contrastive pretraining (SimCLR-style).

    Output elements are (x, y):
      - x: (B, 2, T, C_in) two augmented views per original sample
      - y: {"contrastive": (B, 2, proj_dim)} dummy zeros (InfoNCE ignores y_true)
    """
    x_np = x_np.astype("float32")
    ts_len = int(x_np.shape[1])
    c_in = int(x_np.shape[2])

    ds = tf.data.Dataset.from_tensor_slices(x_np)

    def _augment_pair_np(x_single):
        import numpy as np

        x_single = x_single.astype(np.float32, copy=False)  # (T, C_in)
        if augmenter is None:
            rng = np.random.default_rng()
            v1 = x_single + rng.normal(0.0, 0.02, size=x_single.shape).astype(np.float32, copy=False)
            v2 = x_single + rng.normal(0.0, 0.02, size=x_single.shape).astype(np.float32, copy=False)
            return np.stack([v1, v2], axis=0)

        if independent_views:
            x1 = x_single[None, None, :, :]  # (1,1,T,C_in)
            x2 = x_single[None, None, :, :]
            v1 = augmenter(x1)[0, 0, :, :]
            v2 = augmenter(x2)[0, 0, :, :]
            return np.stack([v1, v2], axis=0)

        x_batch = np.repeat(x_single[None, None, :, :], 2, axis=0)  # (2,1,T,C_in)
        y = augmenter(x_batch)
        return y[:, 0, :, :]  # (2, T, C_in)

    def _map(x):  # x: (T, C_in)
        views = tf.numpy_function(_augment_pair_np, [x], Tout=tf.float32)
        views.set_shape((2, ts_len, c_in))
        y = {"contrastive": tf.zeros((2, proj_dim), dtype=tf.float32)}
        return views, y

    if shuffle:
        ds = ds.shuffle(shuffle)
    return (
        ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def make_supcon_ds(
    x_np,
    y_np,
    *,
    batch_size: int = 64,
    proj_dim: int = 128,
    shuffle: int = 10_000,
    augmenter=None,
    independent_views: bool = True,
):
    """
    Returns a dataset suitable for supervised contrastive pretraining.

    Output elements are (x, y):
      - x: (B, 2, T, C_in) two augmented views per original sample
      - y: {"contrastive": (B,)} integer class id (used by SupConLoss)
    """
    x_np = x_np.astype("float32")
    y_np = y_np.astype("int64")
    ts_len = int(x_np.shape[1])
    c_in = int(x_np.shape[2])

    ds = tf.data.Dataset.from_tensor_slices((x_np, y_np))

    def _augment_pair_np(x_single):
        import numpy as np

        x_single = x_single.astype(np.float32, copy=False)  # (T, C_in)
        if augmenter is None:
            rng = np.random.default_rng()
            v1 = x_single + rng.normal(0.0, 0.02, size=x_single.shape).astype(np.float32, copy=False)
            v2 = x_single + rng.normal(0.0, 0.02, size=x_single.shape).astype(np.float32, copy=False)
            return np.stack([v1, v2], axis=0)

        if independent_views:
            x1 = x_single[None, None, :, :]  # (1,1,T,C_in)
            x2 = x_single[None, None, :, :]
            v1 = augmenter(x1)[0, 0, :, :]
            v2 = augmenter(x2)[0, 0, :, :]
            return np.stack([v1, v2], axis=0)

        x_batch = np.repeat(x_single[None, None, :, :], 2, axis=0)  # (2,1,T,C_in)
        y = augmenter(x_batch)
        return y[:, 0, :, :]  # (2, T, C_in)

    def _map(x, y):  # x: (T,C_in), y: ()
        views = tf.numpy_function(_augment_pair_np, [x], Tout=tf.float32)
        views.set_shape((2, ts_len, c_in))
        return views, {"contrastive": tf.cast(y, tf.int64)}

    if shuffle:
        ds = ds.shuffle(shuffle)
    return (
        ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
