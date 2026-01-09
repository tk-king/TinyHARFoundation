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
