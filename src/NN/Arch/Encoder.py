import tensorflow as tf
import tensorflow.keras.layers as L



def build_imu_backbone(ts_len: int, emb_dim: int = 128, dropout: float = 0.1) -> tf.keras.Model:
    x_in = tf.keras.Input(shape=(ts_len, 9), name="imu")  # (B, T, 9)

    x = L.LayerNormalization(name="ln_in")(x_in)

    # TCN-style stack (fast, strong baseline for IMU)
    for i, (filters, dilation) in enumerate([(64, 1), (128, 2), (128, 4), (256, 8)]):
        res = x
        x = L.Conv1D(filters, 5, padding="same", dilation_rate=dilation, name=f"c{i}_a")(x)
        x = L.BatchNormalization(name=f"bn{i}_a")(x)
        x = L.Activation("swish", name=f"act{i}_a")(x)
        x = L.Dropout(dropout, name=f"do{i}_a")(x)

        x = L.Conv1D(filters, 5, padding="same", dilation_rate=dilation, name=f"c{i}_b")(x)
        x = L.BatchNormalization(name=f"bn{i}_b")(x)

        if res.shape[-1] != filters:
            res = L.Conv1D(filters, 1, padding="same", name=f"proj{i}")(res)

        x = L.Add(name=f"add{i}")([x, res])
        x = L.Activation("swish", name=f"act{i}_out")(x)

    # Attention pooling -> fixed vector regardless of T
    attn = L.Dense(1, name="attn_logits")(x)
    attn = L.Softmax(axis=1, name="attn_weights")(attn)

    weighted = L.Multiply(name="attn_mul")([x, attn])
    pooled = L.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="attn_pool")(weighted)


    # Embedding head
    z = L.Dense(emb_dim, name="emb_dense")(pooled)
    z = L.LayerNormalization(name="emb_ln")(z)
    z = L.UnitNormalization(axis=-1, name="embedding")(z)


    return tf.keras.Model(x_in, z, name="imu_backbone")

def build_backbone_seq(ts_len: int, c: int = 128, dropout: float = 0.1) -> tf.keras.Model:
    x_in = tf.keras.Input(shape=(ts_len, 9), name="imu")  # (B,T,9)
    x = L.LayerNormalization()(x_in)

    # Local features
    x = L.Conv1D(64, 7, padding="same", activation="swish")(x)
    x = L.Dropout(dropout)(x)

    # Lightweight Transformer blocks -> (B,T,C)
    x = L.Dense(c)(x)  # project to C

    for _ in range(4):
        # MHSA
        y = L.LayerNormalization()(x)
        y = L.MultiHeadAttention(num_heads=4, key_dim=c // 4, dropout=dropout)(y, y)
        y = L.Dropout(dropout)(y)
        x = L.Add()([x, y])

        # FFN
        y = L.LayerNormalization()(x)
        y = L.Dense(4 * c, activation="swish")(y)
        y = L.Dropout(dropout)(y)
        y = L.Dense(c)(y)
        y = L.Dropout(dropout)(y)
        x = L.Add()([x, y])

    x = L.LayerNormalization(name="feat_ln")(x)  # (B,T,C)
    return tf.keras.Model(x_in, x, name="imu_backbone_seq")

def build_conv_backbone_seq(ts_len: int, c: int = 128, dropout: float = 0.1) -> tf.keras.Model:
    x_in = tf.keras.Input(shape=(ts_len, 9), name="imu")  # (B,T,9)
    x = L.LayerNormalization()(x_in)

    # Stem
    x = L.Conv1D(64, 9, padding="same", use_bias=False)(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    # ResNet/TCN-style blocks -> (B,T,C)
    for i, (filters, dilation, k) in enumerate([
        (16, 1, 5),
        (32, 2, 5),
        # (64, 1, 5),
        (c,  1, 3),
        (c,  2, 3),
        # (c,  4, 3),
        (c,  8, 3),
    ]):
        res = x

        x = L.Conv1D(filters, k, padding="same", dilation_rate=dilation, use_bias=False, name=f"b{i}_c1")(x)
        x = L.BatchNormalization(name=f"b{i}_bn1")(x)
        x = L.Activation("swish", name=f"b{i}_a1")(x)
        x = L.Dropout(dropout, name=f"b{i}_do")(x)

        x = L.Conv1D(filters, k, padding="same", dilation_rate=dilation, use_bias=False, name=f"b{i}_c2")(x)
        x = L.BatchNormalization(name=f"b{i}_bn2")(x)

        if res.shape[-1] != filters:
            res = L.Conv1D(filters, 1, padding="same", use_bias=False, name=f"b{i}_proj")(res)
            res = L.BatchNormalization(name=f"b{i}_proj_bn")(res)

        x = L.Add(name=f"b{i}_add")([x, res])
        x = L.Activation("swish", name=f"b{i}_out")(x)

    x = L.LayerNormalization(name="feat_ln")(x)  # (B,T,C)
    return tf.keras.Model(x_in, x, name="imu_conv_backbone_seq")