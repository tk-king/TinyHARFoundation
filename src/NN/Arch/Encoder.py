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

def build_backbone_seq(ts_len: int, c: int = 64, dropout: float = 0.1) -> tf.keras.Model:
    return build_transformer_backbone_seq(ts_len=ts_len, c=c, dropout=dropout)


def build_transformer_backbone_seq(
    ts_len: int,
    c: int = 64,
    depth: int = 2,
    num_heads: int = 2,
    mlp_ratio: int = 2,
    stem_channels: int = 32,
    dropout: float = 0.1,
    attn_dropout: float | None = None,
    use_positional_embedding: bool = True,
) -> tf.keras.Model:
    """
    Transformer-based sequential backbone for IMU.

    Input:  (B, T, 9)
    Output: (B, T, C)
    """
    if attn_dropout is None:
        attn_dropout = dropout

    x_in = tf.keras.Input(shape=(ts_len, 9), name="imu")  # (B,T,9)
    x = L.LayerNormalization(name="ln_in")(x_in)

    # Local feature stem
    x = L.Conv1D(stem_channels, 7, padding="same", use_bias=False, name="stem_c")(x)
    x = L.BatchNormalization(name="stem_bn")(x)
    x = L.Activation("swish", name="stem_act")(x)
    x = L.Dropout(dropout, name="stem_do")(x)

    # Project to model dimension
    x = L.Dense(c, name="proj_c")(x)  # (B,T,C)

    if use_positional_embedding:
        positions = tf.range(ts_len)
        pos = L.Embedding(input_dim=ts_len, output_dim=c, name="pos_emb")(positions)  # (T,C)
        pos = tf.expand_dims(pos, axis=0)  # (1,T,C)
        x = L.Add(name="add_pos")([x, pos])

    key_dim = max(1, c // num_heads)

    for i in range(depth):
        # MHSA
        y = L.LayerNormalization(name=f"b{i}_ln1")(x)
        y = L.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=attn_dropout,
            name=f"b{i}_mha",
        )(y, y)
        y = L.Dropout(dropout, name=f"b{i}_attn_do")(y)
        x = L.Add(name=f"b{i}_attn_add")([x, y])

        # FFN
        y = L.LayerNormalization(name=f"b{i}_ln2")(x)
        y = L.Dense(mlp_ratio * c, activation="swish", name=f"b{i}_ff1")(y)
        y = L.Dropout(dropout, name=f"b{i}_ff_do1")(y)
        y = L.Dense(c, name=f"b{i}_ff2")(y)
        y = L.Dropout(dropout, name=f"b{i}_ff_do2")(y)
        x = L.Add(name=f"b{i}_ff_add")([x, y])

    x = L.LayerNormalization(name="feat_ln")(x)  # (B,T,C)
    return tf.keras.Model(x_in, x, name="imu_transformer_backbone_seq")

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
