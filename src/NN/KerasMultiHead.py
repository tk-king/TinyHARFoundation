import tensorflow as tf


class MultiHeadModel(tf.keras.Model):
    def __init__(self, backbone: tf.keras.Model, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self._heads: dict[str, tf.keras.layers.Layer] = {}

    @property
    def heads(self) -> dict[str, tf.keras.layers.Layer]:
        # Return a copy to discourage untracked mutation.
        return dict(self._heads)

    def add_head(self, name: str, head: tf.keras.layers.Layer):
        if name in self._heads:
            raise ValueError(f"Head '{name}' already exists.")
        setattr(self, f"head_{name}", head)
        self._heads[name] = head

    def remove_head(self, name: str):
        if name not in self._heads:
            raise ValueError(f"Head '{name}' does not exist.")
        self._heads.pop(name)
        attr = f"head_{name}"
        if hasattr(self, attr):
            delattr(self, attr)

    def new_with_heads(self, heads: dict[str, tf.keras.layers.Layer]) -> "MultiHeadModel":
        """
        Create a new MultiHeadModel sharing the same backbone (and its weights).

        Keras locks model state after build(), so adding heads to an already-built model
        will raise. Use this to switch heads after pretraining:
            finetune = multi_head.new_with_heads({"classifier": classifier_head})
        """
        m = MultiHeadModel(self.backbone)
        for name, head in heads.items():
            m.add_head(name, head)
        return m

    def call(self, inputs, training=False):
        feat = self.backbone(inputs, training=training)
        return {name: head(feat, training=training) for name, head in self._heads.items()}
