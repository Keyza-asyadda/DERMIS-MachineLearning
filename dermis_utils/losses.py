from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
from .const import register


@register
class CategoricalFocalCrossentropy(keras.losses.Loss):

    def __init__(
        self,
        gamma = 1.,
        alpha = 1.,
        from_logits = False,
        axis = -1,
        reduction = keras.losses.Reduction.AUTO,
        name = "categorical_focal_crossentropy"
    ):
        super().__init__(reduction = reduction, name = name)
        # Ensure scalar
        assert tf.rank(gamma) == 0
        assert tf.rank(alpha) == 0

        self._gamma = tf.constant(gamma, dtype = tf.float32)
        self._alpha = tf.constant(alpha, dtype = tf.float32)
        self.from_logits = from_logits
        self._axis = axis
    
    def call(self, y_true, y_pred, sample_weight = None):

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.convert_to_tensor(y_true, dtype = y_pred.dtype)

        ce = K.categorical_crossentropy(
            y_true,
            y_pred,
            from_logits = self.from_logits,
            axis = self._axis
        )

        if self.from_logits:
            y_prob = K.softmax(y_pred, axis = self._axis)
        else:
            y_prob = y_pred

        y_prob = K.sum(y_true * y_prob, axis = self._axis)
        modulating_factor = K.pow(1 - y_prob, self._gamma)
        return K.mean(self._alpha * modulating_factor * ce)

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self._gamma.numpy().tolist(),
            'alpha': self._alpha.numpy().tolist(),
            'from_logits': self.from_logits,
            'axis': self._axis
        })
        return config