from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf
from .const import register

@register
class Swish(keras.layers.Layer):

    def __init__(self, beta = None, *args, **kwargs):
        # Ensure scalar
        assert beta is None or tf.rank(beta) == 0

        super().__init__(*args, **kwargs)
        self._beta = beta

    def build(self, input_shape):
        if self._beta is None:
            self._beta = keras.initializers.Ones()
        else:
            self._beta = keras.initializers.Constant(self._beta)

        self._beta = self.add_weight(
            name = "beta",
            shape = (),
            initializer = self._beta,
            trainable = self.trainable
        )
        self.built = True
        return input_shape
    
    def call(self, inputs, *args, **kwargs):
        return inputs * K.sigmoid(self._beta * inputs)

@register
class HardSwish(keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return inputs * tf.nn.relu6(inputs) / 6


@register
class PreprocessKecam(keras.layers.Layer):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = tf.constant([[[[123.675, 116.28, 103.53]]]], dtype = tf.float32)
        self.std = tf.constant([[[[58.395, 57.120003, 57.375]]]], dtype = tf.float32)
    
    def call(self, inputs):
        return (inputs - self.mean) / self.std