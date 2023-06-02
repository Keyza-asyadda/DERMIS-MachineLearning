from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf


@keras.saving.register_keras_serializable()
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

@keras.saving.register_keras_serializable()
class HardSwish(keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return inputs * tf.nn.relu6(inputs) / 6

