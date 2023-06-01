from tensorflow.keras import backend as K
from tensorflow import keras
import tensorflow as tf


@keras.saving.register_keras_serializable()
@tf.function
def categorical_focal_crossentropy(
    y_true,
    y_pred,
    gamma = 1.,
    alpha = 1.,
    from_logits = False,
    axis = -1
):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.convert_to_tensor(y_true, dtype = y_pred.dtype)

    ce = K.categorical_crossentropy(
        y_true,
        y_pred,
        from_logits = from_logits,
        axis = axis
    )

    if from_logits:
        y_prob = K.softmax(y_pred, axis = axis)
    else:
        y_prob = y_pred

    y_prob = K.sum(y_true * y_prob, axis = axis)
    modulating_factor = K.pow(1 - y_prob, gamma)
    return K.mean(alpha * modulating_factor * ce)

@keras.saving.register_keras_serializable()
class CategoricalFocalCrossentropy(keras.__internal__.losses.LossFunctionWrapper):

    def __init__(
        self,
        gamma = 1.,
        alpha = 1.,
        from_logits = False,
        axis = -1,
        reduction = keras.losses.Reduction.AUTO,
        name = "categorical_focal_crossentropy"
    ):
        # Ensure scalar
        assert tf.rank(gamma) == 0
        assert tf.rank(alpha) == 0

        gamma = tf.constant(gamma, dtype = tf.float32)
        alpha = tf.constant(alpha, dtype = tf.float32)
        super().__init__(
            categorical_focal_crossentropy,
            reduction = reduction,
            name = name,
            gamma = gamma,
            alpha = alpha,
            from_logits = from_logits,
            axis = axis,
        )