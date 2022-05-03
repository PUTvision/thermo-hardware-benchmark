import tensorflow as tf


class CountMeanRelativeAbsoluteError(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, person_point_weight: float, name: str = 'count_mean_relative_absolute_error', dtype: tf.dtypes.DType = None):
        super().__init__(self.relative_absolute_error, name, dtype)
        self._person_point_weight = person_point_weight


    def relative_absolute_error(self, y_true, y_pred):
        ae = tf.math.abs(tf.math.subtract(y_true, y_pred))

        return tf.math.divide_no_nan(ae, y_true)


    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.math.reduce_sum(y_true, (1, 2)) / self._person_point_weight
        y_pred = tf.math.reduce_sum(y_pred, (1, 2)) / self._person_point_weight

        y_true = tf.math.round(y_true)
        y_pred = tf.math.round(y_pred)

        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)
        
        return super().update_state(y_true, y_pred, sample_weight=sample_weight)


    def get_config(self):
        return {"person_point_weight": self._person_point_weight}
