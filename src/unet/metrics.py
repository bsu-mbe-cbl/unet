import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops, confusion_matrix, init_ops, math_ops


def mean_iou(y_true, y_pred, smooth=1):
    n_classes = y_pred.shape[-1]
    y_true = tf.cast(y_true[..., 0], tf.dtypes.int64)
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.dtypes.int64)

    y_true = tf.one_hot(y_true, n_classes)
    y_pred = tf.one_hot(y_pred, n_classes)

    y_true = tf.cast(y_true, tf.dtypes.float64)
    y_pred = tf.cast(y_pred, tf.dtypes.float64)

    intersection = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
    union = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - intersection
    return tf.reduce_mean((intersection + smooth) / (union + smooth))


def dice_coefficient(y_true, y_pred, smooth=1):
    n_classes = y_pred.shape[-1]
    y_true = tf.cast(y_true[..., 0], tf.dtypes.int64)
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.dtypes.int64)

    y_true = tf.one_hot(y_true, n_classes)
    y_pred = tf.one_hot(y_pred, n_classes)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    total = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(
        y_pred, axis=[1, 2, 3]
    )
    dice = tf.reduce_mean((2.0 * intersection + smooth) / (total + smooth), axis=0)
    return dice


class Jaccard(tf.keras.metrics.Metric):
    """Computes the mean Intersection-Over-Union metric.
    Mean Intersection-Over-Union is a common evaluation metric for semantic
    image segmentation, which first computes the IOU for each semantic class
    and then computes the average over classes. IOU is defined as follows:
      IOU = true_positive / (true_positive + false_positive + false_negative).
    The predictions are accumulated in a confusion matrix, weighted by
    `sample_weight` and the metric is then calculated from it.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
    Usage:
    >>> # cm = [[1, 1],
    >>> #        [1, 1]]
    >>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
    >>> # iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2 = 0.33
    >>> m = tf.keras.metrics.Jaccard(num_classes=2)
    >>> _ = m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
    >>> m.result().numpy()
    0.33333334
    >>> m.reset_states()
    >>> _ = m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
    ...                    sample_weight=[0.3, 0.3, 0.3, 0.1])
    >>> m.result().numpy()
    0.23809525
    Usage with tf.keras API:
    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile(
      'sgd',
      loss='mse',
      metrics=[tf.keras.metrics.Jaccard(num_classes=2)])
    ```
    """

    def __init__(self, num_classes, name=None, dtype=None):
        """Creates a `Jaccard` instance.
        Args:
          num_classes: The possible number of labels the prediction task can
            have. This value must be provided, since a confusion matrix of
            dimension = [num_classes, num_classes] will be allocated.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super(Jaccard, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes

        # Variable to accumulate the predictions in the confusion matrix. Setting
        # the type to be `float64` as required by confusion_matrix_ops.
        self.total_cm = self.add_weight(
            "total_confusion_matrix",
            shape=(num_classes, num_classes),
            initializer=init_ops.zeros_initializer,
            dtype=dtypes.float64,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.
        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
        Returns:
          Update op.
        """
        y_true = y_true[..., 0]
        y_pred = tf.math.argmax(y_pred, axis=-1)

        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)

        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = array_ops.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = array_ops.reshape(y_true, [-1])

        if sample_weight is not None:
            sample_weight = math_ops.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = array_ops.reshape(sample_weight, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = confusion_matrix.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=dtypes.float64,
        )
        return self.total_cm.assign_add(current_cm)

    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = math_ops.cast(
            math_ops.reduce_sum(self.total_cm, axis=0), dtype=self._dtype
        )
        sum_over_col = math_ops.cast(
            math_ops.reduce_sum(self.total_cm, axis=1), dtype=self._dtype
        )
        true_positives = math_ops.cast(
            array_ops.diag_part(self.total_cm), dtype=self._dtype
        )

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col - true_positives

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = math_ops.reduce_sum(
            math_ops.cast(math_ops.not_equal(denominator, 0), dtype=self._dtype)
        )

        iou = math_ops.div_no_nan(true_positives, denominator)

        return math_ops.div_no_nan(
            math_ops.reduce_sum(iou, name="mean_jaccard"), num_valid_entries
        )

    def reset_states(self):
        K.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        config = {"num_classes": self.num_classes}
        base_config = super(Jaccard, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Dice(tf.keras.metrics.Metric):
    """Computes the mean Intersection-Over-Union metric.
    Mean Intersection-Over-Union is a common evaluation metric for semantic
    image segmentation, which first computes the IOU for each semantic class
    and then computes the average over classes. IOU is defined as follows:
      IOU = true_positive / (true_positive + false_positive + false_negative).
    The predictions are accumulated in a confusion matrix, weighted by
    `sample_weight` and the metric is then calculated from it.
    If `sample_weight` is `None`, weights default to 1.
    Use `sample_weight` of 0 to mask values.
    Usage:
    >>> # cm = [[1, 1],
    >>> #        [1, 1]]
    >>> # sum_row = [2, 2], sum_col = [2, 2], true_positives = [1, 1]
    >>> # iou = true_positives / (sum_row + sum_col - true_positives))
    >>> # result = (1 / (2 + 2 - 1) + 1 / (2 + 2 - 1)) / 2 = 0.33
    >>> m = tf.keras.metrics.Dice(num_classes=2)
    >>> _ = m.update_state([0, 0, 1, 1], [0, 1, 0, 1])
    >>> m.result().numpy()
    0.33333334
    >>> m.reset_states()
    >>> _ = m.update_state([0, 0, 1, 1], [0, 1, 0, 1],
    ...                    sample_weight=[0.3, 0.3, 0.3, 0.1])
    >>> m.result().numpy()
    0.23809525
    Usage with tf.keras API:
    ```python
    model = tf.keras.Model(inputs, outputs)
    model.compile(
      'sgd',
      loss='mse',
      metrics=[tf.keras.metrics.Dice(num_classes=2)])
    ```
    """

    def __init__(self, num_classes, name=None, dtype=None):
        """Creates a `Dice` instance.
        Args:
          num_classes: The possible number of labels the prediction task can
            have. This value must be provided, since a confusion matrix of
            dimension = [num_classes, num_classes] will be allocated.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super(Dice, self).__init__(name=name, dtype=dtype)
        self.num_classes = num_classes

        # Variable to accumulate the predictions in the confusion matrix.
        # Setting the type to be `float64` as required by confusion_matrix_ops.
        self.total_cm = self.add_weight(
            "total_confusion_matrix",
            shape=(num_classes, num_classes),
            initializer=init_ops.zeros_initializer,
            dtype=dtypes.float64,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates the confusion matrix statistics.
        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1.
            Can be a `Tensor` whose rank is either 0, or the same rank as
            `y_true`, and must be broadcastable to `y_true`.
        Returns:
          Update op.
        """
        y_true = y_true[..., 0]
        y_pred = tf.math.argmax(y_pred, axis=-1)

        y_true = math_ops.cast(y_true, self._dtype)
        y_pred = math_ops.cast(y_pred, self._dtype)

        # Flatten the input if its rank > 1.
        if y_pred.shape.ndims > 1:
            y_pred = array_ops.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = array_ops.reshape(y_true, [-1])

        if sample_weight is not None:
            sample_weight = math_ops.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = array_ops.reshape(sample_weight, [-1])

        # Accumulate the prediction to current confusion matrix.
        current_cm = confusion_matrix.confusion_matrix(
            y_true,
            y_pred,
            self.num_classes,
            weights=sample_weight,
            dtype=dtypes.float64,
        )
        return self.total_cm.assign_add(current_cm)

    def result(self):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = math_ops.cast(
            math_ops.reduce_sum(self.total_cm, axis=0), dtype=self._dtype
        )
        sum_over_col = math_ops.cast(
            math_ops.reduce_sum(self.total_cm, axis=1), dtype=self._dtype
        )
        true_positives = math_ops.cast(
            array_ops.diag_part(self.total_cm), dtype=self._dtype
        )

        # sum_over_row + sum_over_col =
        #     2 * true_positives + false_positives + false_negatives.
        denominator = sum_over_row + sum_over_col

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = math_ops.reduce_sum(
            math_ops.cast(math_ops.not_equal(denominator, 0), dtype=self._dtype)
        )

        iou = math_ops.div_no_nan(2 * true_positives, denominator)

        return math_ops.div_no_nan(
            math_ops.reduce_sum(iou, name="mean_dice"), num_valid_entries
        )

    def reset_states(self):
        K.set_value(self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        config = {"num_classes": self.num_classes}
        base_config = super(Dice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
