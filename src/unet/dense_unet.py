"""Provide sematic segmentation through the U-Net architecture.

Heavily based on the code repository used to produce:
    `J. Akeret, C. Chang, A. Lucchi, A. Refregier, Published in Astronomy
    and Computing (2017) <https://arxiv.org/abs/1609.09077>`_

which is hosted at:
    https://github.com/jakeret

Raises:
    NotImplementedError: When an unknown normalization scheme is called.
"""
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from autosegment.unet.metrics import dice_coefficient, mean_iou
from tensorflow.keras import Input, Model, layers, losses, regularizers
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam


class ContractionLayer(layers.Layer):
    """Represent the contraction layer of the U-Net architecture (downwards).

    Utilizes two sequential convolutions followed by normalization and dropout
    before activation.

    Inherits:
        layers.Layer: The layer superclass from the Keras API in Tensorflow.
    """

    def __init__(
        self,
        block_idx,
        base_filters,
        kernel_size,
        dropout_rate,
        padding,
        activation,
        normalization,
        **kwargs,
    ):
        """Construct a U-Net ContractionLayer.

        Select the parameters of the U-Net layer.

        Args:
            block_idx (int): The block index used for connecting the expansive
                layers.
            base_filters (int): The initial resulting number of image channels
                while downsampling.
            kernel_size (int): The size of the 'window' around each pixel under
                inspection, which moves across a grid to inspect the entire
                image. dropout_rate (float): Rate of dropout regularization
                to randomly
                remove nodes. Prevents overfitting and improves generalization.
            padding (str): Designates how to pad the outer pixels of the image,
                lost while downsampling.
            activation (str): Final activation of the layer, such as
                rectified linear units: ReLU.
            normalization (str): Type of normalization to be used in the model,
                such as instance normalization.
        """
        super(ContractionLayer, self).__init__(**kwargs)
        self.block_idx = block_idx
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.padding = padding
        self.activation = activation
        self.normalization = _get_normalization(normalization, axis=-1)

        n_filters = _current_filter_count(block_idx, base_filters)
        self.conv2d = layers.Conv2D(
            filters=n_filters,
            kernel_size=(kernel_size, kernel_size),
            kernel_initializer="he_normal",
            kernel_regularizer=_get_kernel_regularizer(),
            strides=1,
            padding=padding,
        )
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.activation = layers.Activation(activation)

    def call(self, inputs, training=None, **kwargs):
        """Call the layer using Keras' functional API.

        Begin with convolution, followed by optional normalization, dropout
        when training, and finally activation.

        Args:
            inputs (tensor): Output of the previous layer, or initial input
                layer.
            training (bool, optional): Internal flag used by tensorflow to
                determine if the model is being trained. Defaults to None.

        Returns:
            tensor: The output tensor.
        """
        x = inputs
        x = self.conv2d(x)
        if self.normalization:
            x = self.normalization(x)

        if training:
            x = self.dropout(x)

        x = self.activation(x)

        return x

    def get_config(self):
        return dict(
            block_idx=self.block_idx,
            base_filters=self.base_filters,
            kernel_size=self.kernel_size,
            dropout_rate=self.dropout_rate,
            padding=self.padding,
            activation=self.activation,
            normalization=self.normalization,
            **super(ContractionLayer, self).get_config(),
        )


class ExpansionLayer(layers.Layer):
    def __init__(
        self,
        block_idx,
        base_filters,
        kernel_size,
        pool_size,
        padding,
        activation,
        **kwargs,
    ):
        """Construct a U-Net ExpansionLayer.

        Select the parameters of the U-Net layer. Many of these must match the
        connecting contraction layer.

        Args:
            block_idx (int): The block index used for connecting the
                contractive layers.
            base_filters (int): The initial resulting number of image channels
                while downsampling.
            kernel_size (int): The size of the 'window' around each pixel under
                inspection, which moves across a grid to inspect the entire
                image. dropout_rate (float): Rate of dropout regularization
                to randomly remove nodes. Prevents overfitting and improves
                generalization.
            pool_size (int): Aggregrate the convolved features over a pool_size
                x pool_size region, taking the maximum value of the region.
            padding (str): Designates how to pad the outer pixels of the image,
                lost while downsampling.
            activation (str): Final activation of the layer, such as
                rectified linear units: ReLU.
        """
        super(ExpansionLayer, self).__init__(**kwargs)
        self.block_idx = block_idx
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.padding = padding
        self.activation = activation

        n_filters = _current_filter_count(block_idx, base_filters)
        self.conv2d_transpose = layers.Conv2DTranspose(
            filters=n_filters,
            kernel_size=(pool_size, pool_size),
            kernel_initializer="he_normal",
            kernel_regularizer=_get_kernel_regularizer(),
            strides=pool_size,
            padding=padding,
        )

        self.activation = layers.Activation(activation)

    def call(self, inputs, **kwargs):
        """Call the layer using Keras' functional API.

        Begin with convolution transposition, then activation.

        Args:
            inputs (tensor): Output of the previous layer, or initial input
                layer.

        Returns:
            tensor: The output tensor.
        """
        x = inputs
        x = self.conv2d_transpose(x)
        x = self.activation(x)
        return x

    def get_config(self):
        return dict(
            block_idx=self.block_idx,
            base_filters=self.base_filters,
            kernel_size=self.kernel_size,
            pool_size=self.pool_size,
            padding=self.padding,
            activation=self.activation,
            **super(ExpansionLayer, self).get_config(),
        )


class ConcatLayer(layers.Layer):
    def call(self, x, down_layer, crop=False, **kwargs):
        if crop:
            x1_shape = tf.shape(down_layer)
            x2_shape = tf.shape(x)

            height_diff = (x1_shape[1] - x2_shape[1]) // 2
            width_diff = (x1_shape[2] - x2_shape[2]) // 2

            down_layer_cropped = down_layer[
                :,
                height_diff: (x1_shape[1] - height_diff),
                width_diff: (x1_shape[2] - width_diff),
                :,
            ]
            x = tf.concat([down_layer_cropped, x], axis=-1)
        else:
            x = layers.concatenate([down_layer, x], axis=-1)
        return x


def build_model(
    nx=None,
    ny=None,
    channels=1,
    num_classes=2,
    layer_depth=5,
    base_filters=64,
    kernel_size=3,
    pool_size=2,
    dropout_rate=0.5,
    padding="valid",
    activation="relu",
    normalization=None,
    crop=False,
    final_activation=None,
    oliver_style=False,
):
    """Construct a U-Net model."""
    if oliver_style:
        nx = 384
        ny = 384
        channels = 1
        num_classes = 2
        layer_depth = 4
        base_filters = 32
        kernel_size = 5
        pool_size = 2
        dropout_rate = 0.05
        padding = "same"
        activation = "relu"
        final_activation = final_activation or "sigmoid"
        normalization = tfa.layers.InstanceNormalization()
        crop = False

    inputs = Input(shape=(nx, ny, channels), name="inputs")  # nx, ny, channels

    x = inputs
    contracting_blocks = {}
    contract_params = dict(
        base_filters=base_filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        padding=padding,
        activation=activation,
        normalization=normalization,
    )

    # --Contractive Path--
    blocks = np.arange(0, layer_depth - 1)
    for block_idx in blocks:
        x = ContractionLayer(block_idx, **contract_params)(x)
        x = ContractionLayer(block_idx, **contract_params)(x)
        contracting_blocks[block_idx] = x
        x = layers.MaxPooling2D((pool_size, pool_size))(x)

    # --Lateral Path--
    x = ContractionLayer(blocks[-1] + 1, **contract_params)(x)
    x = ContractionLayer(blocks[-1] + 1, **contract_params)(x)

    # --Expansive Path--
    expand_params = contract_params.copy()
    expand_params.pop("dropout_rate")
    expand_params.pop("normalization")
    expand_params["pool_size"] = pool_size
    for block_idx in np.flip(blocks):
        x = ExpansionLayer(block_idx, **expand_params)(x)
        x = ConcatLayer()(x, contracting_blocks[block_idx], crop=crop)
        x = ContractionLayer(block_idx, **contract_params)(x)
        x = ContractionLayer(block_idx, **contract_params)(x)

    # --Final Activation--
    if oliver_style:
        x = layers.Conv2D(filters=1, kernel_size=(1, 1))(x)
        outputs = layers.Activation(final_activation, name="outputs")(x)
    else:
        kernel_init = _get_kernel_initializer(base_filters, kernel_size)
        x = layers.Conv2D(
            filters=num_classes,
            kernel_size=(1, 1),
            kernel_initializer=kernel_init,
            strides=1,
            padding=padding,
        )(x)
        x = layers.Activation(activation)(x)
        outputs = layers.Activation(final_activation, name="outputs")(x)

    model = Model(inputs, outputs, name="unet")
    return model


def _get_normalization(normalization, *args, **kwargs):
    if normalization is None:
        return None
    elif isinstance(normalization, tfa.layers.InstanceNormalization):
        return tfa.layers.InstanceNormalization(*args, **kwargs)
    else:
        msg = f"Unknown normalization type {normalization}"
        raise NotImplementedError(msg)


def _current_filter_count(layer_idx, base_filters):
    return 2 ** layer_idx * base_filters


def _get_kernel_regularizer(l1=1e-8, l2=1e-8):
    return regularizers.l1_l2(l1=l1, l2=l2)


def _get_kernel_initializer(filters, kernel_size):
    stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
    return TruncatedNormal(stddev=stddev)


def finalize_model(
    model,
    loss=losses.categorical_crossentropy,
    optimizer=None,
    metrics=None,
    oliver_style=False,
    **opt_kwargs,
):
    """Configure the model for training by setting loss, optimizer, metrics."""
    if oliver_style:
        optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        # loss = losses.binary_crossentropy
        loss = "binary_crossentropy"
        metrics = ["acc"]

    if optimizer is None:
        optimizer = Adam(**opt_kwargs)

    if metrics is None:
        metrics = [
            "categorical_crossentropy",
            "categorical_accuracy",
        ]

    metric_dict = dict(
        mean_iou=mean_iou, dice_coefficient=dice_coefficient, auc=tf.keras.metrics.AUC()
    )
    string_metrics = [s.lower() for s in metrics if isinstance(s, str)]
    metrics = [m for m in metrics if not isinstance(m, str)]
    for key, val in metric_dict.items():
        if key in string_metrics:
            metrics += [val]

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
    )
