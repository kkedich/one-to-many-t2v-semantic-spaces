"""
Implementation adapted from LDM from https://github.com/chao-ji/ldm_tf2
"""

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv3D, Dense, Dropout, GroupNormalization

GROUP_NORM_EPS = 1e-6


logging.basicConfig(level=logging.INFO)


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        channels,
        num_groups=32,
        dropout_rate=0.,
        conv_shortcut=False,
        activation=None,
    ):
        """
        """
        super(ResidualBlock, self).__init__()
        self._channels = channels
        self._num_groups = num_groups
        self._dropout_rate = dropout_rate
        self._conv_shortcut = conv_shortcut
        if activation is None or activation == "swish":
            self._activation = tf.nn.swish
        else:
            raise ValueError("Incorrect activation function.")

        self._group_norm1 = GroupNormalization(num_groups, epsilon=GROUP_NORM_EPS)
        self._conv1 = Conv3D(channels, kernel_size=3, padding="same")
        self._group_norm2 = GroupNormalization(num_groups, epsilon=GROUP_NORM_EPS)
        self._dropout = Dropout(dropout_rate)
        self._conv2 = Conv3D(channels, kernel_size=3, padding="same")
        self._shortcut = Dense(channels)
        # if conv_shortcut:
        #     self._conv_time = Conv3D(channels, kernel_size=3, padding="same")
        # else:
        #     self._dense_time = Dense(channels)

    def call(self, inputs, time=None, training=False):
        logging.debug(f"ResidualBlock: {inputs.shape}")
        outputs = self._activation(self._group_norm1(inputs))
        outputs = self._conv1(outputs)

        if time is not None:
            raise NotImplementedError("Time is not implemented")
            # outputs += self._dense_time(
            #     self._activation(time))[:, tf.newaxis, tf.newaxis]

        outputs = self._activation(self._group_norm2(outputs))
        outputs = self._dropout(outputs, training=training)
        outputs = self._conv2(outputs)

        if inputs.shape[-1] != self._channels:
            inputs = self._shortcut(inputs)

        assert inputs.shape == outputs.shape
        outputs = outputs + inputs
        logging.debug(f"ResidualBlock: {outputs.shape}")

        return outputs


class AttentionBlock(tf.keras.layers.Layer):
    """Attention block."""

    def __init__(self, channels, num_groups=32):
        super(AttentionBlock, self).__init__()
        self._channels = channels
        self._num_groups = num_groups

        self._group_norm = GroupNormalization(num_groups, epsilon=GROUP_NORM_EPS)
        self._dense_query = Dense(channels)
        self._dense_key = Dense(channels)
        self._dense_value = Dense(channels)
        self._dense_output = Dense(channels)

    def call(self, inputs):
        batch_size, frames, height, width, channels = inputs.shape
        outputs = self._group_norm(inputs)

        logging.debug(f"Attention, outputs.shape before reshape: {outputs.shape}")
        outputs = tf.reshape(outputs, [batch_size * frames, height, width, channels])
        logging.debug(f"Attention, outputs.shape after reshape: {outputs.shape}")

        # Size for q, k, and v [batch_size, height, width, channels]
        q = self._dense_query(outputs)
        k = self._dense_key(outputs)
        v = self._dense_value(outputs)

        # [batch_size, height, width, height, width]
        attention_weights = tf.einsum("bhwc,bHWc->bhwHW", q, k) * self._channels ** -0.5
        attention_weights = tf.reshape(
            attention_weights, [batch_size, height, width, height * width])
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)
        attention_weights = tf.reshape(
            attention_weights, [batch_size, height, width, height, width])
        outputs = tf.einsum("bhwHW,bHWc->bhwc", attention_weights, v)
        outputs = self._dense_output(outputs)
        # print(f"attention_weights 5: {attention_weights.shape}")

        outputs = tf.reshape(outputs, [batch_size, frames, height, width, channels])
        logging.debug(f"Attention, outputs.shape after reshape: {outputs.shape}")

        assert inputs.shape == outputs.shape
        outputs = outputs + inputs

        return outputs


class DownBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        channels,
        attention_resolutions,
        dropout_rate=0.,
    ):
        super(DownBlock, self).__init__()
        self._channels = channels
        self._attention_resolutions = attention_resolutions
        self._dropout_rate = dropout_rate

        self._residual = ResidualBlock(channels, dropout_rate=dropout_rate)
        self._attention = AttentionBlock(channels)

    def call(self, inputs, time=None, training=False):
        logging.debug(f"Downblock: {inputs.shape}")
        outputs = self._residual(inputs, time, training=training)
        if outputs.shape[1] in self._attention_resolutions:
            outputs = self._attention(outputs)
        return outputs


class Downsample(tf.keras.layers.Layer):
    def __init__(self, channels, resample_with_conv=True):
        super(Downsample, self).__init__()
        self._channels = channels
        self._resample_with_conv = resample_with_conv

        if resample_with_conv:
            self._conv = Conv3D(channels, kernel_size=3, strides=2, padding="VALID")

    def call(self, inputs, training=False):
        logging.debug(f"Downsample: {inputs.shape}")
        if self._resample_with_conv:
            outputs = tf.pad(inputs, [[0, 0], [0, 0], [0, 1], [0, 1], [0, 0]])
            outputs = self._conv(outputs)
        else:
            outputs = tf.nn.avg_pool3d(inputs, ksize=2, strides=2, padding="VALID")
        return outputs


class MiddleBlock(tf.keras.layers.Layer):
    def __init__(self, channels, dropout_rate=0.):
        super(MiddleBlock, self).__init__()
        self._channels = channels
        self._dropout_rate = dropout_rate

        self._residual1 = ResidualBlock(channels, dropout_rate=dropout_rate)
        self._attention = AttentionBlock(channels)
        self._residual2 = ResidualBlock(channels, dropout_rate=dropout_rate)

    def call(self, inputs, time=None, training=False):
        logging.debug(f"MiddleBlock: {inputs.shape}")
        outputs = self._residual1(inputs, time, training=training)
        outputs = self._attention(outputs)
        outputs = self._residual2(outputs, time, training=training)

        return outputs


class Encoder(tf.keras.layers.Layer):
    def __init__(
            self,
            channels,
            num_blocks=2,
            latent_channels=4,
            attention_resolutions=(),
            dropout_rate=0.0,
            multipliers=(1, 2, 4, 8),
            resample_with_conv=True,
            activation=None,
    ):
        super(Encoder, self).__init__()
        self._channels = channels
        self._num_blocks = num_blocks
        self._latent_channels = latent_channels
        self._attention_resolutions = attention_resolutions
        self._dropout_rate = dropout_rate
        self._multipliers = multipliers
        self._resample_with_conv = resample_with_conv
        if activation is None or activation == "swish":
            self._activation = tf.nn.swish
        else:
            raise ValueError("Incorrect activation function.")

        self._conv_in = tf.keras.layers.Conv3D(channels, kernel_size=3, padding="SAME")

        num_resolutions = len(multipliers)
        channels_list = [channels * mul for mul in multipliers]

        down_blocks = []
        for i in range(num_resolutions):
            for j in range(num_blocks):
                down_blocks.append(DownBlock(channels_list[i], attention_resolutions, dropout_rate))
            if i < num_resolutions - 1:
                down_blocks.append(Downsample(channels_list[i], resample_with_conv))

        self._down = down_blocks
        self._middle = MiddleBlock(channels_list[-1])

        self._group_norm = GroupNormalization(32, epsilon=GROUP_NORM_EPS)
        self._conv_out = Conv3D(latent_channels, kernel_size=3, padding="SAME")

        self._quant_conv = Dense(latent_channels)

    def call(self, inputs, training=False):
        logging.debug(f"Encoder, inputs={inputs.shape}")
        outputs = self._conv_in(inputs)

        for current_block in self._down:
            outputs = current_block(outputs, training=training)

        outputs = self._middle(outputs, training=training)
        outputs = self._conv_out(self._activation(self._group_norm(outputs)))
        logging.debug(outputs.shape)

        outputs = self._quant_conv(outputs)
        mean, logvar = tf.split(outputs, 2, axis=-1)
        logging.debug(f"End encoder: mean={mean.shape}, logvar={logvar.shape}\n")

        return [mean, logvar]


class UpBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        channels,
        attention_resolutions,
        dropout_rate=0.,
    ):
        super(UpBlock, self).__init__()
        self._channels = channels
        self._attention_resolutions = attention_resolutions
        self._dropout_rate = dropout_rate

        self._residual = ResidualBlock(channels, dropout_rate=dropout_rate)
        self._attention = AttentionBlock(channels)

    def call(self, inputs, time=None, training=False):
        logging.debug(f"UpBlock, inputs={inputs.shape}")
        outputs = self._residual(inputs, time, training=training)
        if outputs.shape[1] in self._attention_resolutions:
            outputs = self._attention(outputs)
        return outputs


class Upsample(tf.keras.layers.Layer):
    def __init__(self, channels, resample_with_conv=True):
        super(Upsample, self).__init__()
        self._channels = channels
        self._resample_with_conv = resample_with_conv
        self._transp_conv = tf.keras.layers.Conv3DTranspose(channels, kernel_size=3,
                                                            strides=[2, 2, 2], padding="SAME")

        if resample_with_conv:
            self._conv = Conv3D(channels, kernel_size=3, strides=1, padding="SAME")

    def call(self, inputs, time=None, training=False):
        # batch_size, frames, height, width, channels = inputs.shape
        logging.debug(f"Upsample, {inputs.shape}")

        # inputs = tf.reshape(inputs, [batch_size * frames, height, width, channels])
        # outputs = tf.raw_ops.ResizeNearestNeighbor(
        #     images=inputs, size=[height * 2, width * 2], align_corners=False)
        # print(f"Size outputs after resize: {outputs.shape}")
        # outputs = tf.reshape(outputs, [batch_size, frames, height * 2, width * 2, channels])
        # print(f"Size outputs after reshape: {outputs.shape}")
        outputs = self._transp_conv(inputs)
        # print(f"Transposed3DConv: {outputs.shape}")

        if self._resample_with_conv:
            outputs = self._conv(outputs)
        logging.debug(f"Upsample, outputs: {outputs.shape}\n")

        return outputs


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        channels,
        latent_channels,
        out_channels=3,
        num_blocks=2,
        dropout_rate=0.0,
        resample_with_conv=True,
        attention_resolutions=(),
        multipliers=(1, 2, 4, 8),
        activation=None,
    ):
        super(Decoder, self).__init__()
        self._channels = channels
        self._out_channels = out_channels
        self._num_blocks = num_blocks
        self._dropout_rate = dropout_rate
        self._resample_with_conv = resample_with_conv
        self._attention_resolutions = attention_resolutions
        self._multipliers = multipliers
        if activation is None or activation == "swish":
            self._activation = tf.nn.swish
        else:
            raise ValueError("Incorrect activation function.")

        channels_list = [channels * mul for mul in multipliers]
        self._conv_in = Conv3D(channels_list[-1], kernel_size=3, padding="SAME")
        self._middle = MiddleBlock(channels_list[-1])

        self._transp_conv = tf.keras.layers.Conv3DTranspose(
            channels_list[-1], kernel_size=3, strides=[2, 1, 1], padding="SAME")

        up = []
        num_resolutions = len(multipliers)
        for i in reversed(range(num_resolutions)):
            for j in range(num_blocks + 1):
                up.append(
                    UpBlock(channels_list[i], attention_resolutions, dropout_rate))
            if i > 0:
                up.append(Upsample(channels_list[i], resample_with_conv))
        self._up = up

        self._group_norm = GroupNormalization(32, epsilon=GROUP_NORM_EPS)
        self._conv_out = Conv3D(out_channels, kernel_size=3, padding="SAME")

        self._post_quant_conv = Dense(latent_channels)
        self._tanh_layer = tf.tanh

    def call(self, inputs, training=False):
        logging.debug(f"Decoder: {inputs.shape}")
        outputs = self._post_quant_conv(inputs)
        logging.debug(f"Shape post quant: {outputs.shape}")
        outputs = self._conv_in(outputs)
        outputs = self._middle(outputs, training=training)
        outputs = self._transp_conv(outputs)

        for m in self._up:
            outputs = m(outputs, training=training)
        outputs = self._conv_out(self._activation(self._group_norm(outputs)))

        outputs = self._tanh_layer(outputs)
        logging.debug(f"Decoder outputs: {outputs.shape}\n")

        return outputs


class DiagonalGaussian(object):
    def __init__(self, mean, logvar, deterministic=False):
        """
        Args:
          mean (Tensor of shape [batch_size, z_height, z_width, z_channels]): the
            predicted mean of the latent variable.
          logvar (Tensor of shape [batch_size, z_height, z_width, z_channels]): the
            predicted logvar of the latent variable.
        """
        self._mean = mean
        self._logvar = tf.clip_by_value(logvar, -30.0, 20.0)
        self._deterministic = deterministic
        self._std = tf.exp(0.5 * logvar)
        self._var = tf.exp(logvar)
        if deterministic:
            self._var = self._std = tf.zeros_like(self._mean)

    def sample(self):
        outputs = self._mean + self._std * tf.random.normal(self._std.shape)
        return outputs

    def kl(self, other=None):
        if self._deterministic:
            return tf.constant([0.], dtype="float32")
        else:
            if other is None:
                return 0.5 * tf.reduce_sum(tf.pow(self._mean, 2)
                                           + self._var - 1.0 - self._logvar,
                                           axis=[1, 2, 3])
            else:
                return 0.5 * tf.reduce_sum(
                    tf.pow(self._mean - other._mean, 2) / other._var
                    + self._var / other._var - 1.0 - self._logvar + other._logvar,
                    axis=[1, 2, 3])

    def nll(self, sample, axis=[1, 2, 3]):
        if self._deterministic:
            return tf.constant([0.], dtype="float32")

        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * tf.reduce_sum(
            logtwopi + self._logvar + tf.pow(sample - self._mean, 2) / self._var,
            axis=axis)

    def mode(self):
        return self._mean
