import tensorflow as tf

from text2video.models.video.unet3d_ldm import Decoder
from text2video.models.video.unet3d_ldm import Encoder


class AutoencoderKL(tf.keras.layers.Layer):
    def __init__(
        self,
        latent_channels=4,
        channels=128,
        num_blocks=2,
        attention_resolutions=(),
        dropout_rate=0.,
        multipliers=(1, 2, 4, 4),
        resample_with_conv=True,
        out_channels=3
    ):
        super(AutoencoderKL, self).__init__()
        self._latent_channels = latent_channels
        self._channels = channels
        self._num_blocks = num_blocks
        self._attention_resolutions = attention_resolutions
        self._dropout_rate = dropout_rate
        self._multipliers = multipliers
        self._resample_with_conv = resample_with_conv

        self._encoder = Encoder(
            channels=channels,
            num_blocks=num_blocks,
            latent_channels=latent_channels * 2,
            attention_resolutions=(),
            dropout_rate=dropout_rate,
            multipliers=multipliers,
            resample_with_conv=resample_with_conv,
            activation="swish",
        )
        # self._quant_conv = Dense(latent_channels * 2)
        # self._post_quant_conv = Dense(latent_channels)
        self._decoder = Decoder(
            latent_channels=latent_channels,
            channels=channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            multipliers=multipliers,
            resample_with_conv=resample_with_conv,
            attention_resolutions=(),
            dropout_rate=dropout_rate,
            activation="swish",
        )

    # def call(self, inputs, sample_posterior=True, training=False):
    #     # posterior = self.encode(inputs, training=training)
    #     mean, logvar = self._encoder(inputs, training=training)
    #     # outputs = self._quant_conv(outputs)
    #     # mean, logvar = tf.split(outputs, 2, axis=-1)
    #     posterior = DiagonalGaussian(mean, logvar)
    #
    #     if sample_posterior:
    #         latents = posterior.sample()
    #     else:
    #         latents = posterior.mode()
    #     # print(f"Latents: {latents.shape}")
    #
    #     # outputs = self.decode(latents, training=training)
    #     # outputs = self._post_quant_conv(inputs)
    #     outputs = self._decoder(inputs, training=training)
    #
    #     return outputs, posterior

    def get_last_layer(self):
        return self._decoder._conv_out.weights[0]
