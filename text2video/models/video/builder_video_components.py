import tensorflow as tf

from text2video.models.video.autoencoder_kl_ldm import AutoencoderKL
from text2video.models.video.c3d_discriminator import Discriminator
from text2video.models.video.c3d_wae_decoder import Decoder
from text2video.models.video.c3d_wae_encoder import Encoder
from text2video.models.video.patchhd_discriminator import \
    PatchDiscriminatorHDVideo


def video_shape(options_input):
    # video_shape = [16, 64, 64, 1]
    return [options_input.sequence_length,
            options_input.img_width,
            options_input.img_height,
            options_input.channels]


def create_generator(name=None, options=None):

    if name == "baseline":
        video_encoder = Encoder(
            video_shape=video_shape(options.input),
            code_dimension=options.code_dimension,
            momentum_bn=options.momentum_bn,
            act_function_name=options.activation_function.type,
            lrelu_alpha=options.activation_function.lrelu_alpha,
            kernel_init_name="normal",
            stddev=0.009,
            using_strategy=options.get("using_strategy", False),
            norm_type=options.norm_type
        )
        video_encoder, shape_last_conv_block = video_encoder.build_model(options.batch_size)
        video_encoder.summary()

        video_decoder = Decoder(
            output_channels=options.input.channels,
            code_dimension=options.code_dimension,
            momentum_bn=options.momentum_bn,
            act_function_name=options.activation_function.type,
            lrelu_alpha=options.activation_function.lrelu_alpha,
            kernel_init_name="normal",
            stddev=0.009,
            dropout_rate=options.dropout_rate,  # Original test was considering no dropout
            using_strategy=options.get("using_strategy", False),
            norm_type=options.norm_type
        )
        video_decoder = video_decoder.build_model(options.batch_size)
        video_decoder.summary()

        return [video_encoder, video_decoder]

    elif name == "unet3d_ldm":
        autoencoder = AutoencoderKL(
            channels=options.channels,
            num_blocks=options.num_blocks,
            latent_channels=options.latent_channels * 2,
            attention_resolutions=options.get("attention_resolutions", ()),
            dropout_rate=options.dropout_rate,
            multipliers=options.multipliers,
            resample_with_conv=options.resample_with_conv,
            out_channels=options.input.channels
        )

        return autoencoder
    else:
        raise ValueError(f"Video generator model <{name}> is not a valid option!")


def create_discriminator(name=None, options=None):

    if name == "baseline":
        video_discriminator = Discriminator(
            video_shape=video_shape(options.input),
            features_dimension=options.generator.code_dimension,
            momentum_bn=options.momentum_bn,
            act_function_name=options.discriminator.activation_function.type,
            lrelu_alpha=options.discriminator.activation_function.lrelu_alpha,
            kernel_init_name="normal",
            stddev=0.009,
            dropout_rate=options.discriminator.dropout_rate
        )
        video_discriminator = video_discriminator.build_model()
        video_discriminator.summary()

    elif name == "patch_hd":
        video_discriminator = PatchDiscriminatorHDVideo(
            vid_shape=video_shape(options.input),
            dropout=0.1,
            kernel_initializer=tf.random_normal_initializer(stddev=0.02),
            act_function="leaky_relu",
            lrelu_alpha=0.2,
            type_norm_layer="instance_norm",
        )
        video_discriminator = video_discriminator.build_model()
        video_discriminator.summary()
    else:
        raise ValueError(f"Video discriminator model <{name}> is not a valid option!")

    return video_discriminator
