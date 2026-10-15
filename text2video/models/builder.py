import tensorflow as tf

from text2video.models.vae_ops import reparameterization_trick
from text2video.models.video.builder_video_components import create_generator
from text2video.utils.config import Config


def build_opt(cfg_optimizer):
    """Creates a Tensorflow optimizer with some requirements"""

    global_clipnorm = cfg_optimizer.get("global_clipnorm", None)
    lr_decayed_fn = cfg_optimizer.learning_rate

    if cfg_optimizer.get("cosine_decay", False):
        warmup_steps = cfg_optimizer.get("warmup_steps", -1)
        initial_learning_rate = cfg_optimizer.learning_rate

        if warmup_steps > 0:
            warmup_target = cfg_optimizer.get("warmup_target", initial_learning_rate)

            lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate,
                cfg_optimizer.get("decay_steps", 1000),
                warmup_target,
                warmup_steps
            )
        else:
            lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate,
                cfg_optimizer.get("decay_steps", 1000)
            )

    type_optimizer = cfg_optimizer.get("type_optimizer", "adam")

    if type_optimizer == "adam":
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_decayed_fn,
            beta_1=cfg_optimizer.get("beta_1", 0.9),
            global_clipnorm=global_clipnorm
            # decay=cfg_optimizer.get("decay", 0.0)
        )
    elif type_optimizer == "adamw":
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_decayed_fn,
            beta_1=cfg_optimizer.get("beta_1", 0.9),
            global_clipnorm=global_clipnorm
        )
    else:
        raise ValueError(f"Unknown optimizer type: {type_optimizer}")

    return optimizer


def create_generator_model(cfg_experiment, batch_size=None, separate_enc_dec=False):

    options = cfg_experiment.model_video
    if batch_size is None:
        batch_size = cfg_experiment.train_cfg.batch_size
    print(f"Batch size: {batch_size}")

    # Common parameters
    options_generator = options.generator
    options_generator.momentum_bn = 0.9
    options_generator.input = options.input
    options_generator.batch_size = batch_size

    model_name = options.generator.get("name", "baseline")

    if model_name == "baseline":
        video_encoder, video_decoder = create_generator(name=model_name, options=options_generator)

        # mean and sigma from distribution
        z_mu, z_log_var = video_encoder.output
        video_code = reparameterization_trick(z_mu, z_log_var)

        decoded_video, logits = video_decoder(video_code)

        if not separate_enc_dec:
            model_full = tf.keras.Model(inputs=[video_encoder.input],
                                        outputs=[z_mu, z_log_var, video_code,
                                                 decoded_video, logits],
                                        name="video_generator")
            model_full.summary()

            return video_decoder, model_full
        else:
            video_encoder = tf.keras.Model(
                inputs=[video_encoder.input],
                outputs=[z_mu, z_log_var, video_code],
                name="encoder_wae"
            )

            return video_encoder, video_decoder

    elif model_name == "unet3d_ldm":
        unet3d_ldm_model = create_generator(name=model_name, options=options_generator)

        video_shape = [
            options.input.sequence_length,
            options.input.img_width,
            options.input.img_height,
            options.input.channels
        ]

        print(f"My final batch shape: {(batch_size,) + tuple(video_shape)}")
        input_video = tf.keras.layers.Input(shape=video_shape,
                                            batch_size=batch_size,
                                            name="encoder_input_video")
        print(f"Input shape: {input_video.shape}")

        z_mu, z_log_var = unet3d_ldm_model._encoder(input_video)
        video_code = reparameterization_trick(z_mu, z_log_var)

        if separate_enc_dec:
            # Create a model for the encoder and decoder
            video_encoder = tf.keras.Model(
                inputs=[input_video],
                outputs=[z_mu, z_log_var, video_code],
                name="video_encoder"
            )
            video_encoder.summary()

            video_code_shape = list(video_encoder.output_shape[0][1:])
            print(f"video_code_shape: {video_code_shape}, {video_code_shape[1:]}")
            input_video_code = tf.keras.layers.Input(shape=video_code_shape,
                                                     batch_size=batch_size,
                                                     name="decoder_input_code")
            print(f"input_video_code: {input_video_code.shape}")

            decoded_video = unet3d_ldm_model._decoder(input_video_code)
            video_decoder = tf.keras.Model(
                inputs=[input_video_code],
                outputs=[decoded_video],
                name="video_decoder"
            )
            video_decoder.summary()

            return video_encoder, video_decoder
        else:
            decoded_video = unet3d_ldm_model._decoder(video_code)

            model_full = tf.keras.Model(inputs=[input_video],
                                        outputs=[z_mu, z_log_var, video_code, decoded_video],
                                        name="video_generator")
            model_full.summary()

            return None, model_full

    else:
        raise ValueError(f"Invalid model name type: {model_name}")


def create_video_decoder(cfg_experiment: Config, video_code_dimension: int, input_video_code_shape: list, batch_size: int = None):
    # Build a video decoder model only
    # Create full model first to get the decoder part
    options_generator = cfg_experiment.model_video.generator
    options_generator.momentum_bn = 0.9
    options_generator.input = cfg_experiment.model_video.input
    options_generator.batch_size = batch_size

    unet3d_ldm_model = create_generator(name="unet3d_ldm", options=options_generator)

    print("Creating video decoder model...")
    # Creating the video decoder model, example from (batch_size, 1, 8, 8, 8)
    input_video_code = tf.keras.layers.Input(shape=[video_code_dimension],
                                             batch_size=batch_size,
                                             name="decoder_input_code")
    print(f"input_video_code: {input_video_code.shape}")

    flattened_tensor2input = tf.reshape(input_video_code, shape=input_video_code_shape)
    dec_decoded_video = unet3d_ldm_model._decoder(flattened_tensor2input)

    video_decoder = tf.keras.Model(
        inputs=[input_video_code],
        outputs=[dec_decoded_video],
        name="video_decoder"
    )
    video_decoder.summary()

    return video_decoder, unet3d_ldm_model


def create_video_ldm(cfg_experiment, unet3d_ldm_model, batch_size):
    options = cfg_experiment.model_video

    video_shape = [
        options.input.sequence_length,
        options.input.img_width,
        options.input.img_height,
        options.input.channels
    ]

    print(f"My final batch shape: {(batch_size,) + tuple(video_shape)}")
    input_video = tf.keras.layers.Input(shape=video_shape,
                                        batch_size=batch_size,
                                        name="encoder_input_video")
    print(f"Input shape: {input_video.shape}")

    z_mu, z_log_var = unet3d_ldm_model._encoder(input_video)
    # # To flatten version
    # flattened_z_mu = tf.reshape(z_mu, shape=(batch_size, -1))
    # flattened_z_log_var = tf.reshape(z_log_var, shape=(batch_size, -1))

    video_code = reparameterization_trick(z_mu, z_log_var)
    flattened_video_code = tf.reshape(video_code, shape=(batch_size, -1))

    decoded_video = unet3d_ldm_model._decoder(video_code)

    model_full = tf.keras.Model(
        inputs=[input_video],
        outputs=[z_mu, z_log_var, flattened_video_code, decoded_video],
        name="video_generator"
    )
    model_full.summary()

    return model_full
