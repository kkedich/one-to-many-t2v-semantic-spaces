import copy
import numpy as np
import re
import tensorflow as tf
from pathlib import Path

from text2video.models.builder import (build_opt, create_generator_model,
                                       create_video_decoder, create_video_ldm)
from text2video.models.vae_ops import reparameterization_trick
from text2video.models.video.builder_video_components import (
    create_discriminator, create_generator)
from text2video.trainer.checkpoint import load_unique_layer, load_weights_from


def create_models(cfg_experiment, batch_size=None):
    # Creating the generator and disc. models
    video_decoder, video_generator = create_generator_model(cfg_experiment, batch_size)
    video_discriminator = create_discriminator(
        name=cfg_experiment.model_video.discriminator.get("model_type", "patch_hd"),
        options=cfg_experiment.model_video
    )

    generator_optimizer = build_opt(cfg_experiment.model_video.generator.optimizer)
    discriminator_optimizer = build_opt(cfg_experiment.model_video.discriminator.optimizer)

    video_generator.compiled_metrics = None
    if video_decoder is not None:
        video_decoder.compiled_metrics = None
    video_discriminator.compiled_metrics = None

    models = {
        "generator": video_generator,
        "discriminator": video_discriminator
    }
    if video_decoder is not None:
        models["decoder"] = video_decoder

    optimizers = {
        "generator": generator_optimizer,
        "discriminator": discriminator_optimizer
    }

    return models, optimizers


def instantiate_video_enc_dec_sep_model(
        cfg_model,
        batch_size: int,
        dir_experiment: str,
        model_name: str
):
    # We need to build the models first and load as the batch size could be different
    tf.keras.backend.set_floatx("float32")
    print("\n###########################################################")
    print("Loading video model with encoder and decoder separated...")

    video_encoder, video_decoder, video_code_shape = None, None, None
    print(f"Model: {model_name}")

    print("Creating video encoder and decoder separately")
    # Considering a separated model
    video_encoder, video_decoder = create_generator_model(
        cfg_model,
        batch_size,
        separate_enc_dec=True
    )

    if model_name == "unet3d_ldm":
        # Loading entire video model as we are not training the encoder and decoder separately for this model
        print("Loading entire video generator pre-trained model...")
        video_generator = tf.keras.models.load_model(f"{dir_experiment}/video_generator_model")

        print("Loading weights of pre-trained model...")
        # We load only one layer because the encoder and decoder are in just one layer
        # Load weights from full model
        load_unique_layer(video_generator, video_encoder, target_layer_name="encoder")
        load_unique_layer(video_generator, video_decoder, target_layer_name="decoder")
        print("Loading weights of pre-trained model: done!")

    elif model_name == "baseline":
        print("Loading pre-trained models for encoder and decoder...")
        pretrained_video_decoder = tf.keras.models.load_model(
            f"{dir_experiment}/video_decoder_model")
        pretrained_video_encoder = tf.keras.models.load_model(
            f"{dir_experiment}/video_encoder_model")

        # Load weights
        load_weights_from(pretrained_video_encoder, video_encoder)
        load_weights_from(pretrained_video_decoder, video_decoder)

    if video_code_shape is None:
        # Ignoring batch size part
        video_code_shape = video_decoder.input_shape[1:]

    print("###########################################################\n")

    video_encoder.compiled_metrics = None
    video_decoder.compiled_metrics = None

    return video_encoder, video_decoder, video_code_shape


def instantiate_video_enc_dec_model(
        cfg_model,
        batch_size: int,
        dir_experiment: str,
        from_scratch: bool = False
):
    tf.keras.backend.set_floatx('float32')
    print("###########################################################")
    print("Loading video model...")

    model_name = cfg_model.model_video.generator.get("name", "baseline")
    video_generator = None
    if not from_scratch:
        video_generator = tf.keras.models.load_model(f"{dir_experiment}/video_generator_model")
        video_generator.summary()

    video_encoder, video_decoder, video_code_shape = None, None, None
    print(f"Model: {model_name}")

    if model_name == "unet3d_ldm":
        # Considering a separated model
        video_encoder, video_decoder = create_generator_model(
            cfg_model,
            batch_size,
            separate_enc_dec=True
        )

        print("Loading weights of pre-trained model...")
        # We load only one layer because the encoder and decoder are in just one layer
        # Load weights from full model
        load_unique_layer(video_generator, video_encoder, target_layer_name="encoder")
        load_unique_layer(video_generator, video_decoder, target_layer_name="decoder")
        print("Loading weights of pre-trained model: done!")

    elif model_name == "baseline":
        options = cfg_model.model_video
        # Common parameters
        options_generator = options.generator
        options_generator.momentum_bn = 0.9
        options_generator.input = options.input
        options_generator.batch_size = batch_size

        # Create the models
        original_video_encoder, video_decoder = create_generator(
            name=model_name, options=options_generator)
        # Mean and sigma from distribution
        z_mu, z_log_var = original_video_encoder.output
        video_code = reparameterization_trick(z_mu, z_log_var)

        video_encoder = tf.keras.Model(
            inputs=[original_video_encoder.input],
            outputs=[z_mu, z_log_var, video_code],
            name="encoder_wae"
        )
        video_encoder.summary()

        if not from_scratch:
            # Load weights
            # For some reason the encoder is being separated by different layers,
            # and the decoder is enclosed in layer "decoder_wae"
            load_weights_from(video_generator, video_encoder)

            pre_trained_video_decoder = tf.keras.models.load_model(
                f"{dir_experiment}/video_decoder")
            # pre_trained_video_decoder.summary()
            # load_unique_layer(pre_trained_video_decoder, video_decoder, target_layer_name="decoder_wae")
            load_weights_from(pre_trained_video_decoder, video_decoder)
            video_decoder.summary()

            # Sometimes the model loaded can have a fixed batch size, in this case we will create another model
            # and load weights
            if video_generator.input.shape[0] is not None:
                # Create another model and load weights
                # Create video generator model
                decoded_video, logits = video_decoder(video_code)

                new_video_generator = tf.keras.Model(
                    inputs=[video_encoder.input],
                    outputs=[z_mu, z_log_var, video_code, decoded_video, logits],
                    name="video_generator_new"
                )

                load_weights_from(video_generator, new_video_generator)
                new_video_generator.summary()
                video_generator = new_video_generator
        else:
            video_decoder.summary()

    if video_code_shape is None:
        # Ignoring batch size part
        video_code_shape = video_decoder.input_shape[1:]

    print("###########################################################")

    return video_encoder, video_decoder, video_code_shape, video_generator


def instantiate_video_model(cfg_model, batch_size: int, dir_experiment: str):
    tf.keras.backend.set_floatx('float32')
    print("###########################################################")
    print("Loading video model...")

    model_name = cfg_model.model_video.generator.get("name", "baseline")

    video_generator = tf.keras.models.load_model(f"{dir_experiment}/video_generator_model")
    video_generator.summary()

    video_decoder = None
    video_code_shape = None

    if model_name == "unet3d_ldm":
        print(f"Model: {model_name}")
        layer_dict = dict([(layer.name, layer) for layer in video_generator.layers])

        # Checks if video_decoder is there, probably not
        path_decoder = Path(f"{dir_experiment}/video_decoder")
        unet3d_ldm_model = None

        if not path_decoder.exists():
            print(f"Did not found decoder model at: {dir_experiment}/video_decoder")
            # There is no decoder, so we build one from the full video generator model
            # Get size of video embeddings
            video_code_shape = list(layer_dict["encoder"].output_shape[0][1:])
            video_code_dimension = np.prod(video_code_shape)
            tmp_video_code_shape = copy.deepcopy(video_code_shape)
            tmp_video_code_shape.insert(0, batch_size)

            # Build a video decoder only
            video_decoder, unet3d_ldm_model = create_video_decoder(
                cfg_model,
                video_code_dimension,
                tmp_video_code_shape,
                batch_size
            )

            # Load weights from full model
            pre_trained_dec_weights = layer_dict["decoder"].get_weights()
            print(f"Len pre-trained: {len(pre_trained_dec_weights)}")
            print(f"Shape current_weights: {pre_trained_dec_weights[0].shape}, "
                  f"values: {pre_trained_dec_weights[0][0, 0, 0, 1, 0:5]}")

            decoder_layer = None
            for i, layer in enumerate(video_decoder.layers):
                if "decoder" not in layer.name or "input" in layer.name:
                    continue
                decoder_layer = layer

            print(f"Layer name: {decoder_layer.name}, len: {len(decoder_layer.get_weights())}")
            # Before loading
            video_dec_weights = video_decoder.get_layer(decoder_layer.name).get_weights()
            print(
                f"Layer before: {video_dec_weights[0].shape}, values: {video_dec_weights[0][0, 0, 0, 1, 0:5]}")

            video_decoder.get_layer(decoder_layer.name).set_weights(pre_trained_dec_weights)

            # After loading
            video_dec_weights = video_decoder.get_layer(decoder_layer.name).get_weights()
            print(
                f"Layer after: {video_dec_weights[0].shape}, values: {video_dec_weights[0][0, 0, 0, 1, 0:5]}")

        # Loading pre_trained weights for a new video generator model with the correct batch size
        # This is done as the batch size of the trained model is kind of fixed.
        # Tried https://github.com/tensorflow/tensorflow/issues/59767#issuecomment-1849594440, but
        # there is an error on the reparametrization trick that stays with the previous batch size
        print("\nCreating a video generator model that will load the pre-trained one...")
        new_video_generator = create_video_ldm(cfg_model, unet3d_ldm_model, batch_size)

        # Load weights of pre-trained model
        for i, layer in enumerate(new_video_generator.layers):
            current_weights = layer.get_weights()
            if current_weights:
                name_filtered = re.sub(r'_\d+', '', layer.name)
                print(
                    f"Layer: {layer.name}, filtered: {name_filtered}, weights: {len(current_weights)}, type: {type(current_weights)}")

                pre_trained_weights = layer_dict[name_filtered].get_weights()
                print(f"Pre-trained: {len(pre_trained_weights)}")
                new_video_generator.get_layer(layer.name).set_weights(pre_trained_weights)
        print("Loaded weights for video generator from pre-trained model.")

        video_generator = new_video_generator
        video_generator.summary()

    if video_decoder is None:
        print("Video decoder was not found! Trying to load in default dir.")
        # Try to load the model directly
        video_decoder = tf.keras.models.load_model(f"{dir_experiment}/video_decoder")

    if video_code_shape is None:
        # Ignoring batch size part
        video_code_shape = video_decoder.input_shape[1:]

    print("###########################################################")

    return video_generator, video_decoder, video_code_shape
