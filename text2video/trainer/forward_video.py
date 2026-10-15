import tensorflow as tf

from text2video.utils.tensor_utils import check_nan


def forward_pass_model(input_videos, generator, training, options):
    z_mu, z_log_var, video_code = None, None, None
    gen_output = generator(input_videos, training=training)

    if options.video_model_name == "baseline":
        z_mu, z_log_var, video_code, decoded_video, _ = gen_output

    elif options.video_model_name == "unet3d_ldm":
        z_mu, z_log_var, video_code, decoded_video = gen_output
    else:
        raise ValueError("Invalid generator model name.")

    if video_code is not None:
        check_nan(video_code, tensor_name="video_code")
    if z_mu is not None:
        check_nan(z_mu, tensor_name="z_mu")
        check_nan(z_log_var, tensor_name="z_log_var")

    tf.print("decoded_video", decoded_video)
    check_nan(decoded_video, tensor_name="decoded_video")

    return z_mu, z_log_var, video_code, decoded_video


def forward_enc_dec_model(input_videos, video_encoder, video_decoder,
                          training, options, use_check_nan=True):

    if options.video_model_name == "baseline":
        # True video code considering the training of the video
        z_mu, z_log_var, video_code = video_encoder(input_videos, training=training)
        decoded_video, _ = video_decoder(video_code, training=training)

    elif options.video_model_name == "unet3d_ldm":
        # True video code considering the training of the video
        z_mu, z_log_var, video_code = video_encoder(input_videos, training=training)
        decoded_video = video_decoder(video_code, training=training)
    else:
        raise ValueError(f"Option is not available: '{options.video_model_name}'")

    if use_check_nan:
        check_nan(decoded_video, tensor_name="decoded_video")
        check_nan(z_mu, tensor_name="z_mu")
        check_nan(z_log_var, tensor_name="z_log_var")
        check_nan(video_code, tensor_name="video_code")

    return z_mu, z_log_var, video_code, decoded_video


def forward_decoder(input_codes, video_encoder, video_decoder, training, options):
    """Forward part only for the Decoder model"""
    decoded_video, video_code_shape, video_code_pred = None, None, None

    if options.video_model_name == "baseline":
        video_code_pred = input_codes
        decoded_video, _ = video_decoder(video_code_pred, training=training)

    elif options.video_model_name == "unet3d_ldm":
        video_code_shape = list(video_encoder.output_shape[0][1:])
        video_code_shape.insert(0, input_codes.shape[0])
        video_code_pred = tf.reshape(input_codes, shape=video_code_shape)

        decoded_video = video_decoder(video_code_pred, training=training)

    return decoded_video, video_code_shape, video_code_pred
