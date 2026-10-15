import tensorflow as tf

from text2video.trainer.forward_video import (forward_decoder,
                                              forward_enc_dec_model)
from text2video.utils.tensor_utils import check_nan


def forward_pass_non_progressive(
        text_embeddings,
        batch_videos,
        mf_model,
        video_encoder,
        video_decoder,
        options,
        training):
    # Forward part without losses calculation
    # MF forward
    mf_outputs = mf_model(text_embeddings, training=training)

    if isinstance(mf_outputs, tuple) and options.text_noise_dropout_embeddings > 0.0:
        # The second output are the text embeddings with the dropout part
        mf_code, text_embeddings_with_noise = mf_outputs
    else:
        mf_code = mf_outputs
    check_nan(mf_code, tensor_name="mf_code")

    mf_code2decoded_video, _, _ = forward_decoder(
        mf_code, video_encoder, video_decoder, training, options)
    check_nan(mf_code2decoded_video, tensor_name="mf_code")

    # Forward video part
    z_mu_true, z_log_var_true, video_code_true, decoded_video_true = forward_enc_dec_model(
        batch_videos, video_encoder, video_decoder, training, options)

    # Reshaping video code depending on the model to calculate the losses and make the forward pass
    # for other models
    if options.video_model_name == "unet3d_ldm":
        video_code_true_flattened = tf.reshape(video_code_true, shape=[batch_videos.shape[0], -1])
    else:
        video_code_true_flattened = video_code_true

    outputs = {
        "mf_code": mf_code,
        "mf_decoded_video": mf_code2decoded_video,
        # Pre-trained video autoencoder
        "video_code_true": video_code_true_flattened,
        "decoded_video_true": decoded_video_true,
    }

    return outputs
