import tensorflow as tf

from text2video.trainer.forward_video import (forward_decoder,
                                              forward_enc_dec_model)
from text2video.utils.tensor_utils import check_nan


def forward_text2shared(text2shared_model, text_embeddings, text_embeddings_with_noise, training: bool, options):
    """
    Forward text to shared model
    """
    if text_embeddings_with_noise is not None:
        text2shared_code_outputs = text2shared_model(
            text_embeddings_with_noise, training=training)
    else:
        text2shared_code_outputs = text2shared_model(text_embeddings, training=training)

    if isinstance(text2shared_code_outputs, tuple) and options.text_noise_dropout_embeddings > 0.0:
        # The second output are the text embeddings with the dropout part
        text2shared_code, text_embeddings_with_noise = text2shared_code_outputs
    else:
        text2shared_code = text2shared_code_outputs

    return text2shared_code, text_embeddings_with_noise


def forward_pass_stacked_wae_progressive(
        text_embeddings,
        text_embeddings_with_noise,
        batch_videos,
        mf_text2shared_model,
        mf_video2shared_model,
        mf_shared2video_model,
        video_encoder,
        video_decoder,
        options,
        training):

    # Text2Shared forward
    text2shared_code, text_embeddings_with_noise = forward_text2shared(
        mf_text2shared_model, text_embeddings, text_embeddings_with_noise, training, options)

    text_shared2video_code = mf_shared2video_model(text2shared_code, training=training)
    text2shared2v_decoded_video, _, _ = forward_decoder(
        text_shared2video_code, video_encoder, video_decoder, training, options)
    check_nan(text2shared2v_decoded_video, tensor_name="text2shared2v_decoded_video")

    # Forward video part
    z_mu_true, z_log_var_true, video_code_true, decoded_video_true = forward_enc_dec_model(
        batch_videos, video_encoder, video_decoder, training, options)

    # Reshaping video code depending on the model to calculate the losses and make the forward pass
    # for other models
    if options.video_model_name == "unet3d_ldm":
        video_code_true_flattened = tf.reshape(video_code_true, shape=[batch_videos.shape[0], -1])
    else:
        video_code_true_flattened = video_code_true

    # Video-to-shared and shared-to-video forward
    video2shared_code = mf_video2shared_model(video_code_true_flattened, training=training)
    shared2video_code = mf_shared2video_model(video2shared_code, training=training)

    check_nan(text2shared_code, tensor_name="text2shared_code")
    check_nan(video2shared_code, tensor_name="video2shared_code")
    check_nan(shared2video_code, tensor_name="shared2video_code")

    # Forward of video part with shared codes code too
    shared2v_decoded_video, _, _ = forward_decoder(
        shared2video_code, video_encoder, video_decoder, training, options)
    check_nan(shared2v_decoded_video, tensor_name="shared2v_decoded_video")

    outputs = {
        "text2shared_code": text2shared_code,
        "text_shared2video_code": text_shared2video_code,
        "text2shared2v_decoded_video": text2shared2v_decoded_video,
        # Pre-trained video autoencoder
        "video_code_true": video_code_true_flattened,
        "decoded_video_true": decoded_video_true,
        # Video shared AE
        "video2shared_code": video2shared_code,
        "video_shared2video_code": shared2video_code,
        "shared2v_decoded_video": shared2v_decoded_video,
    }

    return outputs
