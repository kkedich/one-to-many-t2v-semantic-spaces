"""
From
https://stackoverflow.com/questions/65484420/define-custom-loss-perceptual-loss-in-cnn-autoencoder-with-pre-train-vgg19-ten

"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import vgg19


def load_vgg19_model(selected_layers, selected_layer_weights, input_shape=(64, 64, 3)):
    """The weights are saved into .keras/models/vgg19_weights_tf_dim_ordering_tf_kernels.h5
       If you will gonna use with slurm, be sure to have this model pre-loaded there.
    """

    if len(selected_layers) != len(selected_layer_weights):
        raise ValueError("Number of selected layers is different from the number of weights to be used."
                         f"Selected layers: {selected_layers}, selected weights: {selected_layer_weights}")

    print(f"Loading VGG19 model with layers: {selected_layers}")
    vgg = vgg19.VGG19(weights="imagenet", include_top=False, input_shape=input_shape)
    vgg.trainable = False
    outputs = [vgg.get_layer(current_layer).output for current_layer in selected_layers]
    print(f"Loaded output format: {outputs}")

    vgg_model = tf.keras.Model(vgg.input, outputs)
    vgg_model.summary()

    return vgg_model


@tf.function
def perceptual_loss(input_image, reconstruct_image, vgg_model, selected_layer_weights):
    # Check if image is in grayscale and converts to 3 channels
    input_image_rgb = input_image
    reconstruct_image_rgb = reconstruct_image
    if input_image.shape[-1] == 1:
        input_image_rgb = tf.repeat(input_image, repeats=[3], axis=-1)
        reconstruct_image_rgb = tf.repeat(reconstruct_image, repeats=[3], axis=-1)

    # Reshape (batch_size, frames, width, height, channels) to (b*frames, w, h, c)
    vid_shape = tf.shape(input_image_rgb)
    new_shape = [vid_shape[0] * vid_shape[1], vid_shape[-3], vid_shape[-2], vid_shape[-1]]
    input_image_rgb = tf.reshape(input_image_rgb, new_shape)
    reconstruct_image_rgb = tf.reshape(reconstruct_image_rgb, new_shape)

    h1_list = vgg_model(input_image_rgb)
    h2_list = vgg_model(reconstruct_image_rgb)

    rc_loss = 0.0
    if len(selected_layer_weights) == 1:
        # It is just one conv layer, so we need to change the type of the vgg outputs to a list
        h1_list = [h1_list]
        h2_list = [h2_list]

    for h1, h2, weight in zip(h1_list, h2_list, selected_layer_weights):
        h1 = tf.keras.backend.batch_flatten(h1)
        h2 = tf.keras.backend.batch_flatten(h2)
        rc_loss = rc_loss + weight * tf.keras.backend.sum(tf.keras.backend.square(h1 - h2), axis=-1)

    # Reshape loss to (batch_size, frames)
    rc_loss = tf.reshape(rc_loss, [vid_shape[0], vid_shape[1]])

    return rc_loss


def calculate_perceptual_loss(true_videos, pred_videos, options, reduce=True):

    percep_loss = perceptual_loss(true_videos, pred_videos,
                                  vgg_model=options.vgg_model,
                                  selected_layer_weights=options.selected_layer_weights)
    if reduce:
        percep_loss = tf.reduce_mean(percep_loss)
    else:
        percep_loss = tf.reduce_mean(percep_loss,
                                     np.arange(1, percep_loss.shape.rank, dtype=np.int32))
    # print(f"Perceptual loss: {percep_loss.shape}, perceptual loss={percep_loss}")
    percep_loss = options.perceptual_delta * percep_loss

    return percep_loss
