import tensorflow as tf


def videos2images(batch_videos):
    """ Converts tensor of batch of videos [b, f, w, h, c] to batch of images [b*f, w, h, c]"""

    # Check if image is in grayscale and converts to 3 channels
    batch_images = batch_videos
    if batch_images.shape[-1] == 1:
        batch_images = tf.repeat(batch_images, repeats=[3], axis=-1)

    # Reshape (batch_size, frames, width, height, channels) to (b*frames, w, h, c)
    vid_shape = tf.shape(batch_images)
    new_shape = [vid_shape[0] * vid_shape[1], vid_shape[-3], vid_shape[-2], vid_shape[-1]]
    batch_images = tf.reshape(batch_images, new_shape)

    return batch_images
