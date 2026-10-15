import logging
import numpy as np
import tensorflow as tf


def scale(image, feature_range=(-1, 1), undo=False, tensor=True):
    new_image = None
    if undo:
        # We do not use feature_range to undo, we just get the min
        # and max value to infer how to undo scaling
        if not tensor:
            new_image = ((image - image.min()) * 255 / (image.max() - image.min())).astype(np.uint8)
        else:
            current_min = tf.reduce_min(input_tensor=image)
            current_max = tf.reduce_max(input_tensor=image)
            new_image = (image - current_min) * 255 / (current_max - current_min)
            new_image = tf.cast(new_image, tf.int32)
    else:
        if tensor:
            current_min = tf.reduce_min(input_tensor=image)
        else:
            current_min = image.min()
            logging.debug("[Scale] min={}, max={}".format(image.min(), image.max()))

        # Scale to (0, 1)
        # print("before scale -> min()={}, max()={}".format(current_min, current_max))
        new_image = ((image - current_min) / (255 - current_min))
        if not tensor:
            logging.debug(
                "[Scale] after scale -> min()={}, max()={}".format(new_image.min(), new_image.max()))

        # Scale to feature_range
        new_image = new_image * (feature_range[1] - feature_range[0]) + feature_range[0]

        if not tensor:
            logging.debug(
                "[Scale] after scale -> min()={}, max()={}".format(new_image.min(), new_image.max()))

    return new_image


def scale_batch(images, feature_range=(-1, 1), undo=False):
    """Scale batch of images"""
    new_images = None
    current_min = tf.reduce_min(input_tensor=images, axis=[1, 2, 3])  # 0 is the batch

    shape_imgs = tf.shape(input=images)
    imgs_reshaped = tf.reshape(images, shape=[shape_imgs[0], -1])
    sub = imgs_reshaped - tf.reshape(current_min, shape=[-1, 1])

    if undo:
        # we do not use feature_range to undo, we just get the min and max
        # value to infer how to undo scaling
        current_max = tf.reduce_max(input_tensor=images, axis=[1, 2, 3])

        new_images = tf.divide(sub * 255, tf.reshape(current_max - current_min, shape=[-1, 1]))
        # one image -> (image - current_min) * 255 / (current_max - current_min)

        new_images = tf.reshape(new_images, shape=shape_imgs)
        new_images = tf.cast(new_images, tf.int32)
    else:
        # scale to (0, 1)
        new_images = tf.divide(sub, tf.reshape(255 - current_min, shape=[-1, 1]))
        # one image -> new_image = ((image - current_min) / (255 - current_min))

        # scale to feature_range
        _min, _max = feature_range
        new_images = new_images * (_max - _min) + _min

        new_images = tf.reshape(new_images, shape=shape_imgs)

    return new_images
