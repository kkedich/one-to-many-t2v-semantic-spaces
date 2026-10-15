import tensorflow as tf


def tf_resize_videos(videos_input, max_batch_size, target_size=(224, 224)):
    video_elements = get_unpacked_tensors(videos_input, num_partitions=max_batch_size)
    processed = []
    for vid in video_elements:
        vid_resized = tf.image.resize_bilinear(vid, target_size)
        processed.append(vid_resized)

    output = tf.stack(processed, 0)
    return output


def get_unpacked_tensors(x: tf.Tensor, num_partitions: int):
    """
    Get a list of tensors of a tensor of shape [None, ...] or [batch_size, ...]
    Each element of the batch is returned as a tensor in a list
    Args:
        x: input tensor
        num_partitions: maximum batch size that will be used as a number of partitions

    Returns:
        -
    """
    if x.get_shape().as_list()[0] is None:
        partitions = tf.range(num_partitions)
        partitioned = tf.dynamic_partition(x, partitions, num_partitions)

        x_unpacked = []
        element_shape = x.get_shape().as_list()[1:]  # ignores batch dimension
        for element in partitioned:
            element = tf.reshape(element, element_shape)
            element.set_shape(element_shape)
            x_unpacked.append(element)
    else:
        x_unpacked = tf.unstack(x, axis=0)

    return x_unpacked
