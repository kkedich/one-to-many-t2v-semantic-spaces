import tensorflow as tf


def transpose_batch_time(x):
    """
    Code from
    https://github.com/tensorflow/tensorflow/blob/a6d8ffae097d0132989ae4688d224121ec6d8f35/tensorflow/python/ops/rnn.py#L44

    Transposes the batch and time dimensions of a Tensor.
    If the input tensor has rank < 2 it returns the original tensor. Retains as
    much of the static shape information as possible.
    Args:
      x: A Tensor.
    Returns:
      x transposed along the first two dimensions.
    """
    x_static_shape = x.get_shape()
    if x_static_shape.ndims is not None and x_static_shape.ndims < 2:
        return x

    batch_dim = x_static_shape[0] if x_static_shape[0] is None else x_static_shape[0].value
    time_dim = x_static_shape[1] if x_static_shape[1] is None else x_static_shape[1].value

    x_rank = tf.rank(x)
    x_t = tf.transpose(a=x, perm=tf.concat(([1, 0], tf.range(2, x_rank)), axis=0))
    x_t.set_shape(tf.TensorShape(
        [time_dim, batch_dim]).concatenate(x_static_shape[2:]))

    return x_t


def pad_sequence(input_tensor, desired_size, constant_value=0, dtype=tf.int32):
    """
    Pad second dimension of a input sequence tensor with either a constant value,
    or a constant vector when the rank's input is 3.
    Example: [-1,1,2,3] with desired_size = 6 results in [-1,1,2,3,0,0]

    Reference: https://stackoverflow.com/a/45721424
    :param input_tensor: input tensor with rank 2
    :param desired_size: desired size for second dimension
    :return: a padded tensor
    """
    input_rank = tf.rank(input_tensor)
    multiples = tf.cond(pred=tf.equal(input_rank, tf.constant(3, dtype=tf.int32)),
                        true_fn=lambda: tf.stack([tf.shape(input=input_tensor)[0],
                                                  desired_size - tf.shape(input=input_tensor)[1], 1], 0),
                        false_fn=lambda: tf.stack([tf.shape(input=input_tensor)[0],
                                                   desired_size - tf.shape(input=input_tensor)[1]], 0),
                        name="multiples")

    paddings = tf.tile([[constant_value]], multiples, name="paddings")
    paddings = tf.cast(paddings, dtype=dtype)

    result = tf.concat([input_tensor, paddings], 1)

    return result


def check_nan(input_tensor, tensor_name):
    try:
        if input_tensor is not None:
            tf.debugging.check_numerics(input_tensor, message=f"Checking {tensor_name}")
    except Exception as e:
        assert f"Checking {tensor_name}: Tensor had NaN values" in e.message  # pylint: disable=E1101
