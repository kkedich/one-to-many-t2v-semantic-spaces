import tensorflow as tf


def activation_function(name, lrelu_alpha=0.2):
    """
    Creates an activation function operation
    Args:
        name: (str) name of the activation function to be used
        lrelu_alpha: (float) value for the alpha parameter of the LeakyRelu
         function

    Returns:
        - (tf.keras.layers) keras activation function

    """
    act_function_op = None
    if name == "relu":
        act_function_op = tf.keras.layers.ReLU()
    elif name == "leaky_relu":
        act_function_op = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)
    elif name == "elu":
        act_function_op = tf.keras.layers.ELU()
    elif name == "swish":  # or silu
        act_function_op = tf.nn.swish
    else:
        raise ValueError(f"<{name}> is an invalid activation function. "
                         f"Valid types: relu, leaky_relu, elu.")

    return act_function_op


def initializer(name, stddev=0.009, start_value=0.0, min_val=0, max_val=1):
    """
    Creates an initializer operation, e.g., for the model's weights and biases
    Args:
        name: (str) type of initializer
        stddev: (float) standard deviation value to be considered on the
         initializers: 'normal' and 'truncated_normal'
        start_value: (float) start value for the constant initializer
        min_val: (int) minimum value for the uniform initializer
        max_val: (int) maximum value for the uniform initializer

    Returns:
        - a Tensorflow or keras initializer

    """

    initializer_op = None
    if name == "xavier":
        initializer_op = tf.keras.initializers.GlorotNormal()
    elif name == "normal":
        initializer_op = tf.random_normal_initializer(stddev=stddev)
    elif name == "constant":
        initializer_op = tf.constant_initializer(start_value)
    elif name == "truncated_normal":
        initializer_op = tf.keras.initializers.TruncatedNormal(stddev=stddev)
    elif name == "he_normal":
        initializer_op = tf.keras.initializers.he_normal()
    elif name == "uniform":
        initializer_op = tf.random_uniform_initializer(min_val, max_val)
    else:
        raise ValueError(f"<{name}> is an invalid initializer.")

    return initializer_op
