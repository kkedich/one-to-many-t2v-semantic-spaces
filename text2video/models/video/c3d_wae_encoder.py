import tensorflow as tf

from text2video.models.operations import activation_function, initializer


def conv_block(input_tensor,
               name,
               output_filters,
               kernel_size,
               kernel_initializer,
               strides,
               act_function,
               lrelu_alpha=0.2,
               normalize_layer=False,
               momentum_bn=0.9,
               skip_connection=False,
               shortcut_features=None,
               using_strategy=False,
               norm_type="batchnorm"):
    conv_result = tf.keras.layers.Conv3D(output_filters,
                                         kernel_size,
                                         kernel_initializer=kernel_initializer,
                                         padding="SAME",
                                         strides=strides,
                                         name=f"{name}_conv")(input_tensor)

    if normalize_layer:
        if norm_type == "batchnorm":
            conv_result = tf.keras.layers.BatchNormalization(
                momentum=momentum_bn, name=f"{name}_bn", synchronized=using_strategy)(conv_result)
        elif norm_type == "group":
            conv_result = tf.keras.layers.GroupNormalization(groups=32, epsilon=1e-6)(conv_result)

    # Adding skip connection
    if skip_connection:
        print(
            f"Adding shortcut features <{shortcut_features.shape}> to <{conv_result.shape}>"
        )
        conv_result = tf.add(conv_result, shortcut_features)

    act_function = activation_function(act_function, lrelu_alpha)
    act_result = act_function(conv_result)

    print(f"Layer <{name}> with shape <{act_result.shape}>")

    return act_result


class Encoder:
    def __init__(self, video_shape, code_dimension, lrelu_alpha, momentum_bn,
                 act_function_name, kernel_init_name, stddev, using_strategy=False,
                 norm_type="batchnorm"):

        # [frames, width, height, channels]
        self.video_shape = video_shape
        self.code_dimension = code_dimension

        self.lrelu_alpha = lrelu_alpha
        self.momentum_bn = momentum_bn

        self.act_function_name = act_function_name
        self.kernel_init_name = kernel_init_name
        self.stddev = stddev

        self.norm_type = norm_type
        assert norm_type in ["group", "batchnorm"]

        # Strategy for multi-gpu
        self.using_strategy = using_strategy

    def build_model(self, batch_size=None):

        if batch_size is None:
            input_video = tf.keras.layers.Input(shape=self.video_shape,
                                                name="encoder_input_video")
        else:
            input_video = tf.keras.layers.Input(shape=self.video_shape,
                                                batch_size=batch_size,
                                                name="encoder_input_video")

        conv_keywords = {
            "momentum_bn": self.momentum_bn,
            "lrelu_alpha": self.lrelu_alpha,
            "act_function": self.act_function_name,
            "using_strategy": self.using_strategy,
            "norm_type": self.norm_type
        }

        conv1 = conv_block(input_video,
                           name="conv1",
                           output_filters=64,
                           kernel_size=[3, 3, 3],
                           strides=1,
                           kernel_initializer=initializer(
                               name=self.kernel_init_name, stddev=self.stddev),
                           **conv_keywords)

        conv2 = conv_block(conv1,
                           name="conv2",
                           output_filters=64,
                           kernel_size=[3, 3, 3],
                           strides=2,
                           normalize_layer=True,
                           kernel_initializer=initializer(
                               name=self.kernel_init_name, stddev=self.stddev),
                           **conv_keywords)

        conv3 = conv_block(conv2,
                           name="conv3",
                           output_filters=128,
                           kernel_size=[3, 3, 3],
                           strides=2,
                           normalize_layer=True,
                           kernel_initializer=initializer(
                               name=self.kernel_init_name, stddev=self.stddev),
                           **conv_keywords)

        conv2_shortcut = tf.keras.layers.Conv3D(filters=256,
                                                kernel_size=1,
                                                kernel_initializer=initializer(
                                                    name=self.kernel_init_name,
                                                    stddev=self.stddev),
                                                padding="valid",
                                                strides=[4, 4, 4],
                                                name="conv2_shortcut")(conv2)
        print(f"Shortcut shape: {conv2_shortcut.shape}")

        conv4 = conv_block(conv3,
                           name="conv4",
                           output_filters=256,
                           kernel_size=[3, 3, 3],
                           strides=2,
                           normalize_layer=True,
                           skip_connection=True,
                           shortcut_features=conv2_shortcut,
                           kernel_initializer=initializer(
                               name=self.kernel_init_name, stddev=self.stddev),
                           **conv_keywords)

        conv5 = conv_block(conv4,
                           name="conv5",
                           output_filters=256,
                           kernel_size=[3, 3, 3],
                           strides=2,
                           normalize_layer=True,
                           kernel_initializer=initializer(
                               name=self.kernel_init_name, stddev=self.stddev),
                           **conv_keywords)

        conv4_shortcut = tf.keras.layers.Conv3D(filters=256,
                                                kernel_size=1,
                                                kernel_initializer=initializer(
                                                    name=self.kernel_init_name,
                                                    stddev=self.stddev),
                                                padding="valid",
                                                strides=[4, 4, 4],
                                                name="conv4_shortcut")(conv4)
        print(f"Shortcut shape: {conv4_shortcut.shape}")

        conv6 = conv_block(conv5,
                           name="conv6",
                           output_filters=256,
                           kernel_size=[3, 3, 3],
                           strides=2,
                           normalize_layer=True,
                           skip_connection=True,
                           shortcut_features=conv4_shortcut,
                           kernel_initializer=initializer(
                               name=self.kernel_init_name, stddev=self.stddev),
                           **conv_keywords)

        shape_last_conv_block = tf.shape(conv6)
        reshaped = tf.transpose(a=conv6, perm=[0, 1, 4, 2, 3])
        # Reshape output to fit dense layer input
        output_dimension = tf.math.reduce_prod(tf.shape(reshaped)[1:])
        reshaped = tf.reshape(reshaped, [-1, output_dimension])

        fc_z_mu_output = tf.keras.layers.Dense(self.code_dimension,
                                               kernel_initializer=initializer(
                                                   name=self.kernel_init_name,
                                                   stddev=self.stddev),
                                               name="fc_z_mu_output")
        fc_z_log_var_output = tf.keras.layers.Dense(
            self.code_dimension,
            kernel_initializer=initializer(name=self.kernel_init_name,
                                           stddev=self.stddev),
            name="fc_z_log_var_output")

        z_mu = fc_z_mu_output(reshaped)
        z_log_var = fc_z_log_var_output(reshaped)

        model = tf.keras.Model(inputs=input_video,
                               outputs=[z_mu, z_log_var],
                               name="encoder_wae")

        return model, shape_last_conv_block
