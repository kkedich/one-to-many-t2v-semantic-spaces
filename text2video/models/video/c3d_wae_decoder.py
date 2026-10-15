import tensorflow as tf

from text2video.models.operations import activation_function, initializer


def conv_transpose_block(input_tensor,
                         name,
                         output_filters,
                         kernel_size,
                         kernel_initializer,
                         strides,
                         act_function,
                         lrelu_alpha,
                         momentum_bn,
                         skip_connection=False,
                         shortcut_features=None,
                         using_strategy=False,
                         norm_type="batchnorm"):
    conv_result = tf.keras.layers.Conv3DTranspose(
        output_filters,
        kernel_size,
        kernel_initializer=kernel_initializer,
        padding="same",
        strides=strides,
        name=f"{name}_conv_transpose")(input_tensor)

    if norm_type == "batchnorm":
        conv_result = tf.keras.layers.BatchNormalization(
            momentum=momentum_bn, name=f"{name}_bn_transpose", synchronized=using_strategy)(conv_result)
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


class Decoder:
    def __init__(self, output_channels, code_dimension, lrelu_alpha,
                 momentum_bn, act_function_name, kernel_init_name, stddev, dropout_rate=0.5,
                 using_strategy=False, norm_type="batchnorm"):

        # [frames, width, height, channels]
        self.output_channels = output_channels
        self.code_dimension = code_dimension

        self.lrelu_alpha = lrelu_alpha
        self.momentum_bn = momentum_bn

        self.act_function_name = act_function_name
        self.kernel_init_name = kernel_init_name
        self.stddev = stddev
        self.dropout_rate = dropout_rate

        self.norm_type = norm_type
        assert norm_type in ["group", "batchnorm"]

        # Strategy for multi-gpu
        self.using_strategy = using_strategy

    def build_model(self, batch_size=None):
        if batch_size is None:
            input_video_code = tf.keras.layers.Input(shape=self.code_dimension,
                                                     name="decoder_input_code")
        else:
            input_video_code = tf.keras.layers.Input(shape=self.code_dimension,
                                                     batch_size=batch_size,
                                                     name="decoder_input_code")

        conv_keywords = {
            "momentum_bn": self.momentum_bn,
            "lrelu_alpha": self.lrelu_alpha,
            "act_function": self.act_function_name,
            "using_strategy": self.using_strategy,
            "norm_type": self.norm_type
        }

        layer1_fc_output = tf.keras.layers.Dense(
            1024,
            kernel_initializer=initializer(name=self.kernel_init_name,
                                           stddev=self.stddev),
            name="layer1_fc_output")(input_video_code)
        act_function_layer = activation_function(self.act_function_name,
                                                 self.lrelu_alpha)
        act_result = act_function_layer(layer1_fc_output)
        # print(act_result.shape)
        layer1_fc_output = tf.keras.layers.Dropout(rate=self.dropout_rate)(act_result)

        # Reshaping for the c3d stack
        layer1_reshaped = tf.reshape(layer1_fc_output, [-1, 1, 2, 2, 256])
        # print(f"layer1_reshaped: <{layer1_reshaped.shape}>")

        if self.norm_type == "batchnorm":
            layer1_reshaped = tf.keras.layers.BatchNormalization(
                momentum=self.momentum_bn,
                name="layer1_reshaped_bn",
                synchronized=self.using_strategy
            )(layer1_reshaped)
        elif self.norm_type == "group":
            layer1_reshaped = tf.keras.layers.GroupNormalization(
                groups=32, epsilon=1e-6)(layer1_reshaped)

        layer1_reshaped = activation_function(
            self.act_function_name, self.lrelu_alpha)(layer1_reshaped)
        # print(f"layer1_reshaped: <{layer1_reshaped.shape}>")

        input0_shortcut = tf.keras.layers.Conv3D(
            filters=256,
            kernel_size=1,
            kernel_initializer=initializer(name=self.kernel_init_name,
                                           stddev=self.stddev),
            padding="valid",
            strides=1,
            name="input0_shortcut")(layer1_reshaped)
        print(f"input0_shortcut: <{input0_shortcut.shape}>")
        input0_shortcut_padded = tf.pad(tensor=input0_shortcut,
                                        paddings=[[0, 0], [0, 1], [3, 3],
                                                  [3, 3], [0, 0]])
        # print(f"input0_shortcut_padded: <{input0_shortcut_padded.shape}>")

        conv1_transpose = conv_transpose_block(layer1_reshaped,
                                               name="layer2",
                                               output_filters=256,
                                               kernel_size=[3, 3, 3],
                                               strides=[1, 2, 2],
                                               kernel_initializer=initializer(
                                                   name=self.kernel_init_name,
                                                   stddev=self.stddev),
                                               **conv_keywords)

        conv2_transpose = conv_transpose_block(
            conv1_transpose,
            name="layer3",
            output_filters=256,
            kernel_size=[3, 3, 3],
            strides=[2, 2, 2],
            skip_connection=True,
            shortcut_features=input0_shortcut_padded,
            kernel_initializer=initializer(name=self.kernel_init_name,
                                           stddev=self.stddev),
            **conv_keywords)

        conv3_transpose = conv_transpose_block(conv2_transpose,
                                               name="layer4",
                                               output_filters=128,
                                               kernel_size=[3, 3, 3],
                                               strides=[2, 2, 2],
                                               kernel_initializer=initializer(
                                                   name=self.kernel_init_name,
                                                   stddev=self.stddev),
                                               **conv_keywords)

        conv4_transpose = conv_transpose_block(conv3_transpose,
                                               name="layer5",
                                               output_filters=64,
                                               kernel_size=[3, 3, 3],
                                               strides=[2, 2, 2],
                                               kernel_initializer=initializer(
                                                   name=self.kernel_init_name,
                                                   stddev=self.stddev),
                                               **conv_keywords)

        conv4_shortcut = tf.keras.layers.Conv3D(
            filters=64,
            kernel_size=1,
            kernel_initializer=initializer(name=self.kernel_init_name,
                                           stddev=self.stddev),
            padding="valid",
            strides=1,
            name="conv4_shortcut")(conv2_transpose)
        # print(f"conv4_shortcut: <{conv4_shortcut.shape}>")
        conv4_shortcut_padded = tf.pad(tensor=conv4_shortcut,
                                       paddings=[[0, 0], [7, 7], [28, 28],
                                                 [28, 28], [0, 0]])
        # print(f"conv4_shortcut_padded: <{conv4_shortcut_padded.shape}>")

        conv5_transpose = conv_transpose_block(
            conv4_transpose,
            name="layer6",
            output_filters=64,
            kernel_size=[3, 3, 3],
            strides=[2, 2, 2],
            skip_connection=True,
            shortcut_features=conv4_shortcut_padded,
            kernel_initializer=initializer(name=self.kernel_init_name,
                                           stddev=self.stddev),
            **conv_keywords)

        logits = tf.keras.layers.Conv3DTranspose(
            self.output_channels,
            kernel_size=3,
            kernel_initializer=initializer(name=self.kernel_init_name,
                                           stddev=self.stddev),
            padding="same",
            strides=1,
            name="layer7")(conv5_transpose)

        out = tf.tanh(logits)
        print(f"out: <{out.shape}>")

        model = tf.keras.Model(inputs=input_video_code,
                               outputs=[out, logits],
                               name="decoder_wae")

        return model
