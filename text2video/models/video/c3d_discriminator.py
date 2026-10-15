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
               batch_norm=False,
               momentum_bn=0.9,
               dropout_rate=None):
    conv_result = tf.keras.layers.Conv3D(output_filters,
                                         kernel_size,
                                         kernel_initializer=kernel_initializer,
                                         padding="SAME",
                                         strides=strides,
                                         name=f"{name}_conv")(input_tensor)

    if batch_norm:
        conv_result = tf.keras.layers.BatchNormalization(
            momentum=momentum_bn, name=f"{name}_bn")(conv_result)

    act_function = activation_function(act_function, lrelu_alpha)
    act_result = act_function(conv_result)

    if dropout_rate:
        act_result = tf.keras.layers.Dropout(rate=dropout_rate)(act_result)

    print(f"Layer <{name}> with shape <{act_result.shape}>")

    return act_result


class Discriminator():
    def __init__(self, video_shape, features_dimension, lrelu_alpha, momentum_bn,
                 act_function_name, kernel_init_name, stddev, dropout_rate):

        # [frames, width, height, channels]
        self.video_shape = video_shape
        self.features_dimension = features_dimension

        self.lrelu_alpha = lrelu_alpha
        self.momentum_bn = momentum_bn
        self.dropout_rate = dropout_rate

        self.act_function_name = act_function_name
        self.kernel_init_name = kernel_init_name
        self.stddev = stddev

    def build_model(self):

        input_video = tf.keras.layers.Input(shape=self.video_shape,
                                            name="disc_input_video")

        conv_keywords = {
            "momentum_bn": self.momentum_bn,
            "lrelu_alpha": self.lrelu_alpha,
            "act_function": self.act_function_name,
            "dropout_rate": self.dropout_rate
        }

        conv1 = conv_block(input_video,
                           name="disc1",
                           output_filters=64,
                           kernel_size=[3, 3, 3],
                           strides=2,
                           kernel_initializer=initializer(
                               name=self.kernel_init_name, stddev=self.stddev),
                           **conv_keywords)

        conv2 = conv_block(conv1,
                           name="disc2",
                           output_filters=128,
                           kernel_size=[3, 3, 3],
                           strides=2,
                           kernel_initializer=initializer(
                               name=self.kernel_init_name, stddev=self.stddev),
                           batch_norm=True,
                           **conv_keywords)

        conv3 = conv_block(conv2,
                           name="disc3",
                           output_filters=256,
                           kernel_size=[3, 3, 3],
                           strides=2,
                           kernel_initializer=initializer(
                               name=self.kernel_init_name, stddev=self.stddev),
                           batch_norm=True,
                           **conv_keywords)

        conv4 = conv_block(conv3,
                           name="disc4",
                           output_filters=256,
                           kernel_size=[3, 3, 3],
                           strides=2,
                           kernel_initializer=initializer(
                               name=self.kernel_init_name, stddev=self.stddev),
                           batch_norm=True,
                           **conv_keywords)

        conv5 = conv_block(conv4,
                           name="disc5",
                           output_filters=256,
                           kernel_size=[3, 3, 3],
                           strides=2,
                           kernel_initializer=initializer(
                               name=self.kernel_init_name, stddev=self.stddev),
                           batch_norm=True,
                           **conv_keywords)

        reshaped = tf.transpose(a=conv5, perm=[0, 1, 4, 2, 3])
        print(f"transposed: <{reshaped.shape}>")
        # Reshape output to fit dense layer input
        output_dimension = tf.math.reduce_prod(tf.shape(reshaped)[1:])
        intermediate_features = tf.reshape(reshaped, [-1, output_dimension])
        print(f"intermediate_features: <{intermediate_features.shape}>")

        fc1_layer_output = tf.keras.layers.Dense(self.features_dimension,
                                                 kernel_initializer=initializer(
                                                     name=self.kernel_init_name,
                                                     stddev=self.stddev),
                                                 name="fc1_layer_output")
        fc1_output = fc1_layer_output(intermediate_features)
        fc1_output = activation_function(self.act_function_name, self.lrelu_alpha)(fc1_output)
        fc1_output = tf.keras.layers.Dropout(rate=self.dropout_rate)(fc1_output)
        print(fc1_output.shape)

        fc2_layer_output = tf.keras.layers.Dense(1,
                                                 kernel_initializer=initializer(
                                                     name=self.kernel_init_name,
                                                     stddev=self.stddev),
                                                 name="fc2_layer_output")
        fc2_output = fc2_layer_output(fc1_output)
        print(fc2_output.shape)

        model = tf.keras.Model(inputs=input_video,
                               outputs=[intermediate_features, fc2_output],
                               name="c3d_discriminator")

        return model
