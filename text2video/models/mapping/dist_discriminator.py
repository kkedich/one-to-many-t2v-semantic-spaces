import tensorflow as tf

from text2video.models.operations import activation_function, initializer


class DistDiscriminator:
    """
    Implements a vanilla mapping function between text representations
    to the video representations.
    """

    def __init__(self,
                 input_dimension,
                 dropout_rate_embeddings=0.5,
                 activation_fn_name="leaky_relu",
                 lrelu_alpha=0.2):
        self.input_dimension = input_dimension
        self.dropout_rate_embeddings = dropout_rate_embeddings

        self.activation_fn_name = activation_fn_name
        self.lrelu_alpha = lrelu_alpha

    def build_model(self):
        # New code for the mapping function model
        dist_placeholder = tf.keras.layers.Input(
            shape=[self.input_dimension], name="input_dist_disc_placeholder")

        # Initializer variables
        kernel_init_name = "normal"
        stddev = 0.009

        # Layer 1
        layer1_fc_output = tf.keras.layers.Dense(
            256,
            kernel_initializer=initializer(name=kernel_init_name, stddev=stddev)
        )(dist_placeholder)
        layer1_norm_output = tf.keras.layers.LayerNormalization(axis=1)(layer1_fc_output)
        layer1_act_function_layer = activation_function(self.activation_fn_name, self.lrelu_alpha)
        layer1_act_result = layer1_act_function_layer(layer1_norm_output)
        print(f"layer1_act_result: {layer1_act_result.shape}")

        # Layer 2
        layer2_fc_output = tf.keras.layers.Dense(
            256,
            kernel_initializer=initializer(name=kernel_init_name, stddev=stddev)
        )(layer1_act_result)
        layer2_norm_output = tf.keras.layers.LayerNormalization(axis=1)(layer2_fc_output)
        layer2_act_function_layer = activation_function(self.activation_fn_name, self.lrelu_alpha)
        layer2_act_result_features = layer2_act_function_layer(layer2_norm_output)
        print(f"layer2_act_result: {layer2_act_result_features.shape}")
        layer2_act_result = tf.keras.layers.Dropout(
            rate=self.dropout_rate_embeddings)(layer2_act_result_features)

        # Layer 3
        layer3_fc_output = tf.keras.layers.Dense(
            128,
            kernel_initializer=initializer(name=kernel_init_name, stddev=stddev)
        )(layer2_act_result)
        layer3_norm_output = tf.keras.layers.LayerNormalization(axis=1)(layer3_fc_output)
        layer3_act_function_layer = activation_function(self.activation_fn_name, self.lrelu_alpha)
        layer3_act_result_features = layer3_act_function_layer(layer3_norm_output)
        print(f"layer3_act_result: {layer3_act_result_features.shape}")
        layer3_act_result = tf.keras.layers.Dropout(
            rate=self.dropout_rate_embeddings)(layer3_act_result_features)

        logits = tf.keras.layers.Dense(1,
                                       kernel_initializer=initializer(
                                           name=kernel_init_name, stddev=stddev),
                                       name="fc_z_mu_output"
                                       )(layer3_act_result)

        mf_model = tf.keras.Model(
            inputs=[dist_placeholder],
            outputs=[layer1_act_result, layer2_act_result_features,
                     layer3_act_result_features, logits],
            name="dist_discriminator"
        )

        return mf_model
