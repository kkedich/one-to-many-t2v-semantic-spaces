import tensorflow as tf

from text2video.models.operations import activation_function, initializer


class NonGaussianBaseMFProgressive:
    """
    Implements a progressive mapping where we get the intermediary features and apply a noise to
    prevent the mapping from same origin to the same point
    """

    def __init__(self,
                 input_dimension,
                 noise_dimension,
                 output_dimension,
                 hidden_layer_size,
                 dropout_rate_embeddings=0.5,
                 activation_fn_name="leaky_relu",
                 lrelu_alpha=0.2,
                 norm_type="layer",
                 use_noise_up_to_second_layer=False):

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.noise_dimension = noise_dimension
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate_embeddings

        self.activation_fn_name = activation_fn_name
        self.lrelu_alpha = lrelu_alpha
        self.norm_type = norm_type

        self.use_noise_up_to_second_layer = use_noise_up_to_second_layer

        assert norm_type in ["group", "layer"]

    def build_model(self):
        # New code for the mapping function model
        input_emb_placeholder = tf.keras.layers.Input(
            shape=[self.input_dimension], name="input_emb_placeholder")
        input_noise_placeholder = tf.keras.layers.Input(
            shape=[self.noise_dimension], name="input_noise")

        # Concatenated input
        concatenated_layer1 = tf.concat([input_emb_placeholder, input_noise_placeholder], axis=1)

        # Initializer variables
        kernel_init_name = "normal"
        stddev = 0.009  # 0.02, change to 0.02, as DialogWAE used, previous value was 0.009

        # Layer 1
        layer1_fc_output = tf.keras.layers.Dense(
            self.hidden_layer_size,
            kernel_initializer=initializer(name=kernel_init_name, stddev=stddev),
            name="layer1_fc_output")(concatenated_layer1)

        if self.norm_type == "group":
            # Axis to normalize is usually the features axis
            layer1_norm_output = tf.keras.layers.GroupNormalization(
                groups=32, epsilon=1e-6)(layer1_fc_output)
        else:
            layer1_norm_output = tf.keras.layers.LayerNormalization(axis=1)(layer1_fc_output)
        layer1_act_function_layer = activation_function(self.activation_fn_name, self.lrelu_alpha)
        layer1_act_result = layer1_act_function_layer(layer1_norm_output)

        # Layer 2
        if self.use_noise_up_to_second_layer:
            layer1_act_result = tf.concat([layer1_act_result, input_noise_placeholder], axis=1)

        layer2_fc_output = tf.keras.layers.Dense(
            self.hidden_layer_size,
            kernel_initializer=initializer(name=kernel_init_name, stddev=stddev),
            name="layer2_fc_output")(layer1_act_result)
        if self.norm_type == "group":
            # Axis to normalize is usually the features axis
            layer2_norm_output = tf.keras.layers.GroupNormalization(
                groups=32, epsilon=1e-6)(layer2_fc_output)
        else:
            layer2_norm_output = tf.keras.layers.LayerNormalization(axis=1)(layer2_fc_output)
        layer2_act_function_layer = activation_function(self.activation_fn_name, self.lrelu_alpha)
        layer2_act_result = layer2_act_function_layer(layer2_norm_output)
        layer2_act_result = tf.keras.layers.Dropout(rate=self.dropout_rate)(layer2_act_result)

        # Layer 3
        layer3_fc_output = tf.keras.layers.Dense(
            self.hidden_layer_size,
            kernel_initializer=initializer(name=kernel_init_name, stddev=stddev),
            name="layer3_fc_output")(layer2_act_result)

        if self.norm_type == "group":
            # Axis to normalize is usually the features axis
            layer3_norm_output = tf.keras.layers.GroupNormalization(
                groups=32, epsilon=1e-6)(layer3_fc_output)
        else:
            layer3_norm_output = tf.keras.layers.LayerNormalization(axis=1)(layer3_fc_output)
        layer3_act_function_layer = activation_function(self.activation_fn_name, self.lrelu_alpha)
        layer3_act_result = layer3_act_function_layer(layer3_norm_output)
        layer3_act_result = tf.keras.layers.Dropout(rate=self.dropout_rate)(layer3_act_result)

        dense_layer = tf.keras.layers.Dense(
            self.output_dimension,
            kernel_initializer=initializer(name=kernel_init_name, stddev=stddev),
            name="fc_z_mu_and_log_var_output"
        )
        outputs = dense_layer(layer3_act_result)

        mf_model = tf.keras.Model(
            inputs=[input_emb_placeholder, input_noise_placeholder],
            outputs=[outputs, layer1_act_result, layer2_act_result],
            name="mapping_function"
        )

        return mf_model
