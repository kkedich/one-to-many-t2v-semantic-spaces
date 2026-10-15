import tensorflow as tf

from text2video.models.operations import activation_function, initializer


class BASEMF:
    """
    Implements a vanilla mapping function between text representations
    to the video representations.
    """

    def __init__(self,
                 text_dimension,
                 code_dimension,
                 hidden_layer_size,
                 dropout_rate_embeddings=0.5,
                 activation_fn_name="leaky_relu",
                 lrelu_alpha=0.2,
                 split_mu_and_log_var=False,
                 norm_type="layer"):

        self.text_dimension = text_dimension
        self.code_dimension = code_dimension
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate_embeddings = dropout_rate_embeddings
        self.split_mu_and_log_var = split_mu_and_log_var

        self.activation_fn_name = activation_fn_name
        self.lrelu_alpha = lrelu_alpha
        self.norm_type = norm_type

        assert norm_type in ["group", "layer"]

    def build_model(self):
        # New code for the mapping function model
        text_emb_placeholder = tf.keras.layers.Input(
            shape=[self.text_dimension], name="text_emb_placeholder")
        # video_emb_placeholder = tf.keras.layers.Input(shape=[code_dimension], name="video_emb_placeholder")

        # Initializer variables
        kernel_init_name = "normal"
        stddev = 0.009  # 0.02
        # change to 0.02, as DialogWAE used, previous value was 0.009

        # Video embeddings are following a gaussian, maybe sample from a Gaussian
        # and generate the video code

        # Layer 1
        layer1_fc_output = tf.keras.layers.Dense(
            self.hidden_layer_size,  # int(self.hidden_layer_size / 2),
            kernel_initializer=initializer(name=kernel_init_name, stddev=stddev),
            name="layer1_fc_output")(text_emb_placeholder)

        if self.norm_type == "group":
            # Axis to normalize is usually the features axis
            layer1_norm_output = tf.keras.layers.GroupNormalization(
                groups=32, epsilon=1e-6)(layer1_fc_output)
        else:
            layer1_norm_output = tf.keras.layers.LayerNormalization(axis=1)(layer1_fc_output)
        layer1_act_function_layer = activation_function(self.activation_fn_name, self.lrelu_alpha)
        layer1_act_result = layer1_act_function_layer(layer1_norm_output)

        # Layer 2
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
        layer2_act_result = tf.keras.layers.Dropout(
            rate=self.dropout_rate_embeddings)(layer2_act_result)

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
        layer3_act_result = tf.keras.layers.Dropout(
            rate=self.dropout_rate_embeddings)(layer3_act_result)

        if not self.split_mu_and_log_var:
            fc_z_mu_output = tf.keras.layers.Dense(
                self.code_dimension,
                kernel_initializer=initializer(name=kernel_init_name, stddev=stddev),
                name="fc_z_mu_output"
            )
            fc_z_log_var_output = tf.keras.layers.Dense(
                self.code_dimension,
                kernel_initializer=initializer(name=kernel_init_name, stddev=stddev),
                name="fc_z_log_var_output"
            )
            z_mu = fc_z_mu_output(layer3_act_result)
            z_log_var = fc_z_log_var_output(layer3_act_result)
        else:
            dense_layer = tf.keras.layers.Dense(
                self.code_dimension * 2,
                kernel_initializer=initializer(name=kernel_init_name, stddev=stddev),
                name="fc_z_mu_and_log_var_output"
            )
            outputs = dense_layer(layer3_act_result)
            z_mu, z_log_var = tf.split(outputs, 2, axis=-1)

        # print(f"BaseMF builindg: z_mu: {z_mu.shape}, z_log_var: {z_log_var.shape}")
        mf_model = tf.keras.Model(
            inputs=[text_emb_placeholder],
            outputs=[z_mu, z_log_var],
            name="mapping_function"
        )

        return mf_model


class NonGaussianBaseMF:
    """
    Implements a vanilla non gaussian mapping function between two representations
    """

    def __init__(self,
                 input_dimension,
                 output_dimension,
                 hidden_layer_size,
                 dropout_rate_embeddings=0.5,
                 activation_fn_name="leaky_relu",
                 lrelu_alpha=0.2,
                 norm_type="layer",
                 dropout_rate_text_noise=0.0):

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate_embeddings

        self.activation_fn_name = activation_fn_name
        self.lrelu_alpha = lrelu_alpha
        self.norm_type = norm_type

        # Noise by dropout on embeddings
        self.dropout_rate_text_noise = dropout_rate_text_noise

        assert norm_type in ["group", "layer"]

    def build_model(self):
        # New code for the mapping function model
        input_emb_placeholder = tf.keras.layers.Input(
            shape=[self.input_dimension], name="input_emb_placeholder")

        # Initializer variables
        kernel_init_name = "normal"
        stddev = 0.009  # 0.02
        # change to 0.02, as DialogWAE used, previous value was 0.009

        text_input = input_emb_placeholder
        if self.dropout_rate_text_noise > 0.0:
            # Apply dropout on text embeddings before passing through the network
            text_input = tf.keras.layers.Dropout(rate=self.dropout_rate_text_noise)(text_input)

        # Layer 1
        layer1_fc_output = tf.keras.layers.Dense(
            self.hidden_layer_size,
            kernel_initializer=initializer(name=kernel_init_name, stddev=stddev),
            name="layer1_fc_output")(text_input)

        if self.norm_type == "group":
            # Axis to normalize is usually the features axis
            layer1_norm_output = tf.keras.layers.GroupNormalization(
                groups=32, epsilon=1e-6)(layer1_fc_output)
        else:
            layer1_norm_output = tf.keras.layers.LayerNormalization(axis=1)(layer1_fc_output)
        layer1_act_function_layer = activation_function(self.activation_fn_name, self.lrelu_alpha)
        layer1_act_result = layer1_act_function_layer(layer1_norm_output)

        # Layer 2
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

        if self.dropout_rate_text_noise > 0.0:
            mf_model = tf.keras.Model(
                inputs=[input_emb_placeholder],
                outputs=[outputs, text_input],
                name="mapping_function"
            )
        else:
            mf_model = tf.keras.Model(
                inputs=[input_emb_placeholder],
                outputs=[outputs],
                name="mapping_function"
            )

        return mf_model
