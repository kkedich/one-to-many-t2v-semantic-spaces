import tensorflow as tf

from text2video.models.operations import initializer


class ProjectionLayer:
    """
    Implements a vanilla mapping function between text representations
    to the video representations.
    """

    def __init__(self,
                 text_dimension,
                 code_dimension,
                 hidden_layer_size,
                 split_mu_and_log_var=False,
                 use_normalization=False):

        self.text_dimension = text_dimension
        self.code_dimension = code_dimension
        self.hidden_layer_size = hidden_layer_size
        self.split_mu_and_log_var = split_mu_and_log_var

        self.normalize = use_normalization

    def build_model(self):
        # New code for the mapping function model
        text_emb_placeholder = tf.keras.layers.Input(
            shape=[self.text_dimension], name="text_emb_placeholder")

        # Initializer variables
        kernel_init_name = "normal"
        stddev = 0.009  # 0.02
        # change to 0.02, as DialogWAE used, previous value was 0.009

        input_tensor = text_emb_placeholder
        if self.normalize:
            # From ImageBind
            # https://github.com/facebookresearch/ImageBind/blob/3fcf5c9039de97f6ff5528ee4a9dce903c5979b3/imagebind/models/imagebind_model.py#L366
            input_tensor = tf.keras.layers.LayerNormalization(axis=1, epsilon=1e-6)(input_tensor)

        if not self.split_mu_and_log_var:
            fc_z_mu_output = tf.keras.layers.Dense(
                self.code_dimension,
                kernel_initializer=initializer(name=kernel_init_name, stddev=stddev),
                name="fc_z_mu_output",
                use_bias=False
            )
            fc_z_log_var_output = tf.keras.layers.Dense(
                self.code_dimension,
                kernel_initializer=initializer(name=kernel_init_name, stddev=stddev),
                name="fc_z_log_var_output",
                use_bias=False
            )
            z_mu = fc_z_mu_output(input_tensor)
            z_log_var = fc_z_log_var_output(input_tensor)
        else:
            dense_layer = tf.keras.layers.Dense(
                self.code_dimension * 2,
                kernel_initializer=initializer(name=kernel_init_name, stddev=stddev),
                name="fc_z_mu_and_log_var_output",
                use_bias=False
            )
            outputs = dense_layer(input_tensor)
            z_mu, z_log_var = tf.split(outputs, 2, axis=-1)

        mf_model = tf.keras.Model(
            inputs=[text_emb_placeholder],
            outputs=[z_mu, z_log_var],
            name="mapping_function"
        )

        return mf_model


class NonGaussianProjectionLayer:
    """
    Implements a vanilla and non gaussian projection layer between text representations
    to the video representations.
    """

    def __init__(self,
                 input_dimension,
                 output_dimension,
                 hidden_layer_size,
                 use_normalization=False):

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.hidden_layer_size = hidden_layer_size

        self.normalize = use_normalization

    def build_model(self):
        # New code for the mapping function model
        input_emb_placeholder = tf.keras.layers.Input(
            shape=[self.input_dimension], name="input_embeddings_placeholder")

        # Initializer variables
        kernel_init_name = "normal"
        stddev = 0.009  # 0.02
        # change to 0.02, as DialogWAE used, previous value was 0.009

        input_tensor = input_emb_placeholder
        if self.normalize:
            # From ImageBind
            # https://github.com/facebookresearch/ImageBind/blob/3fcf5c9039de97f6ff5528ee4a9dce903c5979b3/imagebind/models/imagebind_model.py#L366
            input_tensor = tf.keras.layers.LayerNormalization(axis=1, epsilon=1e-6)(input_tensor)

        dense_layer = tf.keras.layers.Dense(
            self.output_dimension,
            kernel_initializer=initializer(name=kernel_init_name, stddev=stddev),
            name="fc_z_mu_and_log_var_output",
            use_bias=False
        )
        outputs = dense_layer(input_tensor)

        mf_model = tf.keras.Model(
            inputs=[input_emb_placeholder],
            outputs=[outputs],
            name="projlayer"
        )

        return mf_model
