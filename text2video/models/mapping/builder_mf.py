from text2video.models.mapping.base import BASEMF, NonGaussianBaseMF
from text2video.models.mapping.dist_discriminator import DistDiscriminator
from text2video.models.mapping.progressive_mapping import \
    NonGaussianBaseMFProgressive
from text2video.models.mapping.projection_layer import (
    NonGaussianProjectionLayer, ProjectionLayer)


def create(name=None, options=None):

    if name == "base":
        # Creates the base model
        mf_model = BASEMF(
            text_dimension=options.text_dimension,
            code_dimension=options.code_dimension,
            hidden_layer_size=options.hidden_layer_size,
            dropout_rate_embeddings=options.dropout_rate_embeddings,
            activation_fn_name=options.get("activation_fn_name", "leaky_relu"),
            lrelu_alpha=0.2,
            split_mu_and_log_var=options.get("split_mu_and_log_var", False),
            norm_type=options.norm_type
        )
    elif name == "non_gaussian_base_mf":
        mf_model = NonGaussianBaseMF(
            input_dimension=options.input_dimension,
            output_dimension=options.output_dimension,
            hidden_layer_size=options.hidden_layer_size,
            dropout_rate_embeddings=options.dropout_rate_embeddings,
            activation_fn_name=options.get("activation_fn_name", "leaky_relu"),
            lrelu_alpha=0.2,
            norm_type=options.norm_type
        )
    elif name == "non_gaussian_base_mf_progressive":
        mf_model = NonGaussianBaseMFProgressive(
            input_dimension=options.input_dimension,
            noise_dimension=options.noise_dimension,
            output_dimension=options.output_dimension,
            hidden_layer_size=options.hidden_layer_size,
            dropout_rate_embeddings=options.dropout_rate_embeddings,
            activation_fn_name=options.get("activation_fn_name", "leaky_relu"),
            lrelu_alpha=0.2,
            norm_type=options.norm_type,
            use_noise_up_to_second_layer=options.get("use_noise_up_to_second_layer", False)
        )
    elif name == "projection_layer":
        # Creates the base model
        mf_model = ProjectionLayer(
            text_dimension=options.text_dimension,
            code_dimension=options.code_dimension,
            hidden_layer_size=options.hidden_layer_size,
            split_mu_and_log_var=options.get("split_mu_and_log_var", False),
            use_normalization=options.get("use_normalization", False)
        )
    elif name == "non_gaussian_projection_layer":
        # Creates the base model
        mf_model = NonGaussianProjectionLayer(
            input_dimension=options.input_dimension,
            output_dimension=options.output_dimension,
            hidden_layer_size=options.hidden_layer_size,
            use_normalization=options.get("use_normalization", False)
        )
    else:
        raise ValueError(f"Mapping function model <{name}> is not a valid option!")

    model = mf_model.build_model()
    model.summary()
    model.compiled_metrics = None

    return model


def create_dist_discriminator(name=None, options=None):

    if name == "dist_base":
        # Creates the base model
        dist_discriminator_model = DistDiscriminator(
            input_dimension=options.input_dimension,
            dropout_rate_embeddings=options.dropout_rate_embeddings,
            activation_fn_name="leaky_relu",
            lrelu_alpha=0.2
        )
        model = dist_discriminator_model.build_model()
        model.summary()

        return model
    else:
        raise ValueError(f"Distribution discriminator model <{name}> is not a valid option!")
