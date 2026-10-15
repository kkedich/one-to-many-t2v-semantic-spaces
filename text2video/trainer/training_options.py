from text2video.models.feature_losses import load_vgg19_model
from text2video.utils.config import ConfigDict


def build_perceptual_options(training_options, original_options):

    # Load model for the perceptual loss calculation
    selected_layers = original_options.loss.selected_layers
    # ["block4_conv3"], ['block4_conv3', "block5_conv4"]
    # ["block3_conv3", 'block4_conv3'], ["block3_conv3" ,'block4_conv3','block5_conv4']
    selected_layer_weights = original_options.loss.selected_layer_weights
    vgg_model = load_vgg19_model(selected_layers=selected_layers,
                                 selected_layer_weights=selected_layer_weights,
                                 input_shape=(64, 64, 3))
    training_options.vgg_model = vgg_model
    training_options.selected_layer_weights = selected_layer_weights


def build_mf_stackedwae_options(cfg_mf_experiment):

    # Defining some parameters
    options = ConfigDict()

    # Loss
    cfg_model = cfg_mf_experiment.model

    # Only for KL loss enabled
    options.use_kl_loss = cfg_model.loss.get("use_kl_loss", False)
    options.kl_weight = cfg_model.loss.get("kl_weight", 2)

    options.reconstruction_delta = cfg_model.loss.get("reconstruction_delta", 10)
    options.reconstruction_shared_features_delta = cfg_model.loss.get(
        "reconstruction_shared_features_delta", 1)
    options.reconstruction_features_delta = cfg_model.loss.get("reconstruction_features_delta", 1)
    options.use_recons_vid_loss = cfg_model.loss.get("use_recons_vid_loss", False)
    options.use_recons_feat_loss = cfg_model.loss.get("use_recons_feat_loss", False)

    options.use_byol = cfg_model.loss.get("use_byol", False)
    options.byol_delta = cfg_model.loss.get("byol_delta", 1.0)

    # VicReg
    options.use_vicreg = cfg_model.loss.get("use_vicreg", False)
    options.batch_size = cfg_mf_experiment.train_cfg.batch_size
    options.vicreg = cfg_model.loss.vicreg

    # Sim1toN
    options.use_sim1ton = cfg_model.loss.get("use_sim1ton", False)
    options.use_sim1ton_in_videoshared = cfg_model.loss.get("use_sim1ton_in_videoshared", False)
    options.reconstruction_sim1ton = cfg_model.loss.get("reconstruction_sim1ton", 1)
    options.use_sim1ton_in_shared_space = cfg_model.loss.get("use_sim1ton_in_shared_space", False)
    options.use_cov_intermediary_layers = cfg_model.loss.get("use_cov_intermediary_layers", False)
    options.perfect_matching_weight = cfg_model.loss.get("perfect_matching_weight", None)
    options.pos_weight_sim1ton = cfg_model.loss.get("pos_weight_sim1ton", 1.0)
    options.neg_weight_sim1ton = cfg_model.loss.get("neg_weight_sim1ton", 1.0)

    options.use_dist_feat_loss = cfg_model.loss.get("use_dist_feat_loss", False)
    options.train_videoae_recons_feat_loss = cfg_model.loss.get(
        "train_videoae_recons_feat_loss", False)

    # Text2Shared
    text2shared_options = cfg_mf_experiment.model.get("text2shared", None)
    # VideoShared2Video
    shared2video_options = cfg_mf_experiment.model.get("shared2video", None)
    options.text2shared_options = text2shared_options
    options.shared2video_options = shared2video_options

    if shared2video_options.use_perceptual_loss or text2shared_options.use_perceptual_loss:
        options.perceptual_delta = cfg_model.loss.perceptual_delta
        build_perceptual_options(options, original_options=cfg_model)

    # Text
    options.sample_text_embedding = cfg_model.get("sample_text_embedding", False)
    options.sample_stddev = cfg_model.sample_stddev
    options.text_noise = cfg_model.get("text_noise", False)

    # Noise in text
    options.text_noise_probability = cfg_model.get("text_noise_probability", 0.0)
    options.text_noise_dropout_embeddings = cfg_model.get("text_noise_dropout_embeddings", 0.0)

    # Distribution discriminator
    dist_model = cfg_model.get("dist_discriminator", None)
    if dist_model is not None:
        options.dist_disc = ConfigDict()
        options.dist_disc.dist_loss_type = dist_model.gan_loss_type
        options.dist_disc.reconstruction_features_delta = dist_model.reconstruction_features_delta
        options.dist_disc.disc_model_type = dist_model.get("name", "dist_base")
        options.dist_disc.weight_g_loss = dist_model.get("weight_g_loss", 5.0)

    # Other losses
    options.use_recons_feat_vs2vcode_loss = cfg_model.loss.get(
        "use_recons_feat_vs2vcode_loss", False)

    options.use_infonce_loss = cfg_model.loss.get("use_infonce", False)
    options.use_only_specified_loss = cfg_model.loss.get("use_only_specified_loss", False)

    options.use_infonce_videoshared = cfg_model.loss.get(
        "use_infonce_videoshared", False)

    # Indicates if it is a SOTA model
    options.sota_model = cfg_model.loss.get("sota_model", False)

    options.use_stable_rep_video = cfg_model.loss.get("use_stable_rep_video", False)
    options.use_stable_rep_shared = cfg_model.loss.get("use_stable_rep_shared", False)

    return options


def build_non_progressive_options(cfg_mf_experiment):

    # Defining some parameters
    options = ConfigDict()

    # Loss
    cfg_model = cfg_mf_experiment.model

    # Only for KL loss enabled
    options.use_kl_loss = cfg_model.loss.get("use_kl_loss", False)
    options.kl_weight = cfg_model.loss.get("kl_weight", 2)

    options.reconstruction_delta = cfg_model.loss.get("reconstruction_delta", 10)
    options.reconstruction_features_delta = cfg_model.loss.get("reconstruction_features_delta", 1)
    options.use_recons_vid_loss = cfg_model.loss.get("use_recons_vid_loss", False)
    options.use_recons_feat_loss = cfg_model.loss.get("use_recons_feat_loss", False)

    options.use_byol = cfg_model.loss.get("use_byol", False)
    options.byol_delta = cfg_model.loss.get("byol_delta", 1.0)

    # VicReg
    options.use_vicreg = cfg_model.loss.get("use_vicreg", False)
    options.batch_size = cfg_mf_experiment.train_cfg.batch_size
    options.vicreg = cfg_model.loss.vicreg

    # Sim1toN
    options.use_sim1ton = cfg_model.loss.get("use_sim1ton", False)
    options.reconstruction_sim1ton = cfg_model.loss.get("reconstruction_sim1ton", 1)
    options.perfect_matching_weight = cfg_model.loss.get("perfect_matching_weight", None)
    options.pos_weight_sim1ton = cfg_model.loss.get("pos_weight_sim1ton", 1.0)
    options.neg_weight_sim1ton = cfg_model.loss.get("neg_weight_sim1ton", 1.0)

    options.use_dist_feat_loss = cfg_model.loss.get("use_dist_feat_loss", False)
    options.train_videoae_recons_feat_loss = cfg_model.loss.get(
        "train_videoae_recons_feat_loss", False)

    # MF options
    options.mf_model_options = cfg_model.get("mf_model", None)

    # Text
    options.sample_text_embedding = cfg_model.get("sample_text_embedding", False)
    options.sample_stddev = cfg_model.sample_stddev
    options.text_noise = cfg_model.get("text_noise", False)

    # Noise in text
    options.text_noise_probability = cfg_model.get("text_noise_probability", 0.0)
    options.text_noise_dropout_embeddings = cfg_model.get("text_noise_dropout_embeddings", 0.0)

    # Distribution discriminator
    dist_model = cfg_model.get("dist_discriminator", None)
    if dist_model is not None:
        options.dist_disc = ConfigDict()
        options.dist_disc.dist_loss_type = dist_model.gan_loss_type
        options.dist_disc.reconstruction_features_delta = dist_model.reconstruction_features_delta
        options.dist_disc.disc_model_type = dist_model.get("name", "dist_base")
        options.dist_disc.weight_g_loss = dist_model.get("weight_g_loss", 5.0)

    # Other losses
    options.use_infonce = cfg_model.loss.get("use_infonce", False)
    options.use_only_specified_loss = cfg_model.loss.get("use_only_specified_loss", False)
    options.use_stable_rep = cfg_model.loss.get("use_stable_rep", False)

    return options
