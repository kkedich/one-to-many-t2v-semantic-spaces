import tensorflow as tf

from text2video.models.operations import activation_function


def conv_block(input_tensor,
               filters,
               kernel_size,
               kernel_initializer,
               dropout,
               act_function,
               strides=2,
               lrelu_alpha=0.2,
               apply_dropout=True,
               apply_norm=True,
               type_norm_layer="batch_norm",
               padding=None,
               padding_type="same"):
    """ Creates a convolutional block composed of: the convolution operation,
    a padding over the output of the conv. operation (optional), a normalization
    method, and an activation function."""

    x = tf.keras.layers.Conv3D(filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding_type,
                               kernel_initializer=kernel_initializer)(input_tensor)

    if padding is not None:
        # Zero pad convolutional output, e.g., padding=(1, 1) or padding=1
        x = tf.keras.layers.ZeroPadding3D(padding=padding)(x)

    if apply_norm:
        if type_norm_layer == "batch_norm":
            x = tf.keras.layers.BatchNormalization()(x)
        elif type_norm_layer == "instance_norm":
            # x = tfa.layers.InstanceNormalization(axis=3)(x)
            x = tf.keras.layers.GroupNormalization(groups=-1, axis=3)(x)
        else:
            raise ValueError(f"Invalid normalization layer. Your option: '{type_norm_layer}'")

    x = activation_function(act_function, lrelu_alpha=lrelu_alpha)(x)

    if apply_dropout:
        x = tf.keras.layers.Dropout(rate=dropout)(x)

    print(x.shape)

    return x


class PatchDiscriminatorHDVideo:
    """
    Implementation of the discriminator of the PatchGAN model of Pix2PixHD.
    Instead of classifying the entire image, this discriminator
    classifies patches of the image at multiple scales.

    Architecture is defined in the following paper:
    High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs
    https://arxiv.org/abs/1711.11585

    Reference:
        - Official repository:
        https://github.com/NVIDIA/pix2pixHD/blob/5a2c87201c5957e2bf51d79b8acddb9cc1920b26/models/networks.py#L334

    """

    def __init__(self,
                 vid_shape,
                 dropout,
                 kernel_initializer,
                 act_function,
                 lrelu_alpha=0.2,
                 type_norm_layer="instance_norm",
                 name_suffix=None):
        """Initializes the PatchGANHD model attributes"""

        self.vid_shape = vid_shape
        self.act_function = act_function
        self.dropout = dropout
        self.lrelu_alpha = lrelu_alpha
        self.kernel_initializer = kernel_initializer
        self.type_norm_layer = type_norm_layer
        self.name_suffix = name_suffix

    def build_model(self):
        """Build the Patch Discriminator HD model"""

        input_video = tf.keras.layers.Input(shape=self.vid_shape, name="disc_input_video")

        common_params = {
            "act_function": self.act_function,
            "lrelu_alpha": self.lrelu_alpha,
            "kernel_initializer": self.kernel_initializer,
            "dropout": self.dropout,
            "padding_type": "valid",
            "type_norm_layer": self.type_norm_layer
        }

        down1 = conv_block(input_video, filters=64, kernel_size=4, strides=2,
                           apply_norm=False, padding=1, **common_params)

        down2 = conv_block(down1, filters=128, kernel_size=4, strides=2,
                           apply_norm=True, padding=1, **common_params)

        down3 = conv_block(down2, filters=256, kernel_size=4, strides=2,
                           apply_norm=True, padding=1, **common_params)

        conv4 = tf.keras.layers.Conv2D(filters=1, kernel_size=4, strides=1,
                                       padding='valid')(down3)
        print(conv4.shape)

        return tf.keras.Model(
            inputs=input_video,
            outputs=[down1, down2, down3, conv4],
            name=f"patch_hd_video_{self.name_suffix}"
        )
