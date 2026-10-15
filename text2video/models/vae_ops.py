import numpy as np
import tensorflow as tf


def reparameterization_trick(z_mu, z_log_var, stddev=1.0):

    if z_mu is None or z_log_var is None:
        raise ValueError(f"Some value is defined as None: z_mu={z_mu} and z_log_var={z_log_var}")

    # Sampling from WAE paper, that clips the encoder sigma
    # https://github.com/tolstikhin/wae/blob/master/wae.py#L61
    z_log_var = tf.clip_by_value(z_log_var, -50, 50)

    input_shape = z_mu.get_shape().as_list()
    eps = tf.random.normal(
        input_shape,  # (sample_size, code_dimension),
        mean=0.,
        stddev=stddev,
        dtype=tf.float32,
        name="encoder_eps"
    )

    # z_mu + (eps * tf.exp(0.5 * z_log_var))
    element = tf.multiply(0.5, z_log_var)
    code = tf.add(z_mu, tf.multiply(eps, tf.exp(element)))

    return code


def sample_from_distribution(type_dist, nb_samples, z_dimension):

    if type_dist == "normal":
        mean = np.zeros(z_dimension)
        cov = np.identity(z_dimension)
        z_real_dist = np.random.multivariate_normal(mean, cov, nb_samples).astype(np.float32)

    elif type_dist == "uniform":
        z_real_dist = np.random.uniform(-1, 1, [nb_samples, z_dimension]).astype(np.float32)
    else:
        raise NotImplementedError

    return z_real_dist
