"""
Implementation of the Frechet distance

Alternative?
- https://www.tensorflow.org/api_docs/python/tf/contrib/gan/eval/frechet_classifier_distance

- From https://github.com/bioinf-jku/TTUR/blob/master/fid.py
   - (issue https://github.com/bioinf-jku/TTUR/issues/4)
- https://github.com/bioinf-jku/TTUR/blob/16150b2f1ce217c7def60b9b7c1cd180fb6d476c/fid.py#L104
"""

import numpy as np
import warnings
from scipy import linalg


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the sample mean over activations for generated samples.
    -- mu2   : The sample mean over activations, precalculated on a representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on a representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Original and prediction mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Original and prediction covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    cov_mean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(cov_mean).all():
        msg = "FID calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        cov_mean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(cov_mean):
        if not np.allclose(np.diagonal(cov_mean).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_mean.imag))
            raise ValueError("Imaginary component {}".format(m))
        cov_mean = cov_mean.real

    tr_covmean = np.trace(cov_mean)
    frechet_distance = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return frechet_distance


def statistics_for_features(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)

    return mu, sigma


def calculate_fid_score(features_y_true, features_y_pred):

    assert np.array_equal(features_y_true.shape, features_y_pred.shape), \
        "Dimension mismatch between features of original {} and generated {} videos".format(
            features_y_true.shape, features_y_pred.shape)

    if features_y_true.shape[0] < features_y_true.shape[-1]:
        warnings.warn("Number of samples must be greater than or equal to "
                      "the features dimensionality for the covariance matrix be full ranked.")

    mu_y_true, sigma_y_true = statistics_for_features(features_y_true)
    mu_y_pred, sigma_y_pred = statistics_for_features(features_y_pred)

    fd_value = calculate_frechet_distance(mu_y_pred, sigma_y_pred, mu_y_true, sigma_y_true)
    print("FD: %s" % fd_value)

    fd_with_y_true = calculate_frechet_distance(mu_y_true, sigma_y_true, mu_y_true, sigma_y_true)
    print("FD real: %s" % fd_with_y_true)

    return fd_value, fd_with_y_true
