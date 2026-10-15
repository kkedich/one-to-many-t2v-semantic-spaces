"""
Implementation of the Kernel Video Distance (KVD) from https://arxiv.org/pdf/1812.01717
Towards Accurate Generative Models of Video: A New Metric & Challenges

Reference:
- LVDM code of KVD
"""
import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel


def polynomial_mmd(fake_embeddings, real_embeddings):
    m = fake_embeddings.shape[0]
    n = real_embeddings.shape[0]

    # Compute kernels
    K_XX = polynomial_kernel(fake_embeddings)
    K_YY = polynomial_kernel(real_embeddings)
    K_XY = polynomial_kernel(fake_embeddings, real_embeddings)

    # Compute mmd distance
    K_XX_sum = (K_XX.sum() - np.diagonal(K_XX).sum()) / (m * (m - 1))
    K_YY_sum = (K_YY.sum() - np.diagonal(K_YY).sum()) / (n * (n - 1))
    K_XY_sum = K_XY.sum() / (m * n)
    mmd = K_XX_sum + K_YY_sum - 2 * K_XY_sum

    print(f"KVD = {mmd:.4f}")

    return mmd
