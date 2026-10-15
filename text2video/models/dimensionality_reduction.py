import copy
import logging
import numpy as np
import pandas as pd
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

np.random.seed(42)


def apply_pca(data_frame, feat_cols, nb_components=3):
    logging.debug(f"PCA: reducing to {nb_components} components...")
    pca = PCA(n_components=nb_components)
    pca_result = pca.fit_transform(data_frame[feat_cols].values)

    pca_cols = ["pca-" + str(i) for i in range(nb_components)]
    reduced_data = pd.DataFrame(index=np.arange(data_frame.shape[0]), columns=pca_cols)
    for i in range(nb_components):
        reduced_data[pca_cols[i]] = pca_result[:, i]
    logging.debug(f"Explained variation per principal component: {pca.explained_variance_ratio_}")

    return reduced_data, pca_cols


def apply_tsne(data_frame, feat_cols, nb_components=2, nb_iterations=600):

    reduced_data = copy.deepcopy(data_frame)
    n_sne = reduced_data.shape[0]

    if len(feat_cols) > 32:
        # Reduce data to 32 dimensions
        reduced_data, feat_cols = apply_pca(data_frame, feat_cols, nb_components=32)

    print("\nt-SNE, n_sne={}".format(n_sne))

    time_start = time.time()
    tsne_op = TSNE(n_components=nb_components, verbose=1, perplexity=40,
                   max_iter=nb_iterations, random_state=42)
    tsne_results = tsne_op.fit_transform(reduced_data.loc[:, feat_cols].values)

    print(f"t-SNE done! Time elapsed: {time.time() - time_start:.2f} seconds")

    df_tsne = pd.DataFrame(index=np.arange(data_frame.shape[0]), columns=["x-tsne", "y-tsne"])
    df_tsne["x-tsne"] = tsne_results[:, 0]
    df_tsne["y-tsne"] = tsne_results[:, 1]

    return df_tsne


def apply_umap(data_frame, feat_cols):
    import umap as umap_lib

    print("\nUMAP...")
    time_start = time.time()
    embeddings = umap_lib.UMAP(random_state=42).fit_transform(data_frame[feat_cols].values)
    print(f"UMAP done! Time elapsed: {time.time() - time_start:.2f} seconds")

    df_umap = pd.DataFrame(index=np.arange(data_frame.shape[0]), columns=["x-umap", "y-umap"])
    df_umap["x-umap"] = embeddings[:, 0]
    df_umap["y-umap"] = embeddings[:, 1]

    return df_umap
