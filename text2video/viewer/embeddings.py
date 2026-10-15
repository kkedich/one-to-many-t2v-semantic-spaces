import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from text2video.models.dimensionality_reduction import apply_pca, apply_tsne
from text2video.viewer.base import plot_at_ax

plt.rcParams.update({'figure.max_open_warning': 0})


def embeddings_to_dataframe(embeddings_list, labels=None, colors=None, nb_samples=None):
    current_label, row_start = 0, 0
    list_emb_df = []

    feat_cols = None
    for current_embeddings in embeddings_list:
        if nb_samples is not None:
            current_embeddings = current_embeddings[0:nb_samples]

        rows = row_start + current_embeddings.shape[0]
        feat_cols = ["feature" + str(i) for i in range(current_embeddings.shape[1])]

        emb_dataframe = pd.DataFrame(current_embeddings, columns=feat_cols)
        emb_dataframe["distribution"] = np.full((current_embeddings.shape[0]), current_label)
        emb_dataframe["distribution"] = emb_dataframe["distribution"].apply(lambda i: labels[i])
        emb_dataframe["color"] = np.full((current_embeddings.shape[0]), colors[current_label])
        list_emb_df.append(emb_dataframe)

        row_start = row_start + rows
        current_label += 1

    result_emb = pd.concat(list_emb_df)

    return result_emb, feat_cols


def plot_embeddings(codes_list, height=5, width=10,
                    numpy_version=False, output_path=None, scale=False, tag=None,
                    distributions=["mf_code", "video_code"], colors=["r", "b"],
                    put_id_on_points=True,
                    with_tsne=False, nb_iterations=500,
                    with_umap=True,
                    no_title=False,
                    current_dpi=None,
                    image_format="png"):

    if len(codes_list) != len(distributions) or len(distributions) != len(colors):
        raise ValueError(f"Mismatch of distributions ({len(distributions)}) or "
                         f"colors ({len(colors)}) and codes_list ({len(codes_list)})!")

    last_dimension = None
    nb_samples = None
    for current_embeddings in codes_list:
        current_dimension = current_embeddings.shape[1]
        nb_samples = current_embeddings.shape[0]

        if last_dimension is None:
            last_dimension = current_dimension
        elif last_dimension != current_dimension:
            raise ValueError("Codes list must have the same dimension on axis 1 to "
                             "be able to be considered for the generation of the data frame."
                             f"Last dimension found: {last_dimension},"
                             f" current dimension: {current_dimension}")

    # Check if the dataframe has a minimum of samples: min(n_samples, n_features)=20
    if nb_samples < 20:
        with_tsne = False

    emb_dataframe, feat_cols = embeddings_to_dataframe(
        codes_list,
        labels=distributions,
        colors=colors
    )

    # Reduce dimensionality
    # PCA
    df_pca_reduced_data, pca_cols = apply_pca(emb_dataframe, feat_cols=feat_cols, nb_components=3)
    df_pca_reduced_data["distribution"] = emb_dataframe["distribution"].values
    df_pca_reduced_data["color"] = emb_dataframe["color"].values

    df_tsne_reduced_data = None
    if with_tsne:
        # TSNE
        df_tsne_reduced_data = apply_tsne(
            emb_dataframe, feat_cols=feat_cols, nb_components=2, nb_iterations=nb_iterations)
        if df_tsne_reduced_data is not None:
            df_tsne_reduced_data["distribution"] = emb_dataframe["distribution"].values
            df_tsne_reduced_data["color"] = emb_dataframe["color"].values

    df_umap = None
    if with_umap:
        # UMAP
        embeddings_umap = umap.UMAP(random_state=42).fit_transform(emb_dataframe[feat_cols].values)
        df_umap = pd.DataFrame(
            index=np.arange(emb_dataframe.shape[0]), columns=["x-umap", "y-umap"])
        df_umap["x-umap"] = embeddings_umap[:, 0]
        df_umap["y-umap"] = embeddings_umap[:, 1]
        df_umap["distribution"] = emb_dataframe["distribution"].values
        df_umap["color"] = emb_dataframe["color"].values

    if no_title:
        title_pca = None
        title_tsne = None
        title_umap = None
    else:
        title_pca = "PCA" if tag is None else f"PCA {tag}"
        title_tsne = f"TSNE (iterations={nb_iterations})"
        title_umap = "UMAP"

    if (not with_tsne or (with_tsne and df_tsne_reduced_data is None)) and not with_umap:
        # Plot data
        # First we plot PCA results with all distributions
        fig, axes = plt.subplots(figsize=(width, height), ncols=1, nrows=1)

        plot_at_ax(df_pca_reduced_data, axes,
                   pca_cols[0], pca_cols[1], distributions, title_pca, put_id_on_points=put_id_on_points)

    elif with_tsne and not with_umap:
        fig, axes = plt.subplots(figsize=(width, height), ncols=2, nrows=1)
        ax1, ax2 = axes.ravel()

        plot_at_ax(df_pca_reduced_data, ax1,
                   pca_cols[0], pca_cols[1], distributions, title_pca, put_id_on_points=put_id_on_points)
        plot_at_ax(df_tsne_reduced_data, ax2, "x-tsne", "y-tsne", distributions,
                   title_tsne, put_id_on_points=put_id_on_points)

    elif not with_tsne and with_umap:
        fig, axes = plt.subplots(figsize=(width, height), ncols=2, nrows=1)
        ax1, ax2 = axes.ravel()

        plot_at_ax(df_pca_reduced_data, ax1,
                   pca_cols[0], pca_cols[1], distributions, title_pca, put_id_on_points=put_id_on_points)
        plot_at_ax(df_umap, ax2, "x-umap", "y-umap", distributions,
                   title_umap, put_id_on_points=put_id_on_points)

    else:
        fig, axes = plt.subplots(figsize=(width, height), ncols=3, nrows=1)
        ax1, ax2, ax3 = axes.ravel()

        plot_at_ax(df_pca_reduced_data, ax1,
                   pca_cols[0], pca_cols[1], distributions, title_pca, put_id_on_points=put_id_on_points)
        plot_at_ax(df_tsne_reduced_data, ax2, "x-tsne", "y-tsne", distributions,
                   title_tsne, put_id_on_points=put_id_on_points)
        plot_at_ax(df_umap, ax3, "x-umap", "y-umap", distributions,
                   title_umap, put_id_on_points=put_id_on_points)

    plt.subplots_adjust(top=0.92, bottom=-0.95, left=0.10, wspace=0.15)
    fig.tight_layout()

    if output_path is not None:
        if current_dpi is None:
            current_dpi = fig.dpi
        print(f"Setup for DPI: {current_dpi}, format: {image_format}")
        plt.savefig(output_path, format=image_format, dpi=current_dpi, bbox_inches="tight")

    plt.show()

    data = None
    if numpy_version:
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    if scale:
        # To avoid importing tensorflow for operations that do not require it
        from text2video.utils import scale_utils
        data = scale_utils.scale(data, undo=False, tensor=False)

    return data
