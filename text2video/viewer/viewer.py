import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

from text2video.utils import scale_utils


def show_img_on_ax(ax, img, scale=False, grayscale=True, visible_axis=True,
                   box_forced=True):
    ax.axis("off")
    if not visible_axis:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    if scale:
        img = scale_utils.scale(img, undo=True, tensor=False)
    if box_forced:
        ax.set_adjustable("box-forced")

    if grayscale:
        ax.imshow(img, aspect="equal", cmap="gray")
    else:
        ax.imshow(img, aspect="equal")


def view_result(list_videos, nrows, ncols, figsize=(10, 10), grayscale=True, scale=False,
                tag=None, filename=None):
    if len(list_videos) == nrows:
        all_images = np.concatenate(list_videos, axis=0)
    else:
        all_images = list_videos
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,
                             sharey=True, sharex=True,
                             gridspec_kw={"wspace": 0, "hspace": 0, "left": 0})

    if tag is not None:
        title = "Generated vs real (last row) ({})".format(tag)
    else:
        title = "Generated vs real (last row)"
    fig.suptitle(title, x=0.45, y=0.08, fontsize=12)

    for ax, img in zip(axes.flatten(), all_images):
        show_img_on_ax(ax, img, scale=scale, grayscale=grayscale,
                       visible_axis=False, box_forced=False)

    plt.subplots_adjust(top=0.06, bottom=-0.06, left=0.0, wspace=0.0, hspace=0.00)

    if filename is not None:
        plt.axis("off")
        print(f"Saving plot at <{filename}>")
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        # Clear current figure
        plt.clf()


def overview_experiment(generated, original,
                        videos_to_show=3, batch_index=None, figsize=(16, 16),
                        grayscale=True, scale=True, filename_prefix=None):
    generated = [generated]
    original = [original]

    assert len(generated) == len(original), \
        "Number of batches in <generated samples> is different " \
        "from number of batches in <original samples>"

    if batch_index is None:
        batch_index = len(generated) - 1
    print("Batches saved={}, number of videos in each batch={}. Showing batch {}".format(
        len(generated), generated[batch_index].shape[0], batch_index))

    videos = random.sample(range(generated[batch_index].shape[0]), videos_to_show)
    batch_sample = generated[batch_index]
    batch_original = original[batch_index]

    path_prefix = None
    samples_paths = []
    if filename_prefix is not None:
        path_prefix = Path(filename_prefix)

    new_shape = (16, 64, 64) if grayscale else (16, 64, 64, 3)
    for i in range(videos_to_show):
        tag = f"Last epoch - random video (index={videos[i]}) {i + 1}/{videos_to_show}"
        video_sample = np.reshape(batch_sample[videos[i]], new_shape)
        video_original = np.reshape(batch_original[videos[i]], new_shape)

        if filename_prefix is not None:
            filename = path_prefix.parent / f"{path_prefix.stem}_{i}{path_prefix.suffix}"
            samples_paths.append(filename)
        else:
            filename = None

        view_result((video_sample, video_original), 2, 16, tag=tag, figsize=figsize,
                    scale=scale, grayscale=grayscale, filename=filename)

    return samples_paths


def overview_experiment_single(generated, videos_to_show=3, batch_index=None, figsize=(16, 16),
                               grayscale=True, scale=True, filename_prefix=None):
    generated = [generated]
    if batch_index is None:
        batch_index = len(generated) - 1
    print("Batches saved={}, number of videos in each batch={}. Showing batch {}".format(
        len(generated), generated[batch_index].shape[0], batch_index))

    videos = random.sample(range(generated[batch_index].shape[0]), videos_to_show)
    batch_sample = generated[batch_index]

    path_prefix = None
    samples_paths = []
    if filename_prefix is not None:
        path_prefix = Path(filename_prefix)

    new_shape = (16, 64, 64) if grayscale else (16, 64, 64, 3)
    ncols = new_shape[0]
    tag = f"Overview {videos_to_show} videos"

    if filename_prefix is not None:
        filename = path_prefix.parent / f"{path_prefix.stem}_overview_{path_prefix.suffix}"
        samples_paths.append(filename)
    else:
        filename = None

    samples_list = []
    for i in range(videos_to_show):
        video_sample = np.reshape(batch_sample[videos[i]], new_shape)
        samples_list.append(video_sample)

    nrows = len(samples_list)
    if len(samples_list) == nrows:
        all_images = np.concatenate(samples_list, axis=0)
    else:
        all_images = samples_list
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, sharey=True, sharex=True,
                             gridspec_kw={"wspace": 0, "hspace": 0, "left": 0})
    fig.suptitle(tag, x=0.225, y=0.08, fontsize=10)

    for ax, img in zip(axes.flatten(), all_images):
        show_img_on_ax(ax, img, scale=scale, grayscale=grayscale,
                       visible_axis=False, box_forced=False)

    # Adjust right parameter to remove the blank space
    plt.subplots_adjust(top=0.06, bottom=-0.06, left=0.0, wspace=0.0, hspace=0.00, right=0.45)

    if filename is not None:
        plt.axis("off")
        print(f"Saving plot at <{filename}>")
        plt.savefig(filename, bbox_inches="tight", pad_inches=0)
        # Clear current figure
        plt.clf()

    return samples_paths
