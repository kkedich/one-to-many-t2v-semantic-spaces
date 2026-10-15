import numpy as np


def videos2rgb(videos: np.ndarray):
    if len(videos.shape) == 5 and videos.shape[-1] == 1:
        videos = np.reshape(videos, newshape=videos.shape[0:4])

    nb_videos, frames, w, h = videos.shape
    new = np.empty((nb_videos, frames, w, h, 3), dtype=np.float32)
    new[:, :, :, :, 0] = videos
    new[:, :, :, :, 1] = videos
    new[:, :, :, :, 2] = videos

    return new
