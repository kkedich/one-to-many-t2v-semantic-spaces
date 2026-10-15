import numpy as np
import tensorflow as tf

# Values when comparing equal samples
PNSR_VALUE_EQUAL_SAMPLES = np.inf
SSIM_VALUE_EQUAL_SAMPLES = 1.0


def dynamic_range(videos):
    return tf.reduce_max(videos) - tf.reduce_min(videos)


def ssim(original_videos, generated_videos, name="ssim_video"):
    # Values between [-1,1], when SSIM is 1 two images being compared are identical
    ssim_videos = tf.image.ssim(original_videos, generated_videos,
                                max_val=dynamic_range(original_videos))
    # We consider the ssim of a video as the mean value considering all frames
    # ssim_videos shape (number_videos, number_frames), one value per frame
    values_per_video = tf.reduce_mean(ssim_videos, name=name, axis=-1)
    ssim_mean = tf.reduce_mean(values_per_video)

    return values_per_video, ssim_mean


def psnr(original_videos, generated_videos, name="psnr_video"):
    # Large values are better
    # Note: equal videos are producing inf values
    psnr_per_frame_per_videos = tf.image.psnr(
        original_videos, generated_videos, max_val=dynamic_range(original_videos))
    # We consider the psnr of a video as the mean value considering all frames
    # psnr_videos shape (number_videos, number_frames), one value per frame
    values_per_video = tf.reduce_mean(psnr_per_frame_per_videos, name=name, axis=-1)
    psnr_mean = tf.reduce_mean(values_per_video)

    return values_per_video, psnr_mean
