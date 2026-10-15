# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Loads video samples: fake and real and calculate the FVD and KVD metrics.
Do note that is recommended to have the number os samples bigger than the
number of features considered in the output layer or embeddings size.

Usage:
python ./text2video/vqa/src/kinects_i3d/evaluate_sample.py \
       --full_input="False" \
       --file_original_videos="$real_videos" \
       --file_generated_videos="$fake_videos" \
       --output_file="$output_file_fvd"


Notes:
    - Model reference from https://github.com/google-deepmind/kinetics-i3d
      - rgb_imagenet: https://github.com/google-deepmind/kinetics-i3d/tree/master/data/checkpoints/rgb_imagenet

"""

from __future__ import absolute_import, division, print_function

import i3d
import json
import numpy as np
import pickle as pkl
import tensorflow as tf
import time
# from multiprocessing import Pool
from pathlib import Path

from text2video.metrics.iqa import frechet_distance
from text2video.metrics.kvd import polynomial_mmd
from vidutils.src.convert import videos2rgb
from vidutils.src.tensor_utils import tf_resize_videos

_IMAGE_SIZE = 224
_BATCH_SIZE = 1
_SAMPLE_VIDEO_FRAMES = 16

_CHECKPOINT_PATHS = {
    'rgb': '../data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': './data/checkpoints/rgb_scratch_kin600/model.ckpt',
    # 'flow': './data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': '/work/text2video/vqa/src/kinects_i3d/data/checkpoints/rgb_imagenet/model.ckpt',
    # 'flow_imagenet': './data/checkpoints/flow_imagenet/model.ckpt',
}

FLAGS = tf.flags.FLAGS
# name, default value, help
tf.flags.DEFINE_string('eval_type', 'rgb',
                       'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')
# Shape of original and generated files (number of videos, number of frames, width, height, channels)
# Example: (6624, 16, 64, 64, 1)

# file_original_videos and file_generated_videos can be folders as well
tf.flags.DEFINE_string('file_original_videos', None,
                       'File or folder containing the numpy arrays of the original videos (ground-truth).'
                       'Already pre-processed to be in interval (-1,1)')
tf.flags.DEFINE_string('file_generated_videos', None,
                       'File or folder containing the numpy arrays of the generated videos.'
                       'Already pre-processed to be in interval (-1,1)')
tf.flags.DEFINE_string('output_file', None,
                       'Output file where we are going to save the FID values (e.g., .pickle or .json file)')
tf.flags.DEFINE_string("output_layer", "input_to_Logits",
                       'Output layer from the i3D model to consider to calculate the FVD metric. '
                       'Valid options "input_to_Logits" and "Logits".')


def get_features(videos, resize, session,
                 rgb_input_original, rgb_input, rgb_i3d_features):

    # Get features (activations) for the generated videos
    print("Extraction of features for the videos starting...")
    features_videos = np.empty((videos.shape[0], 1024))
    for i_vid in range(videos.shape[0]):
        feed_dict = {}
        # Expand dimension of the batch
        rgb_sample = np.expand_dims(videos[i_vid], axis=0)

        if resize:
            feed_dict[rgb_input_original] = rgb_sample
        else:
            feed_dict[rgb_input] = rgb_sample

        if i_vid % 100 == 0:
            print(f"Extraction of i_vid={i_vid}/{videos.shape[0]}, rgb_sample={rgb_sample.shape},"
                  f" rgb_i3d_features={rgb_i3d_features.shape}")

        out_features = session.run(rgb_i3d_features, feed_dict=feed_dict)
        features_videos[i_vid] = out_features[0]
    print("Extraction of features for the videos done!")

    return features_videos


def get_files_into_array(path_files):
    # Read first file to allocate memory according to the shape
    # and reduce processing time
    first_data = np.load(path_files[0])
    target_shape = [len(path_files)]
    target_shape.extend(list(first_data.shape[1:]))
    all_videos = np.zeros(target_shape)

    for i, current_file in enumerate(path_files):
        all_videos[i] = np.load(current_file)

        if i % 150 == 0:
            print("Read {}/{} files".format(i, len(path_files)))
    print(all_videos.shape)

    return all_videos


def read_numpy(input_file: str):
    current_data = None
    if input_file.endswith('npz'):
        current_data = np.load(input_file)['arr_0']
    elif input_file.endswith('npy'):
        current_data = np.load(input_file)

    return current_data


def check_scale(input_videos):
    scaled_videos = input_videos
    if np.min(input_videos) >= 0 and np.max(input_videos) <= 255:
        print("Scaling from [0,255] to [-1,1]")
        scaled_videos = 2. * input_videos / 255. - 1  # [-1, 1]
    elif np.min(input_videos) >= 0 and np.max(input_videos) <= 1:
        # Scale to [-1, 1]
        print("Scaling from [0,1] to [-1,1]")
        scaled_videos = (input_videos * 2) + -1

    return scaled_videos


def main(unused_argv):
    start = time.time()

    tf.logging.set_verbosity(tf.logging.INFO)

    eval_type = FLAGS.eval_type
    imagenet_pretrained = FLAGS.imagenet_pretrained
    file_original_videos = FLAGS.file_original_videos
    file_generated_videos = FLAGS.file_generated_videos
    output_file = FLAGS.output_file
    output_layer = FLAGS.output_layer

    if output_file is None:
        raise ValueError(f"Output file: <{output_file}> has no value!")

    output_path_obj = Path(output_file)
    if not output_path_obj.parent.exists():
        # Create folder
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    original_vids_path = Path(file_original_videos)
    generated_vids_path = Path(file_generated_videos)

    # Pre-process the videos
    print("\nPreprocessing videos...")
    print(f"original_vids_path.is_file(): {original_vids_path.is_file()}")
    print(f"original_vids_path.is_dir(): {original_vids_path.is_dir()}")

    if original_vids_path.is_file() and generated_vids_path.is_file():
        input_true = read_numpy(str(file_original_videos))
        input_pred = read_numpy(str(file_generated_videos))

        if input_true.shape[0] < input_pred.shape[0]:
            input_pred = input_pred[0:input_true.shape[0], :, :]
            print("Changing input_pred={}".format(input_pred.shape))
        elif input_true.shape[0] > input_pred.shape[0]:
            input_true = input_true[0:input_pred.shape[0], :, :]
            print("Changing input_true={}".format(input_true.shape))

    elif original_vids_path.is_dir() and generated_vids_path.is_dir():
        # Is actually a directory
        prediction_files = sorted(generated_vids_path.glob("*.npy"))
        print("Found {} prediction files".format(len(prediction_files)))
        gt_files = sorted(original_vids_path.glob("*.npy"))
        print("Found {} original/gt files".format(len(gt_files)))

        print("Merging files into one array...")
        # with Pool(2) as pool:
        #     result = pool.map(, [prediction_files, gt_files])
        # print("Results have length: {}".format(len(result)))
        input_pred = get_files_into_array(prediction_files)
        input_true = get_files_into_array(gt_files)
    else:
        raise ValueError("Input files does not follow the required types! "
                         f"gt: {original_vids_path}, fake: {generated_vids_path}")

    print("input_true={}, input_pred={}\n".format(input_true.shape, input_pred.shape))
    assert np.array_equal(input_pred.shape, input_true.shape), \
        "Dimension mismatch between input_true={} and input_pred={} videos".format(
            input_true.shape, input_pred.shape)

    # Check if video is already on [-1, 1]
    print("Input true, min={}, max={}".format(np.min(input_true), np.max(input_true)))
    print("Input prediction, min={}, max={}".format(np.min(input_pred), np.max(input_pred)))
    input_true = check_scale(input_true)
    input_pred = check_scale(input_pred)
    print("Input true, min={}, max={}".format(np.min(input_true), np.max(input_true)))
    print("Input prediction, min={}, max={}\n".format(np.min(input_pred), np.max(input_pred)))

    # Check color
    if input_true.shape[-1] == 1:
        input_true = videos2rgb(input_true)
        input_pred = videos2rgb(input_pred)
    print("\nChecking color: input_true={}, input_pred={}".format(input_true.shape, input_pred.shape))

    # Node operation to resize the videos if needed
    real_height = input_true.shape[-3]
    real_width = input_true.shape[-2]
    resize = False
    if real_height != _IMAGE_SIZE or real_width != _IMAGE_SIZE:
        resize = True

    # Define model
    num_classes = 400
    if eval_type == 'rgb600':
        num_classes = 600

    if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
        raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

    rgb_input_original, rgb_input, rgb_saver = None, None, None
    rgb_i3d_features = None
    if eval_type in ['rgb', 'rgb600', 'joint']:
        if resize:
            rgb_input_original = tf.placeholder(tf.float32, shape=(
                _BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, real_height, real_width, 3))
            print("resize: rgb_input_original={}".format(rgb_input_original))
            rgb_input = tf_resize_videos(
                rgb_input_original, max_batch_size=_BATCH_SIZE, target_size=(_IMAGE_SIZE, _IMAGE_SIZE))
            print("resize: rgb_input={}".format(rgb_input))
        else:
            # RGB input has 3 channels.
            rgb_input = tf.placeholder(tf.float32, shape=(
                _BATCH_SIZE, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(num_classes, spatial_squeeze=True, final_endpoint='Logits')
            _, end_points = rgb_model(rgb_input, is_training=False, dropout_keep_prob=1.0)

            print("end_points keys={}".format(end_points.keys()))
            for key_entry in end_points.keys():
                print("end_points[{}]={}".format(key_entry, end_points[key_entry].shape))
            print("\n")

            rgb_i3d_features = end_points[output_layer]

        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                if eval_type == 'rgb600':
                    rgb_variable_map[variable.name.replace(
                        ':0', '')[len('RGB/inception_i3d/'):]] = variable
                else:
                    rgb_variable_map[variable.name.replace(':0', '')] = variable

        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    with tf.Session(config=sess_config) as sess:
        if eval_type in ['rgb', 'rgb600', 'joint']:
            if imagenet_pretrained:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            else:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
            tf.logging.info('RGB checkpoint restored')

        # Get features (activations)
        print("Getting features of true/real videos...")
        features_true = get_features(input_true, resize, sess,
                                     rgb_input_original, rgb_input, rgb_i3d_features)
        print("Getting features of predicted videos...")
        features_pred = get_features(input_pred, resize, sess,
                                     rgb_input_original, rgb_input, rgb_i3d_features)
        print("features_orig={}, features_gen={}".format(features_true.shape, features_pred.shape))

        if features_pred.shape[0] <= features_pred.shape[1]:
            print(f"Size of samples is smaller than the embedding size. "
                  f"Samples: {features_pred.shape[0]} vs embedding size: {features_pred.shape[1]}")

        # Calculate the FVD score
        fvd_value, fvd_with_real = frechet_distance.calculate_fid_score(
            features_y_true=features_true,
            features_y_pred=features_pred
        )

        # Calculating KVD score
        print("Calculating KVD score...")
        kvd = polynomial_mmd(features_pred, features_true)

        result = {
            "FVD_value": fvd_value,
            "FVD_with_real": fvd_with_real,
            "KVD_value": kvd
        }

        if output_file.endswith(".json"):
            with open(output_file, 'w') as file_results:
                json.dump(result, file_results)
        else:
            with open(output_file, "wb") as file_results:
                pkl.dump(result, file_results)
        print("FVD/KVD results saved at {}".format(output_file))

    print(f"It took {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    tf.app.run(main)
