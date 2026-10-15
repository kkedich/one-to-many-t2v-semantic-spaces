import math
import numpy as np
import sys
import tensorflow as tf
from pathlib import Path

from text2video.dataset import record
from text2video.utils import image, scale_utils


def example_bucket(video, text, sequence_length):
    video_feature = {}
    height, width = None, None

    for frame in range(0, sequence_length):
        img = video[frame]
        height = img.shape[0]
        width = img.shape[1]

        img_raw = img.tostring()
        video_feature[f"{frame}"] = record._bytes_feature(img_raw)

    video_feature["height"] = record._int64_feature(height)
    video_feature["width"] = record._int64_feature(width)
    video_feature["text"] = record._bytes_feature(tf.compat.as_bytes(text))

    example = tf.train.Example(features=tf.train.Features(feature=video_feature))

    return example


def write_all_buckets(data_per_text, tfrecord_filename, sequence_length=16):
    count = 0

    with tf.io.TFRecordWriter(tfrecord_filename) as tfrecord_writer:
        for text in data_per_text:
            print(f"\nWriting videos of <{text}>")
            for video_index in range(data_per_text[text]["videos"].shape[0]):
                sys.stdout.write('\r>> Converting video %d/total=%d' %
                                 (video_index + 1, data_per_text[text]["videos"].shape[0]))
                sys.stdout.flush()

                example = example_bucket(video=data_per_text[text]["videos"][video_index],
                                         text=text,
                                         sequence_length=sequence_length)
                tfrecord_writer.write(example.SerializeToString())
                count += 1

    sys.stdout.write('\n')
    sys.stdout.flush()


def write_bucket(data, text, tfrecord_filename, sequence_length=16):
    count = 0

    with tf.io.TFRecordWriter(tfrecord_filename) as tfrecord_writer:
        for video_index in range(data.shape[0]):
            sys.stdout.write('\r>> Converting video %d/total=%d' % (video_index + 1, data.shape[0]))
            sys.stdout.flush()

            example = example_bucket(video=data[video_index],
                                     text=text,
                                     sequence_length=sequence_length)
            tfrecord_writer.write(example.SerializeToString())
            count += 1

    sys.stdout.write('\n')
    sys.stdout.flush()


def write_from_npy(npy_data, tfrecord_file, tag, sequence_length=16, num_shards=1,
                   overwrite=True, max_videos=None,
                   with_text=False, text_data=None, options=None):

    if not overwrite:
        tfrecord_files = list(Path(tfrecord_file).parent.glob("*.tfrecord"))
        if len(tfrecord_files) > 0:
            print("TFRecord file <{}> already exists.\n {}\nNothing to be done. Returning.".format(
                tfrecord_file, tfrecord_files))
            return tfrecord_files, -1, -1, -1

    if max_videos is None:
        max_videos = len(npy_data)

    print("npy_data.shape={}, dtype={}".format(npy_data.shape, npy_data.dtype))

    videos_per_shard = int(math.ceil(max_videos / float(num_shards)))

    height, width, nb_frames = None, None, None
    tfrecord_files = []
    count = 0

    file_path = Path(tfrecord_file)
    filename_without_suffix = file_path.parent / file_path.stem

    for shard_id in range(num_shards):
        output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
            filename_without_suffix, tag, shard_id, num_shards - 1)
        tfrecord_files.append(output_filename)

        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * videos_per_shard
            end_ndx = min((shard_id + 1) * videos_per_shard, max_videos)

            for current_video in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting video %d/shard_end=%d/total=%d, shard %d' %
                                 (current_video + 1, videos_per_shard * (shard_id + 1), max_videos, shard_id))
                sys.stdout.flush()

                example = example_video_from_npy(
                    npy_data, current_video, with_text, text_data, sequence_length)
                tfrecord_writer.write(example.SerializeToString())
                count += 1

                # Data augmentation options
                if options is not None:
                    if options['augment_data'] == 1:
                        for i in options['dataset']['options']:
                            operation = options['dataset']['options'][i]['name']

                            if operation == "flip":
                                example = example_video_from_npy(
                                    npy_data, current_video, with_text, text_data, sequence_length, flip=True)
                                tfrecord_writer.write(example.SerializeToString())
                                count += 1
                            elif operation == "rotate":
                                angle = options['dataset']['options'][i]['angle']
                                # print("angle={}".format(angle))
                                example = example_video_from_npy(
                                    npy_data, current_video, with_text, text_data, sequence_length, rotate=True, rotate_angle=angle)
                                tfrecord_writer.write(example.SerializeToString())
                                count += 1
                            else:
                                raise NotImplementedError

    sys.stdout.write('\n')
    sys.stdout.flush()

    return tfrecord_files, nb_frames, height, width, count


def example_video_from_npy(npy_data, current_video, with_text, text_data, sequence_length,
                           flip=False, rotate=False, rotate_angle=0):
    video_feature = {}
    height, width = None, None
    for frame in range(0, sequence_length):
        img = npy_data[current_video][frame]
        height = img.shape[0]
        width = img.shape[1]

        if flip:
            img = np.fliplr(img)
        elif rotate:
            img = image.rotate_image(img, rotate_angle, grayscale=True)
            assert img.shape[0] == height and img.shape[1] == width, "Shape mismatch"

        img_raw = np.reshape(img, (height, width, 1)).tostring()
        video_feature['{}'.format(frame)] = record._bytes_feature(img_raw)

    video_feature['height'] = record._int64_feature(height)
    video_feature['width'] = record._int64_feature(width)
    if with_text:
        video_feature['text'] = record._bytes_feature(tf.compat.as_bytes(text_data[current_video]))

    example = tf.train.Example(features=tf.train.Features(feature=video_feature))

    return example


def example_video(video_data, video_id, grayscale, target_size,
                  flip=False, rotate=False, rotate_angle=0, with_text=False, flip_dict=None):
    video_feature = {}
    height, width = None, None
    for frame in range(0, video_data[video_id]['nb_frames']):
        current_frame = '{}'.format(frame)
        tmp_img = image.load_img(video_data[video_id][current_frame], grayscale=grayscale, target_size=target_size,
                                 flip=flip, rotate_img=rotate, rotate_angle=rotate_angle)
        if type(tmp_img) is not np.ndarray:
            img = np.array(tmp_img)
        else:
            img = tmp_img

        height = img.shape[0]
        width = img.shape[1]
        video_feature['{}'.format(frame)] = record._bytes_feature(img.tostring())

    video_feature['height'] = record._int64_feature(height)
    video_feature['width'] = record._int64_feature(width)

    if with_text:
        current_text = video_data[video_id]['text']
        if flip:
            assert flip_dict is not None, 'Augmentation through <flip> is going to be performed but <flip dictionary> is empty (None)'

            for substr in flip_dict:
                if substr in current_text:
                    current_text = current_text.replace(substr, flip_dict["{}".format(substr)])
                    break  # change was made, now bye (otherwise it will again change the text)

        video_feature['text'] = record._bytes_feature(tf.compat.as_bytes(current_text))

    # video_feature['nb_frames'] = _int64_feature(video_data[video_id]['nb_frames'])
    # video_feature['start_frame'] = _int64_feature(video_data[video_id]['start_frame'])
    # video_feature['last_frame'] = _int64_feature(video_data[video_id]['last_frame'])
    # video_feature['id'] = _bytes_feature(tf.compat.as_bytes(video_id))
    example = tf.train.Example(features=tf.train.Features(feature=video_feature))

    return example


def write(tfrecord_file, tag, video_data, list_videos=None,
          num_shards=1, overwrite=True, target_size=None, grayscale=False,
          with_text=False, options=None, flip_dict=None):

    if not overwrite:
        tfrecord_files = list(Path(tfrecord_file).parent.glob("*.tfrecord"))
        if len(tfrecord_files) > 0:
            print("TFRecord file <{}> already exists.\n {}\nNothing to be done. Returning.".format(
                tfrecord_file, tfrecord_files))
            return tfrecord_files, -1, -1, -1

    list_keys = (list(video_data) if list_videos is None else list_videos)
    videos_per_shard = int(math.ceil(len(list_keys) / float(num_shards)))
    tfrecord_files = []
    count = 0

    file_path = Path(tfrecord_file)
    filename_without_suffix = file_path.parent / file_path.stem

    for shard_id in range(num_shards):
        output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
            filename_without_suffix, tag, shard_id, num_shards - 1)
        tfrecord_files.append(output_filename)

        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * videos_per_shard
            end_ndx = min((shard_id + 1) * videos_per_shard, len(list_keys))

            for current_video in range(start_ndx, end_ndx):
                # flip_str = ', with flip={}/{}'.format((current_video+1)*2, (end_ndx-1)*2) if flip else ''
                sys.stdout.write('\r>> Converting video %d/%d, total_videos_in_shard=%d/total_videos=%d, shard %d/%d' %
                                 (current_video + 1, end_ndx - 1, videos_per_shard, len(list_keys), shard_id + 1, num_shards))
                sys.stdout.flush()

                video_id = Path(list_keys[current_video]).name

                # Regular video
                example = example_video(
                    video_data, video_id=video_id, grayscale=grayscale, target_size=target_size, with_text=with_text)
                tfrecord_writer.write(example.SerializeToString())
                count += 1

                # Data augmentation options
                for i in options['dataset']['options']:
                    operation = options['dataset']['options'][i]['name']

                    if operation == "flip":
                        example = example_video(video_data, video_id=video_id, grayscale=grayscale,
                                                target_size=target_size, flip=True, with_text=with_text, flip_dict=flip_dict)
                    elif operation == "rotate":
                        angle = options['dataset']['options'][i]['angle']
                        example = example_video(video_data, video_id=video_id, grayscale=grayscale,
                                                target_size=target_size, rotate=True, rotate_angle=angle, with_text=with_text)
                    else:
                        raise NotImplementedError

                    tfrecord_writer.write(example.SerializeToString())
                    count += 1

    sys.stdout.write('\n')
    sys.stdout.flush()

    return tfrecord_files, count


def read_and_decode(filename_queue, sequence_length, img_height, img_width, batch_size, channels,
                    capacity=10000, num_threads=8, min_after_dequeue=1000, seed=1):
    reader = tf.compat.v1.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    first_features = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64)
        # 'nb_frames': tf.FixedLenFeature([], tf.int64),
        # 'start_frame': tf.FixedLenFeature([], tf.int64),
        # 'last_frame': tf.FixedLenFeature([], tf.int64),
        # 'id': tf.FixedLenFeature([], tf.string)
    }
    features = tf.io.parse_single_example(serialized=serialized_example, features=first_features)

    frames = parse_frames(
        features=features,
        video_proto=serialized_example,
        sequence_length=sequence_length,
        img_height=img_height,
        img_width=img_width,
        channels=channels
    )

    video_batch = tf.compat.v1.train.shuffle_batch([frames],
                                                   batch_size=batch_size,
                                                   capacity=capacity,
                                                   num_threads=num_threads,
                                                   min_after_dequeue=min_after_dequeue,
                                                   allow_smaller_final_batch=False,
                                                   seed=seed)
    # well https://stackoverflow.com/questions/44289925/tensorflow-properly-natively-handling-data-batching-overlaps-mini-batches?noredirect=1&lq=1
    # Just letting this without smaller last batch

    return video_batch


def parse_frames(features, video_proto,
                 sequence_length, img_height, img_width, channels,
                 feature_range=(-1, 1), scale_frames=True):

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    frames = []
    for frame in range(0, sequence_length):
        frame_str = '{}'.format(frame)
        features = {frame_str: tf.io.FixedLenFeature([], tf.string)}
        features = tf.io.parse_single_example(serialized=video_proto, features=features)

        current_image = tf.io.decode_raw(features[frame_str], tf.uint8)
        image_shape = tf.stack([height, width, channels])
        img = tf.reshape(current_image, image_shape)
        img.set_shape([img_height, img_width, channels])

        if scale_frames:
            current_image = tf.cast(current_image, tf.float32)
            # Scaling frames to feature range
            current_image = scale_utils.scale(current_image, feature_range=feature_range)
        frames.append(current_image)

    frames = tf.concat(frames, 0)
    video_shape = tf.stack([sequence_length, height, width, channels])
    frames = tf.reshape(frames, video_shape)
    frames.set_shape([sequence_length, img_height, img_width, channels])

    return frames


def read_and_decode_with_text(filename_queue, sequence_length, img_height, img_width, batch_size, channels,
                              capacity=2000, num_threads=4, min_after_dequeue=1000, seed=1,
                              feature_range=(-1, 1), scale_frames=True):

    reader = tf.compat.v1.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    first_features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'text': tf.io.FixedLenFeature([], tf.string)
    }
    features = tf.io.parse_single_example(serialized=serialized_example, features=first_features)
    current_text = tf.cast(features['text'], tf.string)

    frames = parse_frames(
        features=features,
        video_proto=serialized_example,
        sequence_length=sequence_length,
        img_height=img_height,
        img_width=img_width,
        channels=channels,
        feature_range=feature_range,
        scale_frames=scale_frames
    )

    video_batch, text_batch = tf.compat.v1.train.shuffle_batch(
        [frames, current_text],
        batch_size=batch_size,
        capacity=capacity,
        num_threads=num_threads,
        min_after_dequeue=min_after_dequeue,
        allow_smaller_final_batch=False,
        seed=seed
    )
    return video_batch, text_batch


def parse_data(video_proto,
               sequence_length, img_height, img_width, channels,
               feature_range=(-1, 1), scale_frames=True, with_text=False):

    first_features = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64)
    }

    if with_text:
        first_features["text"] = tf.io.FixedLenFeature([], tf.string)

    features = tf.io.parse_single_example(serialized=video_proto, features=first_features)
    current_text = tf.cast(features['text'], tf.string) if with_text else None

    frames = parse_frames(
        features=features,
        video_proto=video_proto,
        sequence_length=sequence_length,
        img_height=img_height,
        img_width=img_width,
        channels=channels,
        feature_range=feature_range,
        scale_frames=scale_frames
    )

    if with_text:
        return frames, current_text
    else:
        return frames
