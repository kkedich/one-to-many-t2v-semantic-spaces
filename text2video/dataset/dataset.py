import tensorflow as tf

from text2video.dataset import text_record, video_record
from text2video.models.text.tokens import PAD_TOKEN


def input_fn(filenames, batch_size, epochs,
             sequence_length, img_height, img_width, channels,
             shuffle=True, drop_remainder=True, with_text=False,
             feature_range=(-1, 1), scale_frames=True, cache=False,
             num_parallel_reads=6,
             num_parallel_calls=6):

    # We can test using several filenames and
    # use tf.contrib.data.parallel_interleave to optimize data reading
    number_records = number_tfrecords(filenames)
    print("Reading data from {} ({} tfrecords)".format(filenames, number_records))

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=num_parallel_reads)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=256, seed=1)
    dataset = dataset.repeat(epochs)

    # Note
    # - only videos: without drop_remainder we are having problems at line 75 of
    #  image_quality when calling the dynamic partitions function
    # - with_text: the LSTM requires drop_remainder=True
    dataset = dataset.map(
        map_func=lambda x: video_record.parse_data(x,
                                                   sequence_length,
                                                   img_height,
                                                   img_width,
                                                   channels,
                                                   feature_range=feature_range,
                                                   scale_frames=scale_frames,
                                                   with_text=with_text),
        num_parallel_calls=num_parallel_calls
    )
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)

    if cache:
        dataset = dataset.cache()

    dataset = dataset.prefetch(buffer_size=4)

    return dataset, number_records


def input_fn_only_text(filenames, batch_size, word_dict, epochs=1,
                       shuffle=True, drop_remainder=True):
    """
    Code adapted from https://github.com/LightTag/BibSample/blob/master/prepare_dataset.py
    """
    number_records = number_tfrecords(filenames)
    print(f"Reading data from {filenames} ({number_records} tfrecords)")

    # Reads the tfrecords files
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=8)
    # print(f"Cardinality: {tf.data.experimental.cardinality(dataset).numpy()}")

    if shuffle:
        dataset = dataset.shuffle(buffer_size=256, seed=1)

    dataset = dataset.repeat(epochs)
    dataset = dataset.map(map_func=text_record.parse_text,
                          num_parallel_calls=8)

    # Generate the padded batches.
    # https://stackoverflow.com/questions/49840100/tf-data-dataset-padded-batch-pad-differently-each-feature
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes={
            # sequence_length is a scalar it doesn't need any padding, its always length one
            "sequence_length": [],
            # but the sequence have a variable length, thus we pass that
            # information to tensorflow which pads to longest
            "tokens": tf.TensorShape([None])
        },
        padding_values={
            "sequence_length": tf.constant(0, dtype=tf.int64),
            "tokens": tf.constant(word_dict[PAD_TOKEN], dtype=tf.int64)
        },
        drop_remainder=drop_remainder
    )

    dataset = dataset.prefetch(buffer_size=32)

    return dataset, number_records


def number_tfrecords(tf_records_filenames):
    number = 0
    for file in tf_records_filenames:
        # tf_record_iterator will be deprecated
        for _ in tf.compat.v1.python_io.tf_record_iterator(file):
            # print(tf.train.Example.FromString(record))
            number += 1
    return number


def init_dataset(dataset, one_shot_iterator=True):
    """Initiates the dataset by creating an iterator and the elements for it"""
    if one_shot_iterator:
        iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    else:
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)

    # Maybe change this to a slice output function
    next_video, next_text = iterator.get_next()

    return iterator, next_video, next_text
