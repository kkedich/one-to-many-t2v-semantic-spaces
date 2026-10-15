import logging

from text2video.dataset import dataset
from text2video.tools.dataset.utils import check_files
from text2video.utils.config import Config
from text2video.utils.file_utils import get_lines, load_json


def build_dataset(
        sequence_length: int,
        img_height: int,
        img_width: int,
        channels: int,
        cfg_dataset: Config,
        split: str,
        using_text: bool,
        batch_size: int = None,
        scale_frames: bool = True,
        use_bucket: bool = False,
        bucket_key: str = None,
):
    if using_text:
        data_root_folder = cfg_dataset.records.root
    else:
        data_root_folder = cfg_dataset.records.root_wo_text

    if not use_bucket:
        if split == "train":
            which_split = cfg_dataset.records.train
        elif split == "val":
            which_split = cfg_dataset.records.val
        elif split == "test":
            which_split = cfg_dataset.records.test
        else:
            raise ValueError(f"The split value is not valid: '{split}'")
        tfrecord_files = get_lines(data_root_folder + which_split)
    else:
        print(f"Loading bucket info <{cfg_dataset.bucket_info}>...")
        bucket_info = load_json(cfg_dataset.bucket_info)

        if bucket_key == "all_buckets":
            tfrecord_files = get_lines(cfg_dataset.records.root +
                                       cfg_dataset.records.train)
        else:
            # Get the same bucket as we are going to overfit here
            tfrecord_files = [bucket_info[bucket_key]["file/tfrecord"]]
        print(f"Bucket info loaded for <{bucket_key}>")

    # Check if the files exist
    print("\nChecking TFRecord files...")
    check_files(tfrecord_files)

    # Consider the channels of the dataset, as some datasets
    # have specific channels
    if "input" in cfg_dataset:
        channels = cfg_dataset.input.channels

    current_dataset, nb_videos = dataset.input_fn(
        filenames=tfrecord_files,
        batch_size=batch_size,
        epochs=1,
        sequence_length=sequence_length,
        img_height=img_height,
        img_width=img_width,
        channels=channels,
        shuffle=False,
        cache=False,
        with_text=using_text,
        scale_frames=scale_frames
    )

    nb_training_samples = int(nb_videos / batch_size)
    logging.info(f"\nTotal of <{nb_videos}> '{split}' pairs "
                 f"with <{nb_training_samples}> iterations per epoch")

    dataset_variables = {
        split: {
            "dataset": current_dataset,
            "nb_videos": nb_videos,
            "nb_samples_per_epoch": nb_training_samples
        }
    }

    return dataset_variables, channels


def build_train_val_dataset(
        cfg_experiment: Config,
        cfg_dataset: Config,
        using_text: bool,
        epochs: int,
        batch_size: int = None,
        train_with_bucket: bool = False,
        all_buckets: bool = False,
        scale_frames: bool = True,
        multi_gpu_strategy=None,
        one_video_per_text=False,
        shuffle=False,
        nb_threads=8
):
    if batch_size is None:
        batch_size = cfg_experiment.train_cfg.batch_size

    # Loading data
    if train_with_bucket:
        bucket_info = cfg_dataset.bucket_info
        print(f"Loading bucket info <{bucket_info}>...")

        bucket_info = load_json(bucket_info)

        bucket_key = " the digit 7 is moving left and right ."
        if all_buckets:
            tfrecord_files = get_lines(cfg_dataset.records.root +
                                       cfg_dataset.records.train)
        else:
            # Get the same bucket as we are going to overfit here
            tfrecord_files = [bucket_info[bucket_key]["file/tfrecord"]]

        tfrecord_files_val = tfrecord_files
        print("Bucket info loaded")

    elif one_video_per_text:
        # Same set for training is used for validation
        tfrecord_files = [cfg_dataset.one_video_per_text]
        tfrecord_files_val = tfrecord_files
    else:
        # Train with regular files with train and val splits
        if using_text:
            data_root_folder = cfg_dataset.records.root
        else:
            data_root_folder = cfg_dataset.records.root_wo_text
        tfrecord_files = get_lines(data_root_folder + cfg_dataset.records.train)
        tfrecord_files_val = get_lines(data_root_folder + cfg_dataset.records.val)
    print(f"Train: \n{tfrecord_files}\nVal: \n{tfrecord_files_val}")

    # Check if the files exist
    print("\nChecking TFRecord files...")
    check_files(tfrecord_files)
    check_files(tfrecord_files_val)

    if "input" in cfg_dataset:
        channels = cfg_dataset.input.channels
    else:
        channels = cfg_experiment.model_video.input.channels

    sequence_length = cfg_experiment.model_video.input.sequence_length
    img_height = cfg_experiment.model_video.input.img_height
    img_width = cfg_experiment.model_video.input.img_width

    # Obs: to use all videos from val, use a different batch size (e.g., 16), because we are dropping the last batch
    # that does not have the minimum size.
    training_video_dataset, nb_videos_train = dataset.input_fn(
        filenames=tfrecord_files,
        batch_size=batch_size,
        epochs=epochs,
        sequence_length=sequence_length,
        img_height=img_height,
        img_width=img_width,
        channels=channels,
        with_text=using_text,
        scale_frames=scale_frames,
        shuffle=shuffle,
        num_parallel_calls=nb_threads,
        num_parallel_reads=nb_threads
    )

    val_video_dataset, nb_videos_val = dataset.input_fn(
        filenames=tfrecord_files_val,
        batch_size=batch_size,
        epochs=epochs,
        sequence_length=sequence_length,
        img_height=img_height,
        img_width=img_width,
        channels=channels,
        shuffle=False,
        cache=False,
        with_text=using_text,
        scale_frames=scale_frames,
        num_parallel_calls=nb_threads,
        num_parallel_reads=nb_threads
    )

    nb_training_samples = int(nb_videos_train / batch_size)
    nb_val_samples = int(nb_videos_val / batch_size)
    logging.info(f"\nTotal of <{nb_videos_train}> training pairs "
                 f"with <{nb_training_samples}> iterations per epoch (batch size: {batch_size})")
    logging.info(f"\nTotal of <{nb_videos_val}> validation pairs "
                 f"with <{nb_val_samples}> iterations per epoch (batch size: {batch_size})\n")

    if multi_gpu_strategy is not None:
        training_video_dataset = multi_gpu_strategy.experimental_distribute_dataset(
            training_video_dataset)
        val_video_dataset = multi_gpu_strategy.experimental_distribute_dataset(val_video_dataset)

        nb_replicas = multi_gpu_strategy.num_replicas_in_sync
        nb_training_samples = int(nb_videos_train / (batch_size / nb_replicas))
        nb_val_samples = int(nb_videos_val / (batch_size / nb_replicas))

        logging.info(
            f"\nTotal of <{nb_training_samples}> iterations per epoch ({nb_replicas} replicas)")
        logging.info(
            f"\nTotal of <{nb_val_samples}> iterations per epoch ({nb_replicas} replicas)\n")

    dataset_variables = {
        "train": {
            "dataset": training_video_dataset,
            "nb_videos": nb_videos_train,
            "nb_samples_per_epoch": nb_training_samples
        },
        "validation": {
            "dataset": val_video_dataset,
            "nb_videos": nb_videos_val,
            "nb_samples_per_epoch": nb_val_samples
        }
    }

    return dataset_variables, channels
