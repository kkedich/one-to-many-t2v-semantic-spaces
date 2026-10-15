"""
This script has the goal of evaluating the generated videos in inference using our
current VQA metrics.

Steps:
- Perform forward pass of the model
- Get metrics on generated videos

"""

import argparse
import datetime
import logging
import numpy as np
import os
import pandas as pd
import random
import tensorflow as tf
from pathlib import Path

from text2video.dataset.data_builder import build_dataset
from text2video.metrics.iqa.base import IQAMetrics
from text2video.models.builder import create_generator_model
from text2video.trainer.checkpoint import restore_model
from text2video.trainer.forward_video import forward_pass_model
from text2video.utils.config import Config, ConfigDict
from text2video.utils.config_gpu import init_gpu
from text2video.viewer.embeddings import plot_embeddings
from text2video.viewer.viewer import overview_experiment


def parse_args():
    """Argument parsing function"""
    parser = argparse.ArgumentParser(description="Test the Video 3D Conv model")
    parser.add_argument(
        "dataset", help="Configuration file for the dataset")
    parser.add_argument(
        "--config", help="Model and train configuration file path")
    parser.add_argument(
        "--experiment_path", type=str, default=None,
        help="Model and train configuration folder where the configuration is available."
             "If this is set, you can leave the config and ckpt_path parts")
    parser.add_argument(
        "--ckpt_path", type=str, default=None,
        help="Checkpoint path to be considered.")
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Path to save the predictions.")
    parser.add_argument(
        "--gpu_id", type=str, default=None,
        help="GPU ID to execute the experiment")
    parser.add_argument(
        "--new_load_approach", action="store_true",
        help="Approach to load the pretrained model.")
    args = parser.parse_args()

    return args


def inference(dataset_variables, video_generator_model, testing_options,
              dir_experiment, save_output=True, videos_to_show=3):

    print("\nStarting inference...")
    test_video_dataset = dataset_variables["test"]["dataset"]
    nb_samples_per_epoch = dataset_variables["test"]["nb_samples_per_epoch"]

    dir_output_gt = dir_experiment / "gt"
    dir_output_pred = dir_experiment / "pred"
    dir_output_codes = dir_experiment / "codes"

    if save_output:
        dir_output_gt.mkdir(parents=True, exist_ok=True)
        dir_output_pred.mkdir(parents=True, exist_ok=True)
        dir_output_codes.mkdir(parents=True, exist_ok=True)

    testing_options.iqa.initialize_metrics()
    all_video_codes = None

    videos_indexes = random.sample(range(nb_samples_per_epoch), videos_to_show)
    print(f"Video indexes: {videos_indexes}, nb_samples_per_epoch: {nb_samples_per_epoch}")
    samples_path = []

    for batch, input_data in enumerate(test_video_dataset):
        _, _, video_code, decoded_video = forward_pass_model(
            input_data,
            video_generator_model,
            training=False,
            options=testing_options,
        )

        if video_code is not None:
            video_code_reshaped = video_code
            if len(video_code.shape) > 2:
                logging.debug(
                    f"Video code will be reshaped to dimension 2. Current shape: {video_code.shape}")
                output_dimension = tf.math.reduce_prod(tf.shape(video_code)[1:])
                video_code_reshaped = tf.reshape(video_code, [-1, output_dimension])

            if all_video_codes is None:
                all_video_codes = video_code_reshaped
            else:
                all_video_codes = np.vstack((all_video_codes, video_code_reshaped))

        if batch % 100 == 0:
            print(f"Current inference batch: {batch}/{nb_samples_per_epoch}")
            if all_video_codes is not None:
                print(f"Current all_video_codes: {all_video_codes.shape}")

        if batch in videos_indexes:
            print(f"Generating visual sample of batch <{batch}>")
            output_sample = dir_experiment / f"sample_video_{batch}.png"
            current_samples_path = overview_experiment(
                decoded_video,
                input_data,
                videos_to_show=1,
                batch_index=None,
                grayscale=True if testing_options.channels == 1 else False,
                filename_prefix=output_sample
            )
            print(f"Adding sample at <{current_samples_path}>")
            samples_path.extend(current_samples_path)

        if save_output:
            # Save decoded video and other features into .npy
            np.save(dir_output_gt / f"{batch}_gt.npy", input_data)
            np.save(dir_output_pred / f"{batch}_pred.npy", decoded_video)
            if video_code is not None:
                np.save(dir_output_codes / f"{batch}_code.npy", video_code)

        # Calculating IQA metrics
        testing_options.iqa.compute(input_data, decoded_video)

    result_per_metric = testing_options.iqa.result()

    return result_per_metric, all_video_codes, samples_path


def main():
    """Main function of the evaluation/inference script"""
    args = parse_args()

    if args.experiment_path is not None:
        print(f"Loading experiment from <{args.experiment_path}>")
        cfg_experiment = Config(args.experiment_path + "/cfg_experiment.py")
    else:
        # Load config file
        cfg_experiment = Config(args.config)

    cfg_dataset = Config(args.dataset)

    # GPU is set on CUDA_VISIBLE_DEVICES, so id is always 0
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        os.environ["WORLD_SIZE"] = "1"

    init_gpu([0])

    dir_experiment_output = Path(args.output_path)
    if not dir_experiment_output.exists():
        dir_experiment_output.mkdir(parents=True, exist_ok=True)

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(
        filename=f"{dir_experiment_output}/test_video_{current_time}.log", level=logging.DEBUG)
    logging.info("\nStarting the testing of the video model")
    logging.info(f"Output directory: <{dir_experiment_output}>")

    # Read dataset
    print("Creating test dataset...")
    batch_size = cfg_experiment.test_cfg.batch_size
    dataset_variables, channels = build_dataset(
        sequence_length=cfg_experiment.model_video.input.sequence_length,
        img_height=cfg_experiment.model_video.input.img_height,
        img_width=cfg_experiment.model_video.input.img_width,
        channels=cfg_experiment.model_video.input.channels,
        cfg_dataset=cfg_dataset.cfg,
        split="test",
        using_text=False,
        batch_size=batch_size,
    )
    print("Creating test dataset: OK\n")

    # Load pre-trained model
    # Loading the generator model
    print("Creating model...")
    if cfg_experiment.model_video.input.channels != channels:
        # Follow channels from dataset, if available
        cfg_experiment.model_video.input.channels = channels

    video_decoder, video_generator = create_generator_model(cfg_experiment, batch_size)

    video_generator.compiled_metrics = None
    if video_decoder is not None:
        video_decoder.compiled_metrics = None
    print("Creating model: OK\n")

    print(f"args.new_load_approach={args.new_load_approach}")
    if args.experiment_path is None:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = Path(args.experiment_path) / "checkpoints/generator/"

    if not Path(ckpt_path).exists():
        raise ValueError(f"Checkpoint path does not exist: <{ckpt_path}>")

    video_generator = restore_model(
        ckpt_path,
        video_generator,
        new_load_approach=args.new_load_approach
    )

    # Calculate IQA metrics
    testing_options = ConfigDict()
    testing_options.video_model_name = cfg_experiment.model_video.generator.get(
        "name", "baseline")
    testing_options.iqa = IQAMetrics(device="cuda", default=False)
    testing_options.channels = channels

    # Run inference on the test set
    result_per_metric, all_video_codes, samples_path = \
        inference(
            dataset_variables,
            video_generator,
            testing_options,
            dir_experiment_output
        )

    print(f"Results per metric: {result_per_metric}")
    embeddings_plot_str = "Model without embeddings =("
    if all_video_codes is not None:
        print(f"Generates <{all_video_codes.shape}> video code shape.")

        # Generate plot of video code embeddings
        plot_path = f"{str(dir_experiment_output)}/plot_testing.png"
        plot_embeddings(
            [all_video_codes],
            height=5, width=10, numpy_version=True,
            output_path=plot_path,
            scale=True,
            distributions=["video_code"],
            colors=["b"],
            put_id_on_points=False
        )

        embeddings_plot_str = f"<img src='{Path(plot_path).name}' alt='Embeddings plot'>"

    # Generate report on Markdown
    df = pd.DataFrame(result_per_metric.items(), columns=['Metric', 'Value'])
    df['Value'] = df['Value'].map('{:,.3f}'.format)
    markdown_table = df.to_markdown(index=False)
    print(markdown_table)

    markdown_string_samples = ""
    for image_path in samples_path:
        markdown_string_samples += f"<img src={Path(image_path).name}>\n"

    print("\nVideo samples:")
    print(markdown_string_samples)

    markdown_string = f'''
# Inference Results

This is the inference results for the video generator model.

- Checkpoint: `{args.ckpt_path}`
- Dataset: `{args.dataset}`
- Config: `{args.config}`

## Metrics results

{markdown_table}

## Embeddings

{embeddings_plot_str}

## Samples from testing set
'''

    markdown_string += markdown_string_samples
    with open(f"{str(dir_experiment_output)}/report.md", "w", encoding="utf-8") as f:
        f.write(markdown_string)


if __name__ == '__main__':
    main()
