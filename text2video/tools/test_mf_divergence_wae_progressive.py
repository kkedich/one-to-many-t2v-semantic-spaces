"""
Test script for the progressive strategy.
"""
import argparse
import datetime
import logging
import numpy as np
import random
import tensorflow as tf
import time
import torch
from pathlib import Path
from tensorflow.python.ops.numpy_ops import np_config

from text2video.dataset.data_builder import build_dataset
from text2video.metrics.iqa.base import IQAMetrics, IQAMetricsBucket
from text2video.metrics.iqa.perceptual import DISTSMetric, LPIPSMetric
from text2video.models.mapping.builder_mf import create as create_mf
from text2video.report.report_utils import (dict_to_markdown_table,
                                            images_to_markdown_strings)
from text2video.trainer.checkpoint import load_checkpoint
from text2video.trainer.forward_mapping_function import \
    forward_pass_stacked_wae_progressive
from text2video.trainer.misc_utils import build_buckets, merge_list
from text2video.trainer.text import get_text_embeddings, instantiate_text_model
from text2video.trainer.training_options import build_mf_stackedwae_options
from text2video.trainer.video import instantiate_video_enc_dec_sep_model
from text2video.utils.config import Config
from text2video.utils.config_gpu import init_gpu
from text2video.utils.file_utils import save_list
from text2video.viewer.embeddings import plot_embeddings
from text2video.viewer.viewer import overview_experiment

np_config.enable_numpy_behavior()


def parse_args():
    """Argument parsing function"""
    parser = argparse.ArgumentParser(
        description="Test the Text-to-Video Mapping function with Divergence and "
                    "a sequential of WAEs with a progressive strategy")
    parser.add_argument(
        "--dir_video",
        type=str,
        default=None,
        help="Directory to the directory where the video model is saved. "
             "Path before 'video_generator_model'.")
    parser.add_argument(
        "--dataset_config", help="Configuration file for the dataset")
    parser.add_argument(
        "--experiment_path", type=str, default=None,
        help="MF experiment path from where the checkpoint will be restored.")
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Path to save the predictions.")
    parser.add_argument(
        "--offline_models", action="store_true", default=False,
        help="Indicates whether to use offline models (e.g., Slurm experiments).")
    parser.add_argument(
        "--load_model_old_way", action="store_true", default=False,
        help="Indicates whether we use the old way to load the models instead of the checkpoints."
             " Valid for older experiments.")

    args = parser.parse_args()

    return args


def inference(dataset_variables, models, testing_options,
              dir_experiment, save_output=True, videos_to_sample=3):

    print("\nStarting inference of the MF model...")
    start_time = time.time()
    test_video_dataset = dataset_variables["test"]["dataset"]
    nb_samples_per_epoch = dataset_variables["test"]["nb_samples_per_epoch"]

    # IQA init metrics
    testing_options.iqa_video.initialize_metrics()
    testing_options.iqa_video_ae_shared.initialize_metrics()
    testing_options.iqa_text2video.initialize_metrics()

    model_outputs = {
        "video_input_data": None,  # Real video
        "text_embeddings": None,  # Real text
        # Codes text to video
        "text2shared_code": None,
        "text_shared2video_code": None,
        "text2shared2v_decoded_video": None,
        # Code video
        "video_code_true": None,
        "decoded_video_true": None,
        # Codes video shared AE
        "video2shared_code": None,
        "video_shared2video_code": None,
        "shared2v_decoded_video": None,
    }

    # Create folders for each type of data that we will save
    output_text_folder = dir_experiment / "texts"
    if save_output:
        output_text_folder.mkdir(parents=True, exist_ok=True)

    output_folders = {}
    for current_key in model_outputs.keys():
        current_path = dir_experiment / current_key
        output_folders[current_key] = current_path
        if save_output:
            current_path.mkdir(parents=True, exist_ok=True)

    videos_indexes = random.sample(range(nb_samples_per_epoch), videos_to_sample)
    print(f"Video indexes: {videos_indexes}, nb_samples_per_epoch: {nb_samples_per_epoch}")
    video_samples_path, v2shared_samples_path, t2video_samples_path = [], [], []

    for batch, (batch_videos, batch_texts) in enumerate(test_video_dataset):
        # Text embedding
        batch_texts_decoded = [i.decode("utf-8") for i in batch_texts.numpy()]
        text_embeddings = get_text_embeddings(
            batch_texts_decoded, models, testing_options.text_model_type)

        outputs = forward_pass_stacked_wae_progressive(
            text_embeddings=text_embeddings,
            text_embeddings_with_noise=None,
            batch_videos=batch_videos,
            mf_text2shared_model=models["mf_text2shared_model"],
            mf_video2shared_model=models["mf_video2shared_model"],
            mf_shared2video_model=models["mf_shared2video_model"],
            video_encoder=models["video_encoder"],
            video_decoder=models["video_decoder"],
            options=testing_options,
            training=False
        )

        merge_list(text_embeddings, model_outputs, "text_embeddings")
        merge_list(batch_videos, model_outputs, "video_input_data")
        for current_key in outputs:
            # We save individual prediction, but we do not need to accumulate this info
            if "decoded_video" not in current_key:
                merge_list(outputs[current_key], model_outputs, current_key)

        if batch % 100 == 0:
            print(f"Current inference batch: {batch}/{nb_samples_per_epoch}")
            for current_key in model_outputs:
                if model_outputs[current_key] is not None:
                    print(f"Current <{current_key}>: {model_outputs[current_key].shape}")

        if batch in videos_indexes:
            print(f"Generating visual sample of batch <{batch}> for the video auto-encoder")
            output_sample = dir_experiment / f"sample_video_from_video_autoencoder_{batch}.png"
            current_samples_path = overview_experiment(
                outputs["decoded_video_true"],
                batch_videos,
                videos_to_show=1,
                batch_index=None,
                grayscale=True if testing_options.channels == 1 else False,
                filename_prefix=output_sample
            )
            print(f"Adding sample at <{current_samples_path}>")
            video_samples_path.extend(current_samples_path)

            print(
                f"Generating visual sample of batch <{batch}> for the video auto-encoder from the shared code")
            output_sample = dir_experiment / \
                f"sample_video_from_video_autoencoder_shared_{batch}.png"
            current_v2shared_samples_path = overview_experiment(
                outputs["shared2v_decoded_video"],
                batch_videos,
                videos_to_show=1,
                batch_index=None,
                grayscale=True if testing_options.channels == 1 else False,
                filename_prefix=output_sample
            )
            print(f"Adding sample at <{current_v2shared_samples_path}>")
            v2shared_samples_path.extend(current_v2shared_samples_path)

            print(f"Generating visual sample of text to video inference <{batch}>")
            output_sample = dir_experiment / f"sample_video_from_text_{batch}.png"
            current_t2video_samples_path = overview_experiment(
                outputs["text2shared2v_decoded_video"],
                outputs["decoded_video_true"],
                videos_to_show=1,
                batch_index=None,
                grayscale=True if testing_options.channels == 1 else False,
                filename_prefix=output_sample
            )
            print(f"Adding sample at <{current_t2video_samples_path}>")
            t2video_samples_path.extend(current_t2video_samples_path)

        if save_output:
            # Save codes and decoded videos, i.e., other features into .npy
            save_list(str(output_text_folder / f"{batch}.txt"), batch_texts_decoded)

            np.save(output_folders["video_input_data"] / f"{batch}_gt_vid.npy", batch_videos)
            np.save(output_folders["text_embeddings"] / f"{batch}_text_code.npy", text_embeddings)
            np.save(output_folders["text2shared_code"] /
                    f"{batch}_text2shared_code.npy", outputs["text2shared_code"])
            np.save(output_folders["text_shared2video_code"] /
                    f"{batch}_code.npy", outputs["text_shared2video_code"])
            np.save(output_folders["text2shared2v_decoded_video"] / f"{batch}_decoded_video.npy",
                    outputs["text2shared2v_decoded_video"])
            np.save(output_folders["video_code_true"] /
                    f"{batch}_video_code.npy", outputs["video_code_true"])
            np.save(output_folders["decoded_video_true"] /
                    f"{batch}_decoded_video.npy", outputs["decoded_video_true"])
            np.save(output_folders["video2shared_code"] /
                    f"{batch}_video2shared_code.npy", outputs["video2shared_code"])
            np.save(output_folders["video_shared2video_code"] /
                    f"{batch}_code.npy", outputs["video_shared2video_code"])
            np.save(output_folders["shared2v_decoded_video"] / f"{batch}_decoded_video.npy",
                    outputs["shared2v_decoded_video"])

        # Calculating IQA metrics
        # Video Auto-Encoder
        testing_options.iqa_video.compute(batch_videos, outputs["decoded_video_true"])

        # Video AE shared
        testing_options.iqa_video_ae_shared.compute(
            outputs["decoded_video_true"], outputs["shared2v_decoded_video"])

        # Text to video
        testing_options.iqa_text2video.compute(
            input_texts=batch_texts_decoded,
            y_pred=outputs["text2shared2v_decoded_video"]
        )

    result_per_metric_video = testing_options.iqa_video.result()
    result_per_metric_video_ae_shared = testing_options.iqa_video_ae_shared.result()
    result_per_metric_text2video = testing_options.iqa_text2video.result()

    print(f"Finished inference execution: {time.time() - start_time:.2f}\n")

    return (result_per_metric_video, result_per_metric_video_ae_shared, result_per_metric_text2video,
            model_outputs, video_samples_path, v2shared_samples_path, t2video_samples_path)


def create_models(cfg_mf_experiment,
                  cfg_text_experiment,
                  cfg_dataset,
                  cfg_video,
                  dir_experiment_video,
                  batch_size,
                  offline_models=False):
    # Testing options
    testing_options = build_mf_stackedwae_options(cfg_mf_experiment)

    # Text part
    testing_options.text_model_type = cfg_mf_experiment.model.get("text_model", "base")
    if cfg_mf_experiment.model.get("text2shared", False):
        testing_options.text_model_type = cfg_mf_experiment.model.text2shared.get(
            "text_model", "base")
    text_models = instantiate_text_model(
        cfg_text_experiment,
        model_type=testing_options.text_model_type,
        offline_models=offline_models
    )

    # Video part
    # I do not need the video discriminator part as we are not training it the video yet, we will
    # freeze this part and try to learn the mapping part
    testing_options.video_model_name = cfg_video.model_video.generator.get("name", "baseline")

    if cfg_video.model_video.input.channels != cfg_dataset.input.channels:
        # Follow channels from dataset, if available
        cfg_video.model_video.input.channels = cfg_dataset.input.channels

    video_encoder, video_decoder, video_code_shape = instantiate_video_enc_dec_sep_model(
        cfg_video,
        batch_size=batch_size,
        dir_experiment=dir_experiment_video,
        model_name=testing_options.video_model_name
    )

    # MF text2shared_space part
    mf_text2shared_model = create_mf(
        name=cfg_mf_experiment.model.text2shared.get("name", "non_gaussian_projection_layer"),
        options=cfg_mf_experiment.model.text2shared
    )
    # mf_text2shared_optimizer = build_opt(cfg_mf_experiment.model.optimizer)

    # MF video2shared_space part
    mf_video2shared_model = create_mf(
        name=cfg_mf_experiment.model.video2shared.get("name", "non_gaussian_projection_layer"),
        options=cfg_mf_experiment.model.video2shared
    )
    mf_shared2video_model = create_mf(
        name=cfg_mf_experiment.model.shared2video.get("name", "non_gaussian_projection_layer"),
        options=cfg_mf_experiment.model.shared2video
    )
    optimizers = {}

    models = {
        "text_encoder_model": text_models["encoder"],
        # Mapping function related
        "mf_text2shared_model": mf_text2shared_model,
        "mf_video2shared_model": mf_video2shared_model,
        "mf_shared2video_model": mf_shared2video_model,
        # Video
        "video_encoder": video_encoder,
        "video_decoder": video_decoder,
    }
    if testing_options.text_model_type == "clip":
        models["tokenizer"] = text_models["tokenizer"]

    models["word_dict"] = text_models["word_dict"]
    print("Finished defining models.")

    return testing_options, models, optimizers


def generate_plot_embeddings(model_outputs, dir_experiment_output):

    embeddings_plots = []

    print(f"Generates <{model_outputs['text_embeddings'].shape}> text_embeddings shape.")
    # Generate plot of text embeddings
    plot_path = f"{str(dir_experiment_output)}/plot_testing_text_embeddings.png"
    plot_embeddings(
        [model_outputs["text_embeddings"]],
        height=4, width=12, numpy_version=True,
        output_path=plot_path,
        scale=True,
        distributions=["text_embeddings"],
        colors=["r"],
        put_id_on_points=False,
        with_tsne=True
    )
    embeddings_plot_str = f"<img src='{Path(plot_path).name}' alt='Text embeddings plot'>"
    embeddings_plots.append(embeddings_plot_str)

    # Shared codes
    print(f"Generates <{model_outputs['text2shared_code'].shape}> shared codes.")
    # Generate plot of text embeddings
    plot_path = f"{str(dir_experiment_output)}/plot_testing_shared_embeddings.png"
    plot_embeddings(
        [
            model_outputs["text2shared_code"],
            model_outputs["video2shared_code"]
        ],
        height=4, width=12, numpy_version=True,
        output_path=plot_path,
        scale=True,
        distributions=["text_shared_code", "video_shared_code"],
        colors=["r", "b"],
        put_id_on_points=False,
        with_tsne=True
    )
    embeddings_plot_str = f"<img src='{Path(plot_path).name}' alt='Text embeddings plot'>"
    embeddings_plots.append(embeddings_plot_str)

    # Video codes
    print(f"Generates <{model_outputs['text_shared2video_code'].shape}> video codes.")
    # Generate plot of text embeddings
    plot_path = f"{str(dir_experiment_output)}/plot_testing_video_embeddings.png"
    plot_embeddings(
        [
            model_outputs["text_shared2video_code"],
            model_outputs["video_shared2video_code"],
            model_outputs["video_code_true"]
        ],
        height=4, width=12, numpy_version=True,
        output_path=plot_path,
        scale=True,
        distributions=["text_shared2video_code", "video_shared2video_code", "video_code_true"],
        colors=["r", "b", "#7fabd7"],
        put_id_on_points=False,
        with_tsne=True
    )
    embeddings_plot_str = f"<img src='{Path(plot_path).name}' alt='Text embeddings plot'>"
    embeddings_plots.append(embeddings_plot_str)

    return embeddings_plots


def main():
    start_time = time.time()
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # GPU setup
    init_gpu(0)

    # General parameters
    args = parse_args()
    dir_experiment_input = Path(args.experiment_path)
    # Configs
    cfg_mf_experiment = Config(args.experiment_path + "/cfg_mf_experiment.py")
    cfg_dataset = Config(args.dataset_config).cfg
    cfg_text_experiment = Config(args.experiment_path + "/cfg_text_experiment.py")

    # Video directory
    if args.dir_video is not None:
        path_dir_video = Path(args.dir_video)
        if path_dir_video.exists() and path_dir_video.is_dir():
            logging.info(
                f"Input <{str(path_dir_video)}> is going to be used as video directory checkpoint.")
        else:
            raise FileNotFoundError(f"Directory <{path_dir_video}> not found!")

        dir_experiment_video = str(path_dir_video)
        cfg_video = Config(str(path_dir_video / "cfg_experiment.py"))
    else:
        raise ValueError("No video directory found")
    print(f"Directory for the video model: {dir_experiment_video}")
    print(f"Resuming from <{args.experiment_path}>")

    if args.output_path is None:
        dir_experiment_output = dir_experiment_input / "inference"
    else:
        dir_experiment_output = Path(args.output_path)
    if not dir_experiment_output.exists():
        dir_experiment_output.mkdir(parents=True, exist_ok=True)
    filename_log = f"{str(dir_experiment_output)}/test_mf_{current_time}.log"

    logging.basicConfig(filename=filename_log, level=logging.DEBUG)
    logging.info("Starting the training of the mapping function with divergence")

    print("Creating models...")
    batch_size = cfg_mf_experiment.test_cfg.batch_size
    testing_options, models, optimizers = (
        create_models(
            cfg_mf_experiment=cfg_mf_experiment,
            cfg_text_experiment=cfg_text_experiment,
            cfg_dataset=cfg_dataset,
            cfg_video=cfg_video,
            dir_experiment_video=dir_experiment_video,
            batch_size=batch_size,
            offline_models=args.offline_models
        )
    )

    print("Restoring models...")
    if not args.load_model_old_way:
        ckpt_models = tf.train.Checkpoint(
            step=tf.Variable(1),
            mf_text2shared_model=models["mf_text2shared_model"],
            mf_video2shared_model=models["mf_video2shared_model"],
            mf_shared2video_model=models["mf_shared2video_model"],
        )
        checkpoints_dir = str(dir_experiment_input) + "/checkpoints/all/"
        if not Path(checkpoints_dir).exists():
            raise ValueError(f"Checkpoint path does not exist: <{checkpoints_dir}>")

        ckpt_manager = tf.train.CheckpointManager(
            ckpt_models,
            directory=checkpoints_dir,
            max_to_keep=3,
            checkpoint_name="ckpt_models_full_"
        )
        load_checkpoint(ckpt_manager, ckpt_models)
    else:
        checkpoints_dir = dir_experiment_input
        # Load the model in the old way, before using checkpoints here
        print(f"Restoring from {str(dir_experiment_input)}/mf_text2shared_model")
        models["mf_text2shared_model"] = tf.keras.models.load_model(
            f"{str(dir_experiment_input)}/mf_text2shared_model")
        print(f"Restoring from {str(dir_experiment_input)}/mf_video2shared_model")
        models["mf_video2shared_model"] = tf.keras.models.load_model(
            f"{str(dir_experiment_input)}/mf_video2shared_model")
        print(f"Restoring from {str(dir_experiment_input)}/mf_shared2video_model")
        models["mf_shared2video_model"] = tf.keras.models.load_model(
            f"{str(dir_experiment_input)}/mf_shared2video_model")

    # Read TEST dataset
    print("Creating test dataset...")
    dataset_variables, channels = build_dataset(
        sequence_length=cfg_mf_experiment.model.input.sequence_length,
        img_height=cfg_mf_experiment.model.input.img_height,
        img_width=cfg_mf_experiment.model.input.img_width,
        channels=cfg_mf_experiment.model.input.channels,
        cfg_dataset=cfg_dataset,
        split="test",
        using_text=True,
        batch_size=batch_size,
    )
    print("Creating test dataset: OK\n")

    # Build of buckets
    buckets_per_text = build_buckets(dataset_variables)

    # Calculate IQA metrics
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lpips_model = LPIPSMetric(device=device)
    dists_model = DISTSMetric(device=device)
    testing_options.iqa_video = IQAMetrics(
        device=device, default=False, lpips_model=lpips_model, dists_model=dists_model)
    testing_options.iqa_video_ae_shared = IQAMetrics(
        device=device, default=False, lpips_model=lpips_model, dists_model=dists_model)
    testing_options.iqa_text2video = IQAMetricsBucket(
        device=device, default=False, buckets_per_text=buckets_per_text,
        lpips_model=lpips_model, dists_model=dists_model)
    testing_options.channels = channels
    print(f"Testing options: {testing_options}\n")

    # Run inference on the test set
    inference_outputs = inference(
        dataset_variables,
        models,
        testing_options,
        dir_experiment_output,
        videos_to_sample=5
    )
    (result_per_metric_video, result_per_metric_video_ae_shared, result_per_metric_text2video,
     model_outputs, samples_path, v2shared_samples_path, t2video_samples_path) = inference_outputs

    print("Generating embeddings plot...")
    embeddings_plots = generate_plot_embeddings(model_outputs, dir_experiment_output)
    print(embeddings_plots)

    print("Generating report...")
    # Generate report on Markdown
    markdown_table_video = dict_to_markdown_table(result_per_metric_video)
    markdown_table_video_ae_shared = dict_to_markdown_table(result_per_metric_video_ae_shared)
    markdown_table_text2video = dict_to_markdown_table(result_per_metric_text2video)

    print("\nVideo samples:")
    markdown_string_samples = images_to_markdown_strings(samples_path)
    markdown_string_samples_vshared = images_to_markdown_strings(v2shared_samples_path)
    markdown_string_samples_text2video = images_to_markdown_strings(t2video_samples_path)
    print(markdown_string_samples)
    print(markdown_string_samples_vshared)
    print(markdown_string_samples_text2video)

    embeddings_plot_str = ""
    for current_plot_str in embeddings_plots:
        embeddings_plot_str += f"{current_plot_str}\n"

    markdown_string = f'''
# Inference Results

This is the inference results for the mapping function model.

- Checkpoint: `{checkpoints_dir}`
- Dataset: `{args.dataset_config}`
- Config: `{cfg_mf_experiment}`

## Metrics results

Video
{markdown_table_video}

Video Shared AE
{markdown_table_video_ae_shared}

Text to Video (Bucket version)
{markdown_table_text2video}

## Embeddings

{embeddings_plot_str}

## Samples from video autoencoder testing set
'''

    markdown_string += markdown_string_samples
    markdown_string += "\n## Samples from decoded video from video shared code testing set"
    markdown_string += markdown_string_samples_vshared
    markdown_string += "\n## Samples from text2video testing set"
    markdown_string += markdown_string_samples_text2video

    report_path = f"{str(dir_experiment_output)}/report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_string)
    print(f"Report saved at <{report_path}>")

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
