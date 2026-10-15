import numpy as np
import tensorflow as tf
import time

from text2video.metrics.iqa.common import psnr, ssim
from text2video.metrics.iqa.perceptual import DISTSMetric, LPIPSMetric


LPIPS_METRIC = "lpips"
DISTS_METRIC = "dists"
PSNR_METRIC = "psnr"
SSIM_METRIC = "ssim"


def get_iqa_metrics(
        device: str = "cuda",
        default_metrics=True,
        lpips_model=None,
        dists_model=None,
):
    shared_model = True if lpips_model is not None and dists_model is not None else False
    print(f"[IQA] Instance of IQA metrics will be loaded at <{device}>."
          f" Shared model? {shared_model}")

    iqa_metrics = {
        PSNR_METRIC: psnr,
        SSIM_METRIC: ssim,
    }

    if not default_metrics:
        if lpips_model is None:
            iqa_metrics[LPIPS_METRIC] = LPIPSMetric(device=device)
        else:
            iqa_metrics[LPIPS_METRIC] = lpips_model

        if dists_model is None:
            iqa_metrics[DISTS_METRIC] = DISTSMetric(device=device)
        else:
            iqa_metrics[DISTS_METRIC] = dists_model

    # Individual metrics
    iqa_metrics_over_unique_input = {}

    return iqa_metrics, iqa_metrics_over_unique_input


class IQAMetrics:
    def __init__(self, device: str = "cuda", default=True,
                 lpips_model=None, dists_model=None):
        self.common_iqa_metrics, self.individual_iqa_metrics = \
            get_iqa_metrics(
                device,
                default_metrics=default,
                lpips_model=lpips_model,
                dists_model=dists_model
            )

        # Setting arrays per video
        self.metrics_per_video = {}

    def initialize_metrics(self):
        for metric in self.common_iqa_metrics:
            self.metrics_per_video[metric] = []

        for current_data in ["true", "pred"]:
            for metric in self.individual_iqa_metrics:
                self.metrics_per_video[f"{metric}_{current_data}"] = []

    def _process_metric_result(self, current_result, metric):
        if isinstance(current_result, np.ndarray):
            self.metrics_per_video[metric] = np.append(
                self.metrics_per_video[metric], current_result)
        elif isinstance(current_result, tf.Tensor):
            self.metrics_per_video[metric] = tf.concat(
                [self.metrics_per_video[metric], current_result], axis=0)
        else:
            raise ValueError(
                f"[IQA] Type not considered: current_type={type(current_result)}, {current_result}")

    def compute(self, y_true: tf.Tensor, y_pred: tf.Tensor, strategy=None):

        # Calculating IQA metrics
        for metric in self.common_iqa_metrics:
            if strategy is not None:
                per_replica_results = strategy.run(
                    self.common_iqa_metrics[metric], args=(y_true, y_pred)
                )

                per_replica_per_video_results = per_replica_results
                if isinstance(per_replica_results, tuple):
                    # more than one variable per result
                    per_replica_per_video_results, _ = per_replica_results

                if strategy.num_replicas_in_sync > 1:
                    # List of results per replica
                    for current_result in per_replica_per_video_results.values:
                        self._process_metric_result(current_result, metric)
                else:
                    current_result = per_replica_per_video_results
                    self._process_metric_result(current_result, metric)
            else:
                current_result = self.common_iqa_metrics[metric](y_true, y_pred)

                if isinstance(current_result, tuple):
                    # more than one variable per result
                    current_result, _ = current_result

                self._process_metric_result(current_result, metric)

    def reset_states(self):
        # print("Resetting metrics states...")
        self.initialize_metrics()

    def result(self) -> dict:
        result_per_metric = {}
        for metric in self.metrics_per_video:
            if isinstance(self.metrics_per_video[metric], np.ndarray):
                iqa_current_value = np.mean(self.metrics_per_video[metric])
                std_value = np.std(self.metrics_per_video[metric])

            elif isinstance(self.metrics_per_video[metric], tf.Tensor):
                iqa_current_value = tf.reduce_mean(self.metrics_per_video[metric])
                std_value = tf.math.reduce_std(self.metrics_per_video[metric])

            else:
                raise ValueError(
                    f"[IQA] Type not considered: current_type={type(self.metrics_per_video[metric])},"
                    f" {self.metrics_per_video[metric]}")

            print(f"[IQA] metric={metric}, iqa_current_value={iqa_current_value:.3f}")
            if isinstance(iqa_current_value, tf.Tensor):
                iqa_current_value = iqa_current_value.numpy()

            result_per_metric[metric] = iqa_current_value
            result_per_metric[metric + "_std"] = std_value

        return result_per_metric


class IQAMetricsBucket:
    def __init__(self,
                 device: str = "cuda",
                 default: bool = True,
                 buckets_per_text: dict = None,
                 lpips_model=None,
                 dists_model=None):
        self.common_iqa_metrics, self.individual_iqa_metrics = \
            get_iqa_metrics(
                device,
                default_metrics=default,
                lpips_model=lpips_model,
                dists_model=dists_model
            )

        # Setting arrays per video
        self.metrics_per_video = {}

        self.buckets_per_text = buckets_per_text

    def initialize_metrics(self):
        for metric in self.common_iqa_metrics:
            self.metrics_per_video[metric] = []

        for current_data in ["true", "pred"]:
            for metric in self.individual_iqa_metrics:
                self.metrics_per_video[f"{metric}_{current_data}"] = []

    def _process_metric_result(self, current_result, metric):
        if isinstance(current_result, np.ndarray):
            self.metrics_per_video[metric] = np.append(
                self.metrics_per_video[metric], current_result)
        elif isinstance(current_result, tf.Tensor):
            self.metrics_per_video[metric] = tf.concat(
                [self.metrics_per_video[metric], current_result], axis=0)
        else:
            raise ValueError(
                f"Type not considered: current_type={type(current_result)}, {current_result}")

    def compute(self, input_texts: list, y_pred: tf.Tensor):

        start_time = time.time()
        if isinstance(y_pred, tf.Tensor):
            y_pred = y_pred.numpy()

        # Calculating IQA metrics
        for metric in self.common_iqa_metrics:
            metric_values_per_text = []

            for i, current_text in enumerate(input_texts):
                current_bucket = self.buckets_per_text[current_text]
                current_pred_video = np.expand_dims(y_pred[i, :], axis=0)
                pred_video_tiled = np.tile(
                    current_pred_video, (current_bucket.shape[0], 1, 1, 1, 1))

                distance_values = self.common_iqa_metrics[metric](
                    current_bucket, pred_video_tiled)
                if isinstance(distance_values, tuple):
                    # more than one variable per result
                    distance_values, _ = distance_values
                if isinstance(distance_values, tf.Tensor):
                    distance_values = distance_values.numpy()

                # Check the metric type to consider either the smaller or the higher valuer
                # PSNR, SSIM -> higher is better, DISTS, LPIPS -> smaller is better
                if metric in (LPIPS_METRIC, DISTS_METRIC):
                    # We get the smallest
                    best_value = np.min(distance_values, axis=-1)
                elif metric in (PSNR_METRIC, SSIM_METRIC):
                    best_value = np.max(distance_values, axis=-1)
                else:
                    raise ValueError(f"Invalid metric: {metric}")

                metric_values_per_text.append(best_value)
                print(f"[IQA] {i + 1}/{len(input_texts)}, current_metric: {metric}, "
                      f"distance_values={distance_values.shape}, "
                      f"metric_values_per_text={metric_values_per_text}")

            metric_values_per_text = np.array(metric_values_per_text)

            self._process_metric_result(metric_values_per_text, metric)
        print(f"[IQA] Finished calculation time: {time.time() - start_time:.2f}\n")

    def reset_states(self):
        self.initialize_metrics()

    def result(self) -> dict:
        result_per_metric = {}
        for metric in self.metrics_per_video:
            current_metric = self.metrics_per_video[metric]

            if isinstance(current_metric, np.ndarray):
                iqa_current_value = np.mean(current_metric)
                std_value = np.std(current_metric)

            elif isinstance(current_metric, tf.Tensor):
                iqa_current_value = tf.reduce_mean(current_metric)
                std_value = tf.math.reduce_std(current_metric)

            else:
                raise ValueError(f"[IQA] Type not considered: "
                                 f"current_type={type(current_metric)}, {current_metric}")

            print(f"[IQA] metric={metric}, iqa_current_value={iqa_current_value:.3f}")
            if isinstance(iqa_current_value, tf.Tensor):
                iqa_current_value = iqa_current_value.numpy()

            result_per_metric[metric] = iqa_current_value
            result_per_metric[metric + "_std"] = std_value

        return result_per_metric
