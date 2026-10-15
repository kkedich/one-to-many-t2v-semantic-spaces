#!/bin/bash
# Example of usage:
# ./run_vqa_metrics_video_mapping_function.sh \
#     --dir_video="/work/text2video/output_files/path_video_experiment/" \
#     --dataset="./configs/datasets/syncdraw_mm.py" \
#     --experiment_path="/work/text2video/output_files/path_alignment_experiment"
#     --gpu_id=0
#
# Dont forget to change the permissions of the script with:
#    chmod +x run_vqa_metrics_video_mapping_function.sh

# https://bertvv.github.io/cheat-sheets/Bash.html
set -o errexit  # abort on nonzero exitstatus
set -o pipefail # don't hide errors within pipes
set -o nounset  # abort on unbound variable
# set -o xtrace # flag to debug the bash script

dir=$(pwd)
printf "Current dir: %s\n" "$dir"

# Get the input variables
# https://unix.stackexchange.com/a/353639
for argument in "$@"
do
    key=$(echo "$argument" | cut -f1 -d=)
    value=$(echo "$argument" | cut -f2 -d=)

    case "$key" in
#            --dir_video)  dir_video=${value} ;;
#            --dataset)    dataset=${value} ;;
            --experiment_path)  experiment_path=${value} ;;
            --gpu_id)     gpu_id=${value} ;;
            *)
    esac
done

CUDA_VISIBLE_DEVICES="$gpu_id"
export CUDA_VISIBLE_DEVICES
container_dir=/work

# Video evaluation
echo "Starting the video evaluation script..."
container_default_name=kbogdan-tf215-exp-only

#docker exec -it \
#      -e CUDA_VISIBLE_DEVICES="$gpu_id" \
#      --workdir /work/text2video "$container_default_name" \
#      python "$container_dir"/text2video/text2video/tools/test_mf_divergence_wae_progressive.py \
#             --dir_video="$dir_video" \
#             --dataset_config="$dataset" \
#             --experiment_path="$experiment_path"
#echo "Finished inference part."

# FVD - Kinects I3D --------------------------------------------------------------------
echo "Starting the FVD script (shellscript)"
container_name=kbogdan-kinects-i3d-exec
image_name=kbogdan/kinects_i3d_v2

# this one will be the decoded video by the video autoencoder and the true video
printf "Mounting home from: %s\n" "${HOME}"
echo "Computing for the video gt and decoded video from the video autoencoder"
output_path=$experiment_path/"inference/"
input_folder_original_videos=$experiment_path/"inference/video_input_data/"
input_folder_generated_videos=$experiment_path/"inference/decoded_video_true/"

docker run --rm -it --gpus all --userns=host \
	-v "${HOME}":/work \
	--workdir=/work \
	--name "$container_name" "$image_name" \
	--real_videos="$input_folder_original_videos" \
	--fake_videos="$input_folder_generated_videos" \
	--output_path="$output_path" \
	--gpu_id="$gpu_id"

# Runs the update of the report with the FVD additional metric generated after inference samples
echo "Starting the update of the report..."
report_path=$experiment_path/"inference/report.md"
output_file_fvd=$output_path/"fvd_i3d.json"

docker exec -it --workdir /work/text2video "$container_default_name" \
	python "$container_dir"/text2video/text2video/report/update_report.py \
         --report_path="$report_path" \
         --result_to_add="$output_file_fvd" \
         --title="Video Autoencoder"


# this one will be the decoded video by the video shared autoencoder and the AE decoded video
echo "Computing for the decoded video from previous part and the decoded video from video shared AE"
output_path=$experiment_path/"inference_video_shared_ae/"
input_folder_original_videos=$experiment_path/"inference/decoded_video_true/"
input_folder_generated_videos=$experiment_path/"inference/shared2v_decoded_video/"

docker run --rm -it --gpus all --userns=host \
	-v "${HOME}":/work \
	--workdir=/work \
	--name "$container_name" "$image_name" \
	--real_videos="$input_folder_original_videos" \
	--fake_videos="$input_folder_generated_videos" \
	--output_path="$output_path" \
	--gpu_id="$gpu_id"

# Runs the update of the report with the FVD additional metric generated after inference samples
echo "Starting the update of the report..."
report_path=$experiment_path/"inference/report.md"
output_file_fvd=$output_path/"fvd_i3d.json"

docker exec -it --workdir /work/text2video "$container_default_name" \
	python "$container_dir"/text2video/text2video/report/update_report.py \
         --report_path="$report_path" \
         --result_to_add="$output_file_fvd" \
         --title="Video Shared Autoencoder"


# this one will be the mapping function decoded video by the MF and the true decoded video
echo "Computing for the MF decoded video and the decoded video from video shared AE"
output_path=$experiment_path/"inference_video_mf/"
input_folder_original_videos=$experiment_path/"inference/video_input_data/"
input_folder_generated_videos=$experiment_path/"inference/text2shared2v_decoded_video/"

docker run --rm -it --gpus all --userns=host \
	-v "${HOME}":/work \
	--workdir=/work \
	--name "$container_name" "$image_name" \
	--real_videos="$input_folder_original_videos" \
	--fake_videos="$input_folder_generated_videos" \
	--output_path="$output_path" \
	--gpu_id="$gpu_id"


# Runs the update of the report with the FVD additional metric generated after inference samples
echo "Starting the update of the report..."
report_path=$experiment_path/"inference/report.md"
output_file_fvd=$output_path/"fvd_i3d.json"

docker exec -it --workdir /work/text2video "$container_default_name" \
	python "$container_dir"/text2video/text2video/report/update_report.py \
         --report_path="$report_path" \
         --result_to_add="$output_file_fvd" \
         --title="Mapping Function text2video"

# ----------------------------------------------------------------------------------

echo "All done. Finished!"
