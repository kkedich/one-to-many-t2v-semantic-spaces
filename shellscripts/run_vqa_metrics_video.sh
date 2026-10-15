#!/bin/bash
# Example of usage:
# ./run_vqa_metrics_video.sh \
#     --config="../text2video/configs/video/video_unet_ldm.py" \
#     --dataset"=../text2video/configs/configs/datasets/kth.py" \
#     --ckpt_path="./text2video/output_files/path/experiment/checkpoints/generator" \
#     --output_path="./text2video/output_files/path/experiment/inference/" \
#     --gpu_id=0
#
# Dont forget to change the permissions of the script with:
#    chmod +x run_vqa_metrics_video.sh

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
            --experiment_path)  experiment_path=${value} ;;
            --dataset)      dataset=${value} ;;
            --output_path)  output_path=${value} ;;
            --gpu_id)       gpu_id=${value} ;;
            *)
    esac
done

CUDA_VISIBLE_DEVICES="$gpu_id"
export CUDA_VISIBLE_DEVICES
container_dir=/work

# Video evaluation
echo "Starting the video evaluation script."
container_default_name=kbogdan-tf215-exp-only

docker exec -it \
      -e CUDA_VISIBLE_DEVICES="$gpu_id" \
      --workdir /work/text2video "$container_default_name" \
      python "$container_dir"/text2video/text2video/tools/test_video.py \
             "$dataset" \
             --experiment_path="$experiment_path" \
             --output_path="$output_path" \
             --gpu_id="$gpu_id" \
             --new_load_approach
echo "Finished inference part."

# FVD - Kinects I3D
echo "Starting the FVD script"
container_name=kbogdan-kinects-i3d-exec
image_name=kbogdan/kinects_i3d_v2

printf "Mounting home from: %s\n" "${HOME}"
input_folder_original_videos=$output_path/"gt/"
input_folder_generated_videos=$output_path/"pred/"

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
report_path=$output_path/"report.md"
output_file_fvd=$output_path/"fvd_i3d.pickle"

docker exec -it --workdir /work/text2video "$container_default_name" \
	python "$container_dir"/text2video/text2video/report/update_report.py \
         --report_path="$report_path" \
         --result_to_add="$output_file_fvd"

echo "All done. Finished!"
