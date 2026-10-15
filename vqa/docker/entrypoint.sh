#!/bin/bash
# Example of usage:
# ./entrypoint.sh \
#     --output_path="./path/to/output/inference/folder" \
#     --gpu_id=0
#
# Dont forget to change the permissions of the script with:
#    chmod +x entrypoint.sh

# https://bertvv.github.io/cheat-sheets/Bash.html
set -o errexit  # abort on nonzero exitstatus
set -o pipefail # don't hide errors within pipes
set -o nounset  # abort on unbound variable
# set -o xtrace # flag to debug the bash script

dir=$(pwd)
printf "Current dir: %s\n" "$dir"

# Install the editable packages of vidutils and text2video (for frechet distance)
echo "Installing dependencies..."
pip install -e /work/text2video/vidutils/
pip install -e /work/text2video/

# Get the input variables
# https://unix.stackexchange.com/a/353639
for argument in "$@"
do
    key=$(echo "$argument" | cut -f1 -d=)
    value=$(echo "$argument" | cut -f2 -d=)

    case "$key" in
            --real_videos)  real_videos=${value} ;;
            --fake_videos)  fake_videos=${value} ;;
            --output_path)  output_path=${value} ;;
            --gpu_id)       gpu_id=${value} ;;
            *)
    esac
done

# FVD
# real videos path follow an output_path/gt/ folder path
# fake videos path follow an output_path/pred/ folder path
echo "Starting the FVD script (entrypoint)."
output_file_fvd=$output_path/"fvd_i3d.json"  # Before was .pickle

CUDA_VISIBLE_DEVICES="$gpu_id"
export CUDA_VISIBLE_DEVICES

# The parameters related to the files can be folders as well
python "$dir"/text2video/vqa/src/kinects_i3d/evaluate_sample.py \
       --file_original_videos="$real_videos" \
       --file_generated_videos="$fake_videos" \
       --output_file="$output_file_fvd"
