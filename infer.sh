#!/bin/bash
# Single-input inference: generate a video from ONE image + prompt.
#   - By default the video follows the motion sequence (use "skip:" for hard cuts).
#   - Set traj to a camera pose file (see examples/i2v/camera/traj.json) to
#     follow a camera trajectory instead; motion is then ignored.
#
# Usage:
#   bash infer.sh <wan22_ckpt> <car_checkpoint>
# Edit image / prompt / motion / traj below to change the inputs.

set -e

wan22_ckpt="${1:-/path/to/Wan2.2-TI2V-5B}"
car_checkpoint="${2:-/path/to/car_checkpoint}"

image="examples/i2v/images/sample_001"          # image file or image folder
prompt=""                                        # empty -> use prompt.txt next to input
motion="w+up,down,skip:right,a,skip:left,w"      # action commands, "skip:" = hard cut
traj=""                                           # camera trajectory json (optional)
output_dir="output/infer"

if [ -n "${traj}" ]; then
    mode_args=(--mode camera --target_poses "${traj}")
elif echo "${motion}" | grep -q "skip:"; then
    mode_args=(--mode hardcut --motion_sequence "${motion}")
else
    mode_args=(--mode action --motion_sequence "${motion}")
fi

python inference.py \
    --checkpoint_dir "${wan22_ckpt}" \
    --car_checkpoint "${car_checkpoint}" \
    --input_path "${image}" \
    --output_dir "${output_dir}" \
    ${prompt:+--prompt "${prompt}"} \
    "${mode_args[@]}" \
    --height 480 --width 832 --frame_num 81 \
    --sampling_steps 50 --guide_scale 3.0 --seed 0

echo "[infer] Done. Output saved under ${output_dir}"
