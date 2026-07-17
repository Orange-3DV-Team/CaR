#!/bin/bash
# Single-input inference.
# Edit input / output / generation settings below, then run:
#   bash infer.sh <wan22_ckpt> <car_checkpoint>

set -e

wan22_ckpt="${1:-/path/to/Wan2.2-TI2V-5B}"
car_checkpoint="${2:-/path/to/car_checkpoint}"

input="examples/i2v/images/sample_001"
output_dir="output/infer"

prompt=""
motion="w+up,down,skip:right,a,skip:left,w"
traj=""
context_poses=""

height=480
width=832
frame_num=81
sampling_steps=50
guide_scale=3.0
seed=0
translation_step=4.0
rotate_angle=30.0
pitch_angle=15.0

python inference.py \
    --checkpoint_dir "${wan22_ckpt}" \
    --car_checkpoint "${car_checkpoint}" \
    --input_path "${input}" \
    --output_dir "${output_dir}" \
    --prompt "${prompt}" \
    --motion_sequence "${motion}" \
    --target_poses "${traj}" \
    --context_poses "${context_poses}" \
    --translation_step ${translation_step} \
    --rotate_angle ${rotate_angle} \
    --pitch_angle ${pitch_angle} \
    --height ${height} --width ${width} --frame_num ${frame_num} \
    --sampling_steps ${sampling_steps} --guide_scale ${guide_scale} --seed ${seed}

echo "[infer] Done. Output saved under ${output_dir}"
