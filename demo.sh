#!/bin/bash
# Unified demo script: run all samples for a given mode.
#
# Usage:
#   bash demo.sh <mode> <wan22_ckpt> <car_checkpoint>
#
# Examples:
#   bash demo.sh camera   /path/to/Wan2.2-TI2V-5B /path/to/car_checkpoint
#   bash demo.sh action   /path/to/Wan2.2-TI2V-5B /path/to/car_checkpoint
#   bash demo.sh hardcut  /path/to/Wan2.2-TI2V-5B /path/to/car_checkpoint
#   bash demo.sh continue /path/to/Wan2.2-TI2V-5B /path/to/car_checkpoint

set -e

mode="${1:-camera}"
wan22_ckpt="${2:-/path/to/Wan2.2-TI2V-5B}"
car_checkpoint="${3:-/path/to/car_checkpoint}"

case "${mode}" in
    camera|action|hardcut|continue) ;;
    *)
        echo "Usage: bash demo.sh <mode> <wan22_ckpt> <car_checkpoint>"
        echo ""
        echo "Modes:"
        echo "  camera   - I2V with explicit camera pose trajectory"
        echo "  action   - I2V with action commands (w/s/a/d/left/right/up/down)"
        echo "  hardcut  - I2V with action commands + skip: for hard-cut transitions"
        echo "  continue - V2V continuation from context video + action commands"
        exit 1
        ;;
esac

# ==== Shared config ==================================================
mem_cfg_drop=""
height=480
width=832
frame_num=81
guide_scale=3.0
seed=0

# ==== Per-mode config ================================================
case "${mode}" in
    camera)
        poses=traj
        sampling_steps=50
        input_root="examples/i2v/images"
        ;;
    action)
        motion="right,right,right,left,left,left"
        sampling_steps=50
        input_root="examples/i2v/images"
        motion_tag=$(echo "${motion}" | sed 's/,/_/g; s/+/-/g; s/:/-/g')
        ;;
    hardcut)
        motion="w+up,down,skip:right,a,skip:left,w"
        sampling_steps=50
        input_root="examples/i2v/images"
        motion_tag=$(echo "${motion}" | sed 's/,/_/g; s/+/-/g; s/:/-/g')
        ;;
    continue)
        motion="d,left"
        sampling_steps=50
        input_root="examples/continue"
        motion_tag=$(echo "${motion}" | sed 's/,/_/g; s/+/-/g; s/:/-/g')
        ;;
esac

# ==== Run all samples ================================================
for d in "${input_root}"/sample_*/; do
    input="${d%/}"
    sample=$(basename "${input}")
    echo "===================================================="
    echo "[demo] mode=${mode}  sample=${sample}"
    echo "===================================================="

    case "${mode}" in
        camera)
            python inference.py \
                --checkpoint_dir "${wan22_ckpt}" \
                --car_checkpoint "${car_checkpoint}" \
                --input_path "${input}" \
                --target_poses "examples/i2v/camera/${poses}.json" \
                --output_dir "output/camera_demo/${sample}_${poses}" \
                --height ${height} --width ${width} --frame_num ${frame_num} \
                --sampling_steps ${sampling_steps} --guide_scale ${guide_scale} \
                --seed ${seed} \
                ${mem_cfg_drop}
            ;;
        action|hardcut)
            output_dir="output/${mode}_demo/${sample}/${motion_tag}"

            python inference.py \
                --checkpoint_dir "${wan22_ckpt}" \
                --car_checkpoint "${car_checkpoint}" \
                --input_path "${input}" \
                --motion_sequence "${motion}" \
                --translation_step 4.0 \
                --rotate_angle 30.0 \
                --pitch_angle 15.0 \
                --output_dir "${output_dir}" \
                --height ${height} --width ${width} --frame_num ${frame_num} \
                --sampling_steps ${sampling_steps} --guide_scale ${guide_scale} \
                --seed ${seed} \
                ${mem_cfg_drop}

            echo ""
            echo "=== Rendering action UI overlay ==="
            python examples/gen_indicator_video.py \
                --segments_dir "${output_dir}" \
                --motion_sequence "${motion}" \
                --output "${output_dir}/final_ui.mp4" \
                --fps 24
            ;;
        continue)
            output_dir="output/continue_demo/${sample}/${motion_tag}"

            python inference.py \
                --checkpoint_dir "${wan22_ckpt}" \
                --car_checkpoint "${car_checkpoint}" \
                --input_path "${input}" \
                --context_poses "${input}/context_poses.json" \
                --motion_sequence "${motion}" \
                --translation_step 4.0 \
                --rotate_angle 30.0 \
                --pitch_angle 15.0 \
                --output_dir "${output_dir}" \
                --height ${height} --width ${width} --frame_num ${frame_num} \
                --sampling_steps ${sampling_steps} --guide_scale ${guide_scale} \
                --seed ${seed} \
                ${mem_cfg_drop}
            ;;
    esac
done

echo ""
echo "===================================================="
echo "[demo] Done: mode=${mode}, all samples completed."
echo "===================================================="
