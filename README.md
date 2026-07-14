<div align="center">

# CaR

## Compression and Retrieval: Implicit Memory Retrieval for Video World Models

<a href="https://pzzz-cv.github.io/">Zhan Peng</a><sup>1,2</sup>,
Jie Ma<sup>2</sup>,
Huiqiang Sun<sup>1</sup>,
Chong Gao<sup>2,3</sup>,
Zhijie Xue<sup>1</sup>,
Zhiyu Pan<sup>1</sup>,
Zhiguo Cao<sup>1*</sup>,
Jun Liang<sup>2*</sup>,
Jing Li<sup>2</sup>

<sup>1</sup>Huazhong University of Science and Technology &nbsp;
<sup>2</sup>HUJING Digital Media & Entertainment Group &nbsp;
<sup>3</sup>Sun Yat-sen University

<sup>*</sup>Corresponding author

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2606.23105)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://orange-3dv-team.github.io/CaR/)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/Orange-3DV-Team/CaR)
![Dataset](https://img.shields.io/badge/SceneFly_Dataset-Coming_Soon-lightgrey)
[![Model](https://img.shields.io/badge/Base_Model-Wan2.2--TI2V--5B-orange)](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)

</div>

---

<table>
<tr>
<td align="center"><b>I2V — Camera</b></td>
<td align="center"><b>V2V — History Extension</b></td>
</tr>
<tr>
<td align="center"><video src="assets/demo/scene_exploration_camera.mp4" controls muted loop width="420"></video></td>
<td align="center"><video src="assets/demo/history_extension.mp4" controls muted loop width="420"></video></td>
</tr>
<tr>
<td align="center"><b>I2V — Action</b></td>
<td align="center"><b>Hard-cut</b></td>
</tr>
<tr>
<td align="center"><video src="assets/open/i2v/sample_010/final_video_indicator.mp4" controls muted loop width="420"></video></td>
<td align="center"><video src="assets/demo/camera_motion_demo_opendomain_32.mp4" controls muted loop width="420"></video></td>
</tr>
</table>

<p align="center">
Additional examples are available on the <a href="https://orange-3dv-team.github.io/CaR/">project site</a>.
</p>

---

## Abstract

Video world models hold promise for simulating interactive environments, yet maintaining consistent long-term memory across complex camera trajectories remains a critical challenge. Existing methods typically rely on computationally expensive context scaling or rigid heuristic retrieval mechanisms, which lacks generalization to varying camera trajectories and environments. In this paper, we propose **CaR**, an attention-driven implicit memory retrieval mechanism to overcome these limitations. By injecting viewpoint information via positional encoding, our method performs flexible memory retrieval through attention computation. To efficiently process extended contexts with minimal computational overhead, we further introduce a lightweight context compression network. Furthermore, we construct **SceneFly**, a large-scale synthetic dataset featuring realistic camera trajectories and frame-level annotations to train and evaluate long-horizon video world models. Extensive experiments demonstrate that our approach achieves state-of-the-art results on established benchmarks and exhibits strong generalization to open-domain scenes.

This repository provides the open-source inference code for **CaR on Wan2.2-TI2V-5B**, including demo scripts, example inputs, camera trajectories, and single-input inference utilities.

---

## TL;DR

**CaR** eliminates handcrafted retrieval rules by letting the model retrieve directly within the attention mechanism. A dedicated **Retrieval Attention** branch encodes each token with Relative Pose Encoding, so attention scores naturally reflect viewpoint similarity. Combined with efficient **context compression**, CaR enables consistent long-horizon generation and uniquely supports **Camera Hard Cut** transitions with fully discontinuous trajectories.

---

## Motivation

<div align="center">

![Motivation](assets/fig1.png)

</div>

(1) Scaling up the context window to utilize the entire context as memory is computationally prohibitive, rendering it impractical. (2) Explicit retrieval relies on hand-crafted heuristic rules; the rigidity of these rules severely restricts the model's generalization across diverse camera trajectories and scenes. (3) In contrast, our implicit retrieval is an attention-driven mechanism where the model performs retrieval directly within the global context, yielding both greater flexibility and superior performance.

---

## Method Overview

<div align="center">

![Method Overview](assets/pipeline.png)

</div>

A dual-branch compression network converts the historical video into compact context tokens. The context, an uncompressed sink frame, and noisy target tokens are then processed by two parallel attention branches: standard self-attention preserves the pretrained video prior, while **Retrieval Attention** uses relative camera poses to retrieve relevant history and control the target viewpoint.

### Core Innovations

| Innovation | Description |
|---|---|
| **Implicit Memory Retrieval** | The model retrieves memory directly in attention space instead of relying on hand-crafted retrieval rules. |
| **Relative Pose Encoding** | Viewpoint information is injected into tokens so attention scores naturally encode camera similarity. |
| **Context Compression** | Historical frames are converted into compact memory tokens, reducing overhead for long-horizon generation. |

---

## SceneFly Dataset

**SceneFly** is a large-scale synthetic dataset featuring realistic camera trajectories and frame-level annotations to train and evaluate long-horizon video world models. It is built in Unreal Engine 5 and contains roughly 1,000 minutes of footage across 100 varied environments, together with precise camera parameters for evaluating long-horizon camera-aware generation. The dataset will be released separately.

---

## Open-source Inference

The inference code supports four modes through the same `inference.py` entry point:

| Mode | Input | Description |
|------|-------|-------------|
| `camera` | image + camera trajectory | I2V following an explicit camera pose sequence |
| `action` | image + action commands | I2V driven by action commands, generated autoregressively |
| `hardcut` | image + action commands with `skip:` | Multi-shot I2V with hard cuts; `skip:` advances camera state without rendering |
| `continue` | video + context poses + action commands | V2V continuation from an existing context video |

`action`, `hardcut`, and `continue` share the same action syntax. Commands are comma-separated and can be composed with `+`, for example `w+right`. A `skip:` prefix is used in hardcut mode to create discontinuous camera transitions.

---

## Installation

### Requirements

- Python 3.10
- PyTorch 2.7.1
- CUDA 12.8

Other CUDA-enabled PyTorch versions may also work if they are compatible with your driver and GPU.

### Environment

```bash
conda create -n car python=3.10
conda activate car
```

### Install CaR

```bash
git clone https://github.com/Orange-3DV-Team/CaR.git
cd CaR

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### Checkpoints

Prepare the following checkpoints:

- [Wan2.2-TI2V-5B base checkpoint](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
- CaR checkpoint

In the commands below:

- `/path/to/Wan2.2-TI2V-5B` is the Wan2.2-TI2V-5B checkpoint directory.
- `/path/to/car_checkpoint` is the CaR checkpoint directory or checkpoint file supported by `core/utils.py`.

---

## Demo

`demo.sh` runs all `sample_*` examples for one mode. Pass the Wan2.2 checkpoint path and CaR checkpoint path after the mode:

```bash
# I2V with an explicit camera pose trajectory
bash demo.sh camera   /path/to/Wan2.2-TI2V-5B /path/to/car_checkpoint

# I2V driven by action commands
bash demo.sh action   /path/to/Wan2.2-TI2V-5B /path/to/car_checkpoint

# Multi-shot I2V with action commands and skip: hard cuts
bash demo.sh hardcut  /path/to/Wan2.2-TI2V-5B /path/to/car_checkpoint

# V2V continuation from a context video
bash demo.sh continue /path/to/Wan2.2-TI2V-5B /path/to/car_checkpoint
```

Generated videos are saved under `output/`. The default inputs, motion sequences, and generation settings are defined near the top of `demo.sh`.

---

## Single-input Inference

Use `infer.sh` to generate one video from a single image or image folder plus a prompt. Pass the two checkpoint paths as arguments:

```bash
bash infer.sh /path/to/Wan2.2-TI2V-5B /path/to/car_checkpoint
```

Edit the input fields near the top of `infer.sh`:

```bash
image="examples/i2v/images/sample_001"      # image file or folder of images
prompt=""                                    # text prompt; empty -> use prompt.txt next to input
motion="w+up,down,skip:right,a,skip:left,w"  # action commands; "skip:" inserts a hard cut
traj=""                                       # camera trajectory json; leave empty to use motion
output_dir="output/infer"
```

Mode selection in `infer.sh`:

- If `traj` is set, camera mode is used and `motion` is ignored.
- If `traj` is empty and `motion` contains `skip:`, hardcut mode is used.
- Otherwise, action mode is used.

Camera trajectory files should follow the format of [`examples/i2v/camera/traj.json`](examples/i2v/camera/traj.json).

---

## Calling `inference.py` Directly

`demo.sh` and `infer.sh` are thin wrappers around `inference.py`. You can call it directly for custom inputs and settings.

<details>
<summary><b>Camera mode</b></summary>

```bash
python inference.py \
  --mode camera \
  --checkpoint_dir /path/to/Wan2.2-TI2V-5B \
  --car_checkpoint /path/to/car_checkpoint \
  --input_path examples/i2v/images/sample_001 \
  --target_poses examples/i2v/camera/traj.json \
  --output_dir output/camera_demo/sample_001_traj \
  --height 480 --width 832 --frame_num 81 \
  --sampling_steps 50 --guide_scale 3.0 \
  --seed 0
```

</details>

<details>
<summary><b>Action mode</b></summary>

```bash
python inference.py \
  --mode action \
  --checkpoint_dir /path/to/Wan2.2-TI2V-5B \
  --car_checkpoint /path/to/car_checkpoint \
  --input_path examples/i2v/images/sample_001 \
  --motion_sequence "right,right,right,left,left,left" \
  --step_size 4.0 \
  --rotate_angle 30.0 \
  --pitch_angle 15.0 \
  --output_dir output/action_demo/sample_001 \
  --height 480 --width 832 --frame_num 81 \
  --sampling_steps 50 --guide_scale 3.0 \
  --seed 0
```

</details>

<details>
<summary><b>Hardcut mode</b></summary>

```bash
python inference.py \
  --mode hardcut \
  --checkpoint_dir /path/to/Wan2.2-TI2V-5B \
  --car_checkpoint /path/to/car_checkpoint \
  --input_path examples/i2v/images/sample_001 \
  --motion_sequence "w+up,down,skip:right,a,skip:left,w" \
  --step_size 4.0 \
  --rotate_angle 30.0 \
  --pitch_angle 15.0 \
  --output_dir output/hardcut_demo/sample_001 \
  --height 480 --width 832 --frame_num 81 \
  --sampling_steps 50 --guide_scale 3.0 \
  --seed 0
```

</details>

<details>
<summary><b>Continue mode</b></summary>

```bash
python inference.py \
  --mode continue \
  --checkpoint_dir /path/to/Wan2.2-TI2V-5B \
  --car_checkpoint /path/to/car_checkpoint \
  --input_path examples/continue/sample_001/context_video.mp4 \
  --context_poses examples/continue/sample_001/context_poses.json \
  --motion_sequence "d,left" \
  --step_size 4.0 \
  --rotate_angle 30.0 \
  --pitch_angle 15.0 \
  --output_dir output/continue_demo/sample_001 \
  --height 480 --width 832 --frame_num 81 \
  --sampling_steps 50 --guide_scale 3.0 \
  --seed 0
```

</details>

---

## Inputs

- **Image input:** `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`
- **Video input:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- **Image-folder input:** a directory containing sorted image files
- **Pose input:** `.json`, `.npy`, or `.npz`

Pose files should contain camera-to-world matrices with shape `[N, 3, 4]` or `[N, 4, 4]`. For JSON files, use:

```json
{
  "poses": [
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]
  ]
}
```

Action commands are comma-separated commands from `w`, `s`, `a`, `d`, `left`, `right`, `up`, and `down`. Composite commands are joined by `+`, for example `w+right`. In hardcut mode, prefix a command with `skip:` to advance the camera without rendering that segment.

---

## Project Layout

```text
CaR/
├── inference.py              # Unified inference entry
├── demo.sh                   # Batch demo script for camera/action/hardcut/continue
├── infer.sh                  # Single-input inference script
├── requirements.txt
├── core/utils.py             # Checkpoint loading utilities
├── wan/                      # Wan2.2-related model code
├── assets/                   # README figures and demo media
└── examples/
    ├── gen_indicator_video.py
    ├── generate_test_poses.py
    ├── camera_extrinsics.json
    ├── i2v/
    │   ├── images/
    │   └── camera/traj.json
    └── continue/
```

---

## Notes

- Coordinate system: CV convention (`x = right`, `y = down`, `z = forward`).
- Default resolution: 480×832 with 81 frames.
- `camera_condition=relray_absmap` and `rope_mode=rope+memrope` follow the trained checkpoint.
- Memory and HR-frame conditioning are enabled by default; use `--no_memory` or `--no_hr_frame` to disable them.
- Segment files are named `segment_<idx>_<label>.mp4` for action/hardcut/continue and `segment_<idx>.mp4` for camera mode.

---

## Citation

```bibtex
@article{peng2026car,
  title   = {Compression and Retrieval: Implicit Memory Retrieval for Video World Models},
  author  = {Peng, Zhan and others},
  journal = {arXiv preprint arXiv:2606.23105},
  year    = {2026}
}
```
