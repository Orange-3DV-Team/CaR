<div align="center">

# Compression and Retrieval: Implicit Memory Retrieval for Video World Models

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

[![Paper](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2606.23105)
[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=googlechrome&logoColor=white)](https://orange-3dv-team.github.io/CaR/)
[![Code](https://img.shields.io/badge/Code-GitHub-black?logo=github)](https://github.com/Orange-3DV-Team/CaR)
[![Model](https://img.shields.io/badge/🤗%20Model-CaR-orange)](https://huggingface.co/Orange-3DV-Team/CaR/tree/main)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-SceneFly-yellow)](https://huggingface.co/datasets/Orange-3DV-Team/SceneFly)


</div>

---

<table>
<tr>
<td align="center"><b>I2V — Camera</b></td>
<td align="center"><b>V2V — History Extension</b></td>
</tr>
<tr>
<td>

https://github.com/user-attachments/assets/dac27cf2-c7ee-42cf-88c8-e57a3e604d98

</td>
<td>

https://github.com/user-attachments/assets/0db2cf63-6f41-4bc9-9844-785e63950b31

</td>
</tr>
<tr>
<td align="center"><b>I2V — Action</b></td>
<td align="center"><b>Hard-cut</b></td>
</tr>
<tr>
<td>

https://github.com/user-attachments/assets/1800c000-edb9-4d0b-8051-67094f8b38d6

</td>
<td>

https://github.com/user-attachments/assets/0183c0ab-1801-4c65-8ebf-4264ba9334db

</td>
</tr>
</table>

<p align="center">
Additional examples are available on the <a href="https://orange-3dv-team.github.io/CaR/">project site</a>.
</p>

---

## TODO

- [x] Release inference code
- [x] Release [CaR checkpoint](https://huggingface.co/Orange-3DV-Team/CaR/tree/main)
- [x] Release partial [SceneFly dataset](https://huggingface.co/datasets/Orange-3DV-Team/SceneFly)
- [ ] Release full SceneFly dataset
- [ ] Release training code
- [ ] Release few-step model


---

## Method Overview

<div align="center">

![Method Overview](assets/pipeline.png)

</div>

A dual-branch compression network converts the historical video into compact context tokens. The context, an uncompressed sink frame, and noisy target tokens are then processed by two parallel attention branches: standard self-attention preserves the pretrained video prior, while **Retrieval Attention** uses relative camera poses to retrieve relevant history and control the target viewpoint.

---

## SceneFly Dataset

**SceneFly** is a large-scale synthetic dataset featuring realistic camera trajectories and frame-level annotations to train and evaluate long-horizon video world models. It is built in Unreal Engine 5 and contains roughly 1,000 minutes of footage across 100 varied environments, together with precise camera parameters for evaluating long-horizon camera-aware generation. We currently release a partial version of SceneFly (about half of the full dataset) on [Hugging Face](https://huggingface.co/datasets/Orange-3DV-Team/SceneFly); the complete dataset will be provided in a future release.

---

## Open-source Inference

The inference code automatically selects one of four modes from the provided inputs:

| Mode | Input | Description |
|------|-------|-------------|
| `camera` | image + camera trajectory | I2V following an explicit camera pose sequence |
| `action` | image + action commands | I2V driven by action commands, generated autoregressively |
| `hardcut` | image + action commands with `skip:` | Multi-shot I2V with hard cuts; `skip:` advances camera state without rendering |
| `continue` | video + context poses + action commands | V2V continuation from an existing context video |


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

Flash Attention is recommended for efficient inference. Install a version compatible with your PyTorch/CUDA environment, for example:

```bash
pip install flash-attn --no-build-isolation
```

### Checkpoints

Prepare the following checkpoints:

- [Wan2.2-TI2V-5B base checkpoint](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
- [CaR checkpoint](https://huggingface.co/Orange-3DV-Team/CaR/tree/main)

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

## Input Format

The recommended `input` is a **sample-level folder**:

```text
sample/
├── context.png | context.mp4
├── prompt.txt
└── context_poses.json  # optional, used for video continuation
```

`context` can be an image or a video. `context_poses.json` is optional and is only needed when continuing from a video context.

For camera-trajectory inference, set `traj` to a pose file such as:

```bash
traj="examples/i2v/camera/traj.json"
```

---

## Inference

Edit `input`, `output_dir`, and generation parameters inside `infer.sh`, then run:

```bash
bash infer.sh /path/to/Wan2.2-TI2V-5B /path/to/car_checkpoint
```

You can also run `inference.py` directly and append the needed arguments:

```bash
python inference.py \
  --checkpoint_dir /path/to/Wan2.2-TI2V-5B \
  --car_checkpoint /path/to/car_checkpoint \
  --input_path examples/i2v/images/sample_001 \
  --output_dir output/infer \
  --prompt "" \
  --motion_sequence "w+up,down,skip:right,a,skip:left,w" \
  --target_poses examples/i2v/camera/traj.json \
  --context_poses examples/continue/sample_001/context_poses.json \
  --translation_step 4.0 \
  --rotate_angle 30.0 \
  --pitch_angle 15.0
```

Prompt behavior:

- With `--prompt "your prompt"`, the given text is used directly.
- Without `--prompt`, `inference.py` reads `prompt.txt` from the input sample folder.


`inference.py` automatically resolves sample folders: `context.png` is used for image samples, while `context.mp4` and `context_poses.json` are used for video continuation samples. It also automatically selects the mode:

- `--target_poses` is set → **camera** mode
- `--context_poses` is set, or input resolves to a video → **continue** mode
- `--motion_sequence` contains `skip:` → **hardcut** mode
- otherwise, `--motion_sequence` → **action** mode

> **Tip:** Action commands are comma-separated commands from `w`, `s`, `a`, `d`, `left`, `right`, `up`, and `down`. Use `+` to compose commands, for example `w+right`. In hardcut mode, prefix a command with `skip:` to advance the camera without rendering that segment.

---

## Notes

- Default resolution: 480×832 with 81 frames.
- `camera_condition=relray_absmap` and `rope_mode=rope+memrope` follow the trained checkpoint.
- Segment files are named `segment_<idx>_<label>.mp4` for action/hardcut/continue and `segment_<idx>.mp4` for camera mode.

---

## Citation

```bibtex
@article{peng2026car,
    title={Compression and Retrieval: Implicit Memory Retrieval for Video World Models},
    author={Peng, Zhan and Ma, Jie and Sun, Huiqiang and Gao, Chong and Xue, Zhijie and Pan, Zhiyu and Cao, Zhiguo and Liang, Jun and Li, Jing},
    journal={arXiv preprint arXiv:2606.23105},
    year={2026}
}
```
