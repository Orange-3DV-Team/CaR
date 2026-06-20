# CaR

### Compression and Retrieval: Implicit Memory Retrieval for Video World Models

**Zhan Peng<sup>1</sup>, Jie Ma<sup>2</sup>, Huiqiang Sun<sup>1</sup>, Chong Gao<sup>3</sup>, Zhijie Xue<sup>1</sup>, Zhiyu Pan<sup>1</sup>, Zhiguo Cao<sup>1\*</sup>, Jun Liang<sup>2</sup>, Jing Li<sup>2</sup>**

<sup>1</sup>Huazhong University of Science and Technology &nbsp; <sup>2</sup>HUJING Digital Media & Entertainment Group &nbsp; <sup>3</sup>Sun Yat-sen University &nbsp; <sup>\*</sup>Corresponding author

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://orange-3dv-team.github.io/CaR/)
![Dataset](https://img.shields.io/badge/SceneFly_Dataset-Coming_Soon-lightgrey)

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

https://github.com/user-attachments/assets/dd8ae154-1ffe-4d76-97ca-3d3dbb7d0ae5

</td>
<td>

https://github.com/user-attachments/assets/055d5b61-a5cd-402f-bd80-7e5bde438767

</td>
</tr>
</table>

<p align="center">
  🎬 More demos on the <b><a href="https://orange-3dv-team.github.io/CaR/">Project Page</a></b>
</p>

---

## Abstract

Video world models hold promise for simulating interactive environments, yet maintaining consistent long-term memory across complex camera trajectories remains a critical challenge. Existing methods typically rely on computationally expensive context scaling or rigid heuristic retrieval mechanisms, which lacks generalization to varying camera trajectories and environments. In this paper, we propose **CaR** (**C**ompression **a**nd **R**etrieval), an attention-driven implicit memory retrieval mechanism to overcome these limitations. By injecting viewpoint information via positional encoding, our method performs flexible memory retrieval through attention computation. To efficiently process extended contexts with minimal computational overhead, we further introduce a lightweight context compression network. Furthermore, we construct **SceneFly**, a large-scale synthetic dataset featuring realistic camera trajectories and frame-level annotations to train and evaluate long-horizon video world models. Extensive experiments demonstrate that our approach achieves state-of-the-art results on established benchmarks and exhibits strong generalization to open-domain scenes.

---

## Method

![Pipeline](assets/pipeline.png)

A dual-branch compression network converts the historical video into compact context tokens. The context, an uncompressed sink frame, and noisy target tokens are then processed by two parallel attention branches: standard self-attention preserves the pretrained video prior, while **Retrieval Attention** uses relative camera poses to retrieve relevant history and control the target viewpoint.

---

## SceneFly Dataset

**SceneFly** is a large-scale synthetic dataset built with Unreal Engine 5, containing approximately 1,000 minutes of video from 100 diverse indoor, outdoor, and stylized scenes, with exact frame-level camera intrinsics and extrinsics. It is specifically designed for training and evaluating long-horizon video world models with complex revisiting trajectories.

---

## Demo

Visit the **[project page](https://github.com/Orange-3DV-Team/CaR)** for video demonstrations of:

- **Single Image Scene Exploration** — camera-controlled and action-controlled novel view synthesis from a single input image
- **History Video Extension** — scene-consistent video-to-video generation
- **Flexible Viewpoint Switching** — hard-cut generation with fully discontinuous camera trajectories

---

## Code

Code will be released soon. Stay tuned!

---

## Citation

```bibtex
@article{peng2026car,
    title={Compression and Retrieval: Implicit Memory Retrieval for Video World Models},
    author={Peng, Zhan and Ma, Jie and Sun, Huiqiang and Gao, Chong and
            Xue, Zhijie and Pan, Zhiyu and Cao, Zhiguo and
            Liang, Jun and Li, Jing},
    journal={arXiv},
    year={2026}
}
```
