[![NVIDIA Source Code License](https://img.shields.io/badge/license-NSCL-blue.svg)](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg)

# SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

<!-- ![image](resources/image.png) -->

<div align="center">
  <img src="./resources/image.png" height="400">
</div>
<p align="center">
  Figure 1: Performance of SegFormer-B0 to SegFormer-B5.
</p>

### [Project page](https://github.com/NVlabs/SegFormer) | [Paper](https://arxiv.org/abs/2105.15203) | [Demo (Youtube)](https://www.youtube.com/watch?v=J0MoRQzZe8U) | [Demo (Bilibili)](https://www.bilibili.com/video/BV1MV41147Ko/) | [Intro Video](https://www.youtube.com/watch?v=nBjXyOLTCHU)

SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers.`<br>`
[Enze Xie](https://xieenze.github.io/), [Wenhai Wang](https://whai362.github.io/), [Zhiding Yu](https://chrisding.github.io/), [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/), [Jose M. Alvarez](https://rsu.data61.csiro.au/people/jalvarez/), and [Ping Luo](http://luoping.me/).`<br>`
NeurIPS 2021.

This repository contains the official PyTorch implementation of training & evaluation code and the pretrained models for [SegFormer](https://arxiv.org/abs/2105.15203).

SegFormer is a simple, efficient and powerful semantic segmentation method, as shown in Figure 1.

**✅ UPGRADED**: This repository now uses **MMSegmentation v1.2.2** (latest stable) with full compatibility for PyTorch 2.x, MMCV 2.x, and MMEngine.

🔥🔥 SegFormer is on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/segformer). 🔥🔥

## 🚀 Quick Upgrade from MMSegmentation 0.x

If you're upgrading from MMSegmentation 0.x, run the automated upgrade script:

```bash
python upgrade_to_v1.py
```

This will:

- Install latest dependencies (PyTorch 2.8.0, MMCV 2.2.0, MMEngine 0.10.7)
- Update import statements for compatibility
- Install the upgraded MMSegmentation

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA 10.2+ (optional, for GPU acceleration)

### Install Dependencies

```bash
# Install PyTorch (choose your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install MMEngine and MMCV
pip install -U openmim
mim install mmengine mmcv

# Install other requirements
pip install timm==0.9.12
pip install -r requirements/runtime.txt
```

### Install MMSegmentation

```bash
cd segformer_pytorch
pip install -e .
```

## Migration from MMSegmentation 0.x

This repository has been upgraded to MMSegmentation 1.x with breaking changes. Key differences:

### 🔄 Configuration Files

- `data` → `train_dataloader`, `val_dataloader`, `test_dataloader`
- `optimizer` → `optim_wrapper`
- `lr_config` → `param_scheduler`
- `evaluation` → `val_evaluator`, `test_evaluator`

### 🔄 Training Scripts

- Old: `python tools/train.py config.py`
- New: `python tools/train.py config.py` (same command, updated internals)

### 🔄 API Changes

- MMCV Runner → MMEngine Runner
- Updated import paths and function signatures

### 📚 Full Migration Guide

See the [official migration guide](https://mmsegmentation.readthedocs.io/en/latest/migration/interface.html) for detailed instructions.

## Evaluation

Download `trained weights`.
(
[google drive](https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA?usp=sharing) |
[onedrive]()
)

Example: evaluate `SegFormer-B1` on `ADE20K`:

```
# Single-gpu testing
python tools/test.py local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint_file

# Multi-gpu testing
./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint_file <GPU_NUM>

# Multi-gpu, multi-scale testing
tools/dist_test.sh local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint_file <GPU_NUM> --aug-test
```

## Training

Download `weights`
(
[google drive](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing) |
[onedrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ)
)
pretrained on ImageNet-1K, and put them in a folder `pretrained/`.

Example: train `SegFormer-B1` on `ADE20K`:

```
# Single-gpu training
python tools/train.py local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py

# Multi-gpu training
./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py <GPU_NUM>
```

## Visualize

Here is a demo script to test a single image. More details refer to [MMSegmentation&#39;s Doc](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).

```shell
python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${DEVICE_NAME}] [--palette-thr ${PALETTE}]
```

Example: visualize `SegFormer-B1` on `CityScapes`:

```shell
python demo/image_demo.py demo/demo.png local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py \
/path/to/checkpoint_file --device cuda:0 --palette cityscapes
```

## License

Please check the LICENSE file. SegFormer may be used non-commercially, meaning for research or
evaluation purposes only. For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).

## Citation

```
@inproceedings{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  booktitle={Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```
