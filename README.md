# MemSeg: Memory-based semantic segmentation for off-road unstructured natural environments

## Introduction

This repository is a PyTorch implementation of [Memory-based semantic segmentation for off-road unstructured natural environments](https://arxiv.org/abs/2108.05635). This work is based on [semseg](https://github.com/hszhao/semseg/blob/1.0.0/README.md).

The codebase mainly uses ResNet18, ResNet50 and MobileNet-V2 as backbone with ASPP module and can be easily adapted to other basic semantic segmentation structures. 

Sample experimented dataset is [RUGD](http://rugd.vision/).

## Requirement
Hardware: >= 11G GPU memory

Software: [PyTorch](https://pytorch.org/)>=1.0.0, python3

## Usage
For installation, follow installation steps below or recommend you to refer to the instructions described [here](https://github.com/hszhao/semseg/blob/1.0.0/README.md).

For its pretrained ResNet50 backbone model, you can download from [URL](https://drive.google.com/file/d/1w5pRmLJXvmQQA5PtCbHhZc_uC4o0YbmA/view?usp=sharing).

## Getting Started

### Installation

1. Clone this repository.
```
git clone https://github.com/youngsjjn/MemSeg.git
```

2. Install Python dependencies.
```
pip install -r requirements.txt
```

### Implementation
1. Download datasets (i.e. [RUGD](http://rugd.vision/)) and change the root of data path in [config](./config/cityscapes/cityscapes_transform101.yaml).
Download data list and pre-trained backbone models (ResNet50/101/152) [here](https://drive.google.com/open?id=15wx9vOM0euyizq-M1uINgN0_wjVRf9J3).

2. Inference using pretrained model
Inference "ResNet50 + Deeplabv3" without the memory module
```
sh tool/test.sh rugd deeplab50
```
Inference "ResNet50 + Deeplabv3" with the memory module
```
sh tool/test_mem.sh rugd deeplab50
```

|  Network  |     mIoU     |
   | :-------: | :----------: |
   | ResNet18 + PSPNet  |    33.42    |
   | ResNet18 + PSPNet (Memory)  |    34.13   |
   | ResNet18 + Deeplabv3  |    33.48    |
   | ResNet18 + Deeplabv3 (Memory)  |    35.07   |
   | ResNet50 + Deeplabv3  |    36.77    |
   | ResNet50 + Deeplabv3 (Memory)  |    37.71   |

3. Train (Evaluation is included at the end of the training)
Train "ResNet50 + Deeplabv3" without the memory module
```
sh tool/train.sh rugd deeplab50
```
Train "ResNet50 + Deeplabv3" without the memory module
```
sh tool/train_mem.sh rugd deeplab50mem
```

Here, the example is for training or testing on "ResNet50 + Deeplabv3".
If you want to train other networks, please change "deeplab50" or "deeplab50mem" as a postfix of a config file name.

For example, train "ResNet18 + PSPNet":
```
sh tool/train_mem.sh rugd pspnet18mem
```
