[![arXiv](https://img.shields.io/badge/arXiv-2304.08965-b31b1b.svg)](https://arxiv.org/abs/2304.08965)
[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

## PointDC:Unsupervised Semantic Segmentation of 3D Point Clouds via Cross-modal Distillation and Super-Voxel Clustering (ICCV 2023)

### Overview

We propose an unsupervised point clouds semantic segmentation framework, called  **PointDC**.

<p align="center">
<img src="figs/framework.jpg" alt="drawing" width=800/>
</p>

## NOTE
 This project is based on Minkowski Engine and refers to the code from [growsp](https://github.com/vLAR-group/GrowSP), but the methods used are consistent with the original paper.

## TODO
- [x] Release code deployed on the ScanNet dataset and model weight files
- [x] Release code deployed on the S3DIS dataset and model weight files
- [x] Release Spare Feature Volume files

## 1. Setup
Setting up for this project involves installing dependencies. 

### Installing dependencies
To install all the dependencies, please run the following:
```shell script
sudo apt install build-essential python3-dev libopenblas-dev
conda env create -f env.yaml
conda activate pointdc_mk
pip install -U MinkowskiEngine --install-option="--blas=openblas" -v --no-deps
```
## 2. Running codes
### 2.1 ScanNet
Download the ScanNet dataset from [the official website](http://kaldir.vc.in.tum.de/scannet_benchmark/documentation). 
You need to sign the terms of use. Uncompress the folder and move it to 
`${your_ScanNet}`.
- Download sp feats files from [here](https://pan.baidu.com/s/1ibxoq3HyxRJa3KrnPafCWw?pwd=6666), and put it in the right path.


- Preparing the dataset:
```shell script
python data_prepare/data_prepare_ScanNet.py --data_path ${your_ScanNet}
```
This code will preprcocess ScanNet and put it under `./data/ScanNet/processed`

- Construct initial superpoints:
```shell script
python data_prepare/initialSP_prepare_ScanNet.py
```
This code will construct superpoints on ScanNet and put it under `./data/ScanNet/initial_superpoints`

- Training:
```shell script
CUDA_VISIBLE_DEVICES=0, python train_ScanNet.py --expname ${your_experiment_name}
```
The output model and log file will be saved in `./ckpt/ScanNet` by default.

- Evaling:
Revise experiment name ```expnames=[eval_experiment_name]```in Lines 141. 
```shell script
CUDA_VISIBLE_DEVICES=0, python eval_ScanNet.py
```

### 2.2 S3DIS
Download the S3DIS dataset from [the official website](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1&pli=1), download the files named **Stanford3dDataset_v1.2.zip**.
Uncompress the folder and move it to '${your_S3DIS}'. And there is an error in line 180389 of file Area_5/hallway_6/Annotations/ceiling_1.txt. It need to be fixed manually.
- Download sp feats and sp files from [here](https://pan.baidu.com/s/1ibxoq3HyxRJa3KrnPafCWw?pwd=6666), and put it in the right path.
>Due to the randomness in the construction of super-voxels, different super-voxels will lead to different super-voxel features. Therefore, we provide both super-voxel features and corresponding supe-voxels. This only affects the distillation stage.

- Preparing the dataset:
```shell script
python data_prepare/data_prepare_S3DIS.py --data_path ${your_S3DIS}
```
This code will preprcocess S3DIS and put it under `./data/S3DIS/processed`

- Construct initial superpoints:
```shell script
python data_prepare/initialSP_prepare_S3DIS.py
```
This code will construct superpoints on S3DIS and put it under `./data/S3DIS/initial_superpoints`

- Training:
```shell script
CUDA_VISIBLE_DEVICES=0, python train_S3DIS.py --expname ${your_experiment_name}
```
The output model and log file will be saved in `./ckpt/S3DIS` by default.

- Evaling:
Revise experiment name `expnames=[eval_experiment_name]`.
```shell script
CUDA_VISIBLE_DEVICES=0, python eval_S3DIS.py
```

## 3. Model Weights Files
The trained models and other processed files can be found at [here](https://pan.baidu.com/s/1ibxoq3HyxRJa3KrnPafCWw?pwd=6666).

## Acknowledge
[MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)

[growsp](https://github.com/vLAR-group/GrowSP)
