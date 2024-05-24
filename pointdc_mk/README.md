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
- [ ] Release code deployed on the S3DIS dataset and model weight files
- [ ] Release code for extracting image features and image weight files
- [ ] Release Spare Feature Volume files

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
CUDA_VISIBLE_DEVICES=0, python train_ScanNet.py --expname ${your_experiment_name}$
```
The output model and log file will be saved in `./ckpt/ScanNet` by default.

- Evaling:
Revise experiment name ```expnames=[eval_experiment_name]```in Lines 141. 
```shell script
CUDA_VISIBLE_DEVICES=0, python eval_ScanNet.py
```

## 3. Model Weights
The trained models and other processed files can be found at [here](https://pan.baidu.com/s/1ibxoq3HyxRJa3KrnPafCWw?pwd=6666)