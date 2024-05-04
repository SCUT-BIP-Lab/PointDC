from pclpy import pcl
import pclpy
import numpy as np
from scipy import stats
from os.path import join, exists, dirname, abspath
import sys, glob
import json

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from lib.helper_ply import read_ply, write_ply
import os
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

trainval_file = [line.rstrip() for line in open(join(BASE_DIR, 'ScanNet_splits/scannetv2_trainval.txt'))]

colormap = []
for _ in range(1000):
    for k in range(12):
        colormap.append(plt.cm.Set3(k))
    for k in range(9):
        colormap.append(plt.cm.Set1(k))
    for k in range(8):
        colormap.append(plt.cm.Set2(k))
colormap.append((0, 0, 0, 0))
colormap = np.array(colormap)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='data/ScanNet/scans/', help='raw data path') # *.segs.json文件路径
parser.add_argument('--sp_path', type=str, default='data/ScanNet/initial_superpoints/') # 保存超点云文件路径
parser.add_argument('--pc_path', type=str, default='data/ScanNet/train/') # 处理后点云文件路径
args = parser.parse_args()

args.input_path = join(ROOT_DIR, args.input_path)
args.sp_path    = join(ROOT_DIR, args.sp_path)
args.pc_path    = join(ROOT_DIR, args.pc_path)

vis = True

def read_superpoints(path):
    # 读取json文件
    with open(path, 'r', encoding='utf-8') as f:  
        json_data = json.load(f)
    # 读取SP，重新排序
    ori_sp = json_data['segIndices']
    unique_vals = sorted(np.unique(ori_sp))
    sp_labels = np.searchsorted(unique_vals, ori_sp)
    # 保存文件
    if not os.path.exists(args.sp_path):
        os.makedirs(args.sp_path)
    np.save(args.sp_path + path.split('/')[-2] + '_superpoint.npy', sp_labels)
    # 可视化文件
    if vis:
        vis_path = args.sp_path+'/vis/'
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
        pc = Path(join(args.pc_path, path.split('/')[-2]+'.ply')) # pc文件路径
        data = read_ply(pc)
        coords = np.vstack((data['x'], data['y'], data['z'])).T.copy().astype(np.float32)
        colors = np.zeros_like(coords)
        for p in range(colors.shape[0]):
            colors[p] = 255 * (colormap[sp_labels[p].astype(np.int32)])[:3]
        colors = colors.astype(np.uint8)
        write_ply(vis_path + '/' + path.split('/')[-2], [coords, colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

print('start constructing initial superpoints')
folders = sorted(glob.glob(args.input_path + '*/*.segs.json'))
# read_superpoints(folders[0])
pool = ProcessPoolExecutor(max_workers=40)
result = list(pool.map(read_superpoints, folders))