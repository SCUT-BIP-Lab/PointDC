import torch, os, argparse, faiss
import torch.nn.functional as F
from datasets.ScanNet import Scannetval, cfl_collate_fn_val
import numpy as np
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from sklearn.cluster import KMeans
from models.fpn import Res16FPN18
from lib.utils import get_fixclassifier, init_get_sp_feature, faiss_cluster, worker_init_fn, set_seed, compute_seg_results, write_list
from tqdm import tqdm
from os.path import join
from datasets.ScanNet import Scannettrain, Scannetdistill, Scannetval, cfl_collate_fn, cfl_collate_fn_distill, cfl_collate_fn_val
from datetime import datetime
from sklearn.cluster._kmeans import k_means
from models.pretrain_models import SubModel
###
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='data/ScanNet4growsp/train',
                        help='pont cloud data path')
    parser.add_argument('--feats_path', type=str, default='data/ScanNet4growsp/traindatas',
                        help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default= 'data/scans_growed_sp',
                        help='initial sp path')
    parser.add_argument('--save_path', type=str, default='ckpt/ScanNet/',
                        help='model savepath')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--workers', type=int, default=10, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--voxel_size', type=float, default=0.02, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=6, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=30, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=20, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=128, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    return parser.parse_args()


def eval_once(args, model, test_loader, classifier, use_sp=False):
    model.mode = 'train'
    all_preds, all_label = [], []
    test_loader_bar = tqdm(test_loader)
    for data in test_loader_bar:
        test_loader_bar.set_description('Start eval...')
        with torch.no_grad():
            coords, features, inverse_map, labels, index, region = data

            in_field = ME.TensorField(features, coords, device=0)
            feats_nonorm = model(in_field)
            feats_norm = F.normalize(feats_nonorm)

            region = region.squeeze()
            if use_sp:
                region_inds = torch.unique(region)
                region_feats = []
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        region_feats.append(feats_norm[valid_mask].mean(0, keepdim=True))
                region_feats = torch.cat(region_feats, dim=0)

                scores = F.linear(F.normalize(feats_norm), F.normalize(classifier.weight))
                preds = torch.argmax(scores, dim=1).cpu()
                
                region_scores = F.linear(F.normalize(region_feats), F.normalize(classifier.weight))
                region_no = 0
                for id in region_inds:
                    if id != -1:
                        valid_mask = id == region
                        preds[valid_mask] = torch.argmax(region_scores, dim=1).cpu()[region_no]
                        region_no +=1
            else:
                scores = F.linear(F.normalize(feats_nonorm), F.normalize(classifier.weight))
                preds = torch.argmax(scores, dim=1).cpu()

            preds = preds[inverse_map.long()]
            all_preds.append(preds[labels!=args.ignore_label]), all_label.append(labels[[labels!=args.ignore_label]])

    return all_preds, all_label

def eval(epoch, args, mode='svc'):
    ## Model
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args, mode='train').cuda()
    model.load_state_dict(torch.load(os.path.join(args.save_path, mode, 'model_' + str(epoch) + '_checkpoint.pth')))
    model.eval()
    ## Merge Cluster Centers
    primitive_centers = torch.load(os.path.join(args.save_path, mode, 'cls_' + str(epoch) + '_checkpoint.pth'))
    centroids, _, _ = k_means(primitive_centers.cpu(), n_clusters=args.semantic_class, random_state=None, n_init=20)
    centroids = F.normalize(torch.FloatTensor(centroids), dim=1).cuda()
    cls = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.semantic_class, centroids=centroids).cuda()
    cls.eval()

    val_dataset = Scannetval(args)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=cfl_collate_fn_val(), num_workers=args.cluster_workers, pin_memory=True)

    preds, labels = eval_once(args, model, val_loader, cls, use_sp=True)
    all_preds = torch.cat(preds).numpy()
    all_labels = torch.cat(labels).numpy()
    
    o_Acc, m_Acc, s = compute_seg_results(args, all_labels, all_preds)
    
    return o_Acc, m_Acc, s

def eval_by_cluster(args, epoch, mode='svc'):
    ## Prepare Data
    trainset = Scannettrain(args)
    cluster_loader = DataLoader(trainset, batch_size=1, shuffle=True, collate_fn=cfl_collate_fn(), \
                                num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn(args.seed))
    val_dataset = Scannetval(args)
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=cfl_collate_fn_val(), num_workers=args.cluster_workers, pin_memory=True)
    ## Define model
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, conv1_kernel_size=args.conv1_kernel_size, config=args, mode='train').cuda()
    model.load_state_dict(torch.load(os.path.join(args.save_path, mode, 'model_' + str(epoch) + '_checkpoint.pth')))
    model.eval()
    ## Get sp features
    print('Start to get sp features...')
    sp_feats_list = init_get_sp_feature(args, cluster_loader, model)
    sp_feats = torch.cat(sp_feats_list, dim=0)
    print('Start to train faiss...')
    ## Train faiss module
    _, primitive_centers = faiss_cluster(args, sp_feats.cpu().numpy())
    ## Merge Primitive
    centroids, _, _ = k_means(primitive_centers.cpu(), n_clusters=args.semantic_class, random_state=None, n_init=20, n_jobs=20)
    centroids = F.normalize(torch.FloatTensor(centroids), dim=1).cuda()
    ## Get cls
    cls = get_fixclassifier(in_channel=args.feats_dim, centroids_num=args.semantic_class, centroids=centroids).cuda()
    cls.eval()
    ## eval
    preds, labels = eval_once(args, model, val_loader, cls, use_sp=True)
    all_preds, all_labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    o_Acc, m_Acc, s = compute_seg_results(args, all_labels, all_preds)
    return o_Acc, m_Acc, s

if __name__ == '__main__':
    args = parse_args()
    expnames = ['230119_0_All']
    epoches = [60]
    seeds = [12, 43, 56, 78, 90]
    for expname in expnames:
        args.save_path = 'ckpt/ScanNet'
        args.save_path = join(args.save_path, expname)
        assert os.path.exists(args.save_path), 'There is no {} !!!'.format(expname)
        if not os.path.exists(join('results', expname)):
            os.makedirs(join('results', expname))
        for epoch in epoches:
            results = []
            results.append('Eval exp {}\n'.format(expname))
            results.append("Eval time: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M")))
            results_file = join('results', expname, 'eval_{}.txt'.format(str(epoch)))
            print('Eval {}, save file to {}'.format(expname, results_file))
            for seed in seeds:
                args.seed = seed
                set_seed(args.seed)
                o_Acc, m_Acc, s = eval_by_cluster(args, epoch, mode='svc')
                print('Epoch {:02d} Seed {}: oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, seed, o_Acc, m_Acc) + s)
                results.append('Epoch {:02d} Seed {}: oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, seed, o_Acc, m_Acc) + s +'\n')
            write_list(results_file, results)
            print('\n')