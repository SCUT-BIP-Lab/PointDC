import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, faiss, shutil, random
import faiss
import shutil
from sklearn.cluster import KMeans
import MinkowskiEngine as ME
from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2

from tqdm import tqdm
from torch_scatter import scatter

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class FocalLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, gamma=2):
        super(FocalLoss,self).__init__()
        self.gamma = gamma
        self.weight = weight        # 是tensor数据格式的列表
        self.ignore_index = ignore_index

    def forward(self, preds, labels):
        """
        preds:logist输出值
        labels:标签
        """
        if self.ignore_index is not None:
            mask = torch.nonzero(labels!=self.ignore_index)
            preds = preds[mask].squeeze(1)
            labels = labels[mask].squeeze(1)

        preds = F.softmax(preds,dim=1)
        eps = 1e-7

        target = self.one_hot(preds.size(1), labels)

        ce = -torch.log(preds+eps) * target
        floss = torch.pow((1-preds), self.gamma) * ce
        if self.weight is not None:
            floss = torch.mul(floss, self.weight)
        floss = torch.sum(floss, dim=1)
        return torch.mean(floss)

    def one_hot(self, num, labels):
        one = torch.zeros((labels.size(0),num)).cuda()
        one[range(labels.size(0)),labels] = 1
        return one

class MseMaskLoss(nn.Module):
    def __init__(self):
        super(MseMaskLoss, self).__init__()
    
    def forward(seelf, sourcefeats, targetfeats):
        targetfeats = F.normalize(targetfeats, dim=1, p=2)
        sourcefeats = F.normalize(sourcefeats, dim=1, p=2)
        mseloss = (sourcefeats - targetfeats)**2
        
        return mseloss.mean()

def worker_init_fn(seed):
    return lambda x: np.random.seed(seed + x)

def set_seed(seed):
    """
    Unfortunately, backward() of [interpolate] functional seems to be never deterministic.

    Below are related threads:
    https://github.com/pytorch/pytorch/issues/7068
    https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi
    """
    # Use random seed.
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
     
def init_get_sp_feature(args, loader, model, submodel=None):
    loader.dataset.mode = 'cluster'

    region_feats_list = []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords, features, _, labels, inverse_map, pseudo_labels, inds, region, index, scenenames = data
            region      = region.squeeze()+1
            raw_region  = region.clone()

            in_field = ME.TensorField(features, coords, device=0)

            if submodel is not None:
                feats_TensorField = model(in_field)
                feats_nonorm = submodel(feats_TensorField)
            else:
                feats_nonorm = model(in_field)
            feats_nonorm = feats_nonorm[inverse_map.long()] # 获取points feats
            feats_norm = F.normalize(feats_nonorm, dim=1) ## NOTE 可能需要normalize？没啥区别

            region_feats = scatter(feats_norm, raw_region.cuda(), dim=0, reduce='mean')

            valid_mask = torch.logical_and(labels!=-1, region>0) # 获取带训练的mask区域
            # labels = labels[valid_mask]
            region_masked = region[valid_mask].long()
            region_masked_num = torch.unique(region_masked)
            region_masked_feats = region_feats[region_masked_num]
            region_masked_feats_norm = F.normalize(region_masked_feats, dim=1).cpu()

            region_feats_list.append(region_masked_feats_norm)
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))
            
    return region_feats_list

def init_get_pseudo(args, loader, model, centroids_norm, submodel=None):

    pseudo_label_folder = args.pseudo_path + '/'
    if not os.path.exists(pseudo_label_folder): os.makedirs(pseudo_label_folder)

    all_pseudo = []
    all_label = []
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            coords, features, _, labels, inverse_map, pseudo_labels, inds, region, index, scenenames = data
            region = region.squeeze()+1
            raw_region = region.clone()
            region_mask = raw_region >= 0

            in_field = ME.TensorField(features, coords, device=0)

            if submodel is not None:
                feats_TensorField = model(in_field)
                feats_nonorm = submodel(feats_TensorField)
            else:
                feats_nonorm = model(in_field)
            feats_nonorm = feats_nonorm[inverse_map.long()] # 获取points feats
            feats_norm = F.normalize(feats_nonorm, dim=1) ## NOTE 可能需要normalize？
            
            scores = F.linear(F.normalize(feats_norm), F.normalize(centroids_norm))
            preds = torch.argmax(scores, dim=1).cpu()
            
            region_feats = scatter(feats_norm, raw_region.cuda(), dim=0, reduce='mean')
            
            region_inds = torch.unique(region)
            region_scores = F.linear(F.normalize(region_feats), F.normalize(centroids_norm))
            region_no = 0
            for id in region_inds:
                if id != 0:
                    valid_mask = id == region
                    preds[valid_mask] = torch.argmax(region_scores, dim=1).cpu()[region_no]
                    region_no +=1

            pseudo_label_file = pseudo_label_folder + '/' + scenenames[0] + '.npy'
            np.save(pseudo_label_file, preds)

            all_label.append(labels)
            all_pseudo.append(preds)

            torch.cuda.empty_cache()
            torch.cuda.synchronize(torch.device("cuda"))

    all_pseudo = np.concatenate(all_pseudo)
    all_label = np.concatenate(all_label)

    return all_pseudo, all_label

def get_fixclassifier(in_channel, centroids_num, centroids):
    classifier = nn.Linear(in_features=in_channel, out_features=centroids_num, bias=False)
    centroids = F.normalize(centroids, dim=1)
    classifier.weight.data = centroids
    for para in classifier.parameters():
        para.requires_grad = False
    return classifier

def compute_hist(normal, bins=10, min=-1, max=1):
    ## normal : [N, 3]
    normal = F.normalize(normal)
    relation = torch.mm(normal, normal.t())
    relation = torch.triu(relation, diagonal=0) # top-half matrix
    hist = torch.histc(relation, bins, min, max)
    # hist = torch.histogram(relation, bins, range=(-1, 1))
    hist /= hist.sum()

    return hist

def faiss_cluster(args, sp_feats, metric='cosin'):
    dim = sp_feats.shape[-1]

    # define faiss module
    res = faiss.StandardGpuResources()
    fcfg = faiss.GpuIndexFlatConfig()
    fcfg.useFloat16 = False 
    fcfg.device     = 0 #NOTE: Single GPU only. 
    if metric == 'l2':
        faiss_module = faiss.GpuIndexFlatL2(res, dim, fcfg) # 欧式距离
    elif metric == 'cosin':
        faiss_module = faiss.GpuIndexFlatIP(res, dim, fcfg) #  余弦距离
    clus = faiss.Clustering(dim, args.primitive_num)
    clus.seed  = np.random.randint(args.seed)
    clus.niter = 80
    
    # train 
    clus.train(sp_feats, faiss_module)
    centroids = faiss.vector_float_to_array(clus.centroids).reshape(args.primitive_num, dim).astype('float32')
    centroids_norm = F.normalize(torch.tensor(centroids), dim=1)
    # D, I = faiss_module.search(sp_feats, 1)

    return None, centroids_norm

def cache_codes(args):
    tardir = os.path.join(args.save_path, 'cache_code')
    if not os.path.exists(tardir):
        os.makedirs(tardir)
    try:
        all_files = os.listdir('./')
        pyfile_list = [file for file in all_files if file.endswith(".py")]
        shutil.copytree(r'./lib', os.path.join(tardir, r'lib'))
        shutil.copytree(r'./data_prepare', os.path.join(tardir, r'data_prepare'))
        shutil.copytree(r'./datasets', os.path.join(tardir, r'datasets'))
        for pyfile in pyfile_list:
            shutil.copy(pyfile, os.path.join(tardir, pyfile))
    except:
        pass
    
def compute_seg_results(args, all_labels, all_preds):
    '''Unsupervised, Match pred to gt'''
    sem_num = args.semantic_class
    mask = (all_labels >= 0) & (all_labels < sem_num)
    histogram = np.bincount(sem_num * all_labels[mask] + all_preds[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    '''Hungarian Matching'''
    m = linear_assignment(histogram.max() - histogram)
    o_Acc = histogram[m[:, 0], m[:, 1]].sum() / histogram.sum()*100.
    m_Acc = np.mean(histogram[m[:, 0], m[:, 1]] / histogram.sum(1))*100
    hist_new = np.zeros((sem_num, sem_num))
    for idx in range(sem_num):
        hist_new[:, idx] = histogram[:, m[idx, 1]]
    '''Final Metrics'''
    tp = np.diag(hist_new)
    fp = np.sum(hist_new, 0) - tp
    fn = np.sum(hist_new, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)

    return o_Acc, m_Acc, s

def write_list(file_path, contents):
    with open(file_path, 'w') as file:
        for content in contents:
            file.write(content)
