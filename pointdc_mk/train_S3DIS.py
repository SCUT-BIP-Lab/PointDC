import os, random, time, argparse, logging, warnings, torch
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment  # pip install scikit-learn==0.22.2
from datasets.S3DIS import S3DISdistill, S3DIStrain, S3DIScluster, cfl_collate_fn_distill, cfl_collate_fn
import MinkowskiEngine as ME
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.fpn import Res16FPN18
from models.pretrain_models import SubModel, SegHead
from eval_S3DIS import eval, eval_once, eval_by_cluster
from lib.utils_s3dis import *
from sklearn.cluster import KMeans 
from os.path import join
from tqdm import tqdm
from torch.optim import lr_scheduler
warnings.filterwarnings('ignore')

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser(description='PyTorch Unsuper_3D_Seg')
    parser.add_argument('--data_path', type=str, default='data/S3DIS/', help='pont cloud data path')
    parser.add_argument('--sp_path', type=str, default= 'data/S3DIS/',  help='initial sp path')
    parser.add_argument('--expname', type=str, default= 'zdefalut', help='expname for logger')
    ###
    parser.add_argument('--save_path', type=str, default='ckpt/S3DIS/', help='model savepath')
    parser.add_argument('--max_epoch', type=list, default=[200, 30, 60], help='max epoch')
    ###
    parser.add_argument('--bn_momentum', type=float, default=0.02, help='batchnorm parameters')
    parser.add_argument('--conv1_kernel_size', type=int, default=5, help='kernel size of 1st conv layers')
    ####
    parser.add_argument('--lrs', type=list, default=[1e-3, 3e-2, 3e-2], help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD parameters')
    parser.add_argument('--dampening', type=float, default=0.1, help='SGD parameters')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='SGD parameters')
    parser.add_argument('--workers', type=int, default=8, help='how many workers for loading data')
    parser.add_argument('--cluster_workers', type=int, default=4, help='how many workers for loading data in clustering')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--log-interval', type=int, default=150, help='log interval')
    parser.add_argument('--batch_size', type=int, default=8, help='batchsize in training')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='voxel size in SparseConv')
    parser.add_argument('--input_dim', type=int, default=6, help='network input dimension')### 6 for XYZGB
    parser.add_argument('--primitive_num', type=int, default=13, help='how many primitives used in training')
    parser.add_argument('--semantic_class', type=int, default=13, help='ground truth semantic class')
    parser.add_argument('--feats_dim', type=int, default=128, help='output feature dimension')
    parser.add_argument('--ignore_label', type=int, default=-1, help='invalid label')
    parser.add_argument('--drop_threshold', type=int, default=50, help='mask counts')

    return parser.parse_args()

def main(args, logger):
    # Cross model distillation
    logger.info('**************Start Cross Model Distillation**************')
    ## Prepare Model/Optimizer
    model = Res16FPN18(in_channels=args.input_dim, out_channels=args.primitive_num, \
                                    conv1_kernel_size=args.conv1_kernel_size, args=args, mode='distill')
    submodel = SubModel(args)
    adam     = torch.optim.Adam([{'params':model.parameters()}, {'params': submodel.parameters()}], \
                                lr=args.lrs[0])
    model, submodel = model.cuda(), submodel.cuda()
    distill_loss    = MseMaskLoss().cuda()
    
    # Prepare Data
    distillset     = S3DISdistill(args)
    distill_loader = DataLoader(distillset, batch_size=args.batch_size, shuffle=True, collate_fn=cfl_collate_fn_distill(), \
                                num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn(seed))
    clusterset = S3DIScluster(args, areas=['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6'])
    cluster_loader = DataLoader(clusterset, batch_size=1, shuffle=True, collate_fn=cfl_collate_fn(), \
                                num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn(seed))
    # Distill
    for epoch in range(1, args.max_epoch[0]+1):
        distill(distill_loader, logger, model, submodel, adam, distill_loss, epoch, args.max_epoch[0])
        if epoch % 10 == 0:
            torch.save(model.state_dict(), join(args.save_path, 'cmd', 'model_' + str(epoch) + '_checkpoint.pth'))
            torch.save(submodel.state_dict(), join(args.save_path, 'cmd', 'submodule_' + str(epoch) + '_checkpoint.pth')) 

    # Compute pseudo label
    model, submodel = model.cuda(), submodel.cuda()
    
    centroids_norm = init_cluster(args, logger, cluster_loader, model, submodel=submodel)

    del adam, distill_loader, distillset, submodel
    logger.info('====>End Cross Model Distill !!!\n')

    # Super Voxel Clustering
    logger.info('**************Start Super Voxel Clustering**************')
    ## Prepare Data
    trainset = S3DIStrain(args, areas=['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_6'])
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=cfl_collate_fn(), \
                               num_workers=args.workers, pin_memory=True, worker_init_fn=worker_init_fn(seed))
    ## Warm Up
    model.mode = 'train'
    ## Prepare Model/Loss/Optimizer
    seghead = SegHead(args.feats_dim, args.primitive_num)
    seghead = seghead.cuda()
    loss = torch.nn.CrossEntropyLoss(ignore_index=-1).cuda()
    warmup_optimizer = torch.optim.SGD([{"params": seghead.parameters()}, {"params": model.parameters()}], \
                                        lr=args.lrs[1], momentum=args.momentum, \
                                        dampening=args.dampening, weight_decay=args.weight_decay)
    
    logger.info('====>Start Warm Up.')
    for epoch in range(1, args.max_epoch[1]+1):
        train(train_loader, logger, model, warmup_optimizer, loss, epoch, seghead, args.max_epoch[1])
        ### Evalutaion and Save checkpoint
        if epoch % 5 == 0:
            torch.save(model.state_dict(), join(args.save_path, 'svc', 'model_' + str(epoch) + '_checkpoint.pth'))
            torch.save(seghead.state_dict()['cluster'], join(args.save_path, 'svc', 'cls_' + str(epoch) + '_checkpoint.pth'))
            with torch.no_grad():
                o_Acc, m_Acc, s = eval(epoch, args)
                logger.info('WarmUp--Eval Epoch: {:02d}, oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc) + s+'\n')
    logger.info('====>End Warm Up !!!\n')

    # Iterative Training
    del seghead, warmup_optimizer # NOTE
    logger.info('====>Start Iterative Training.')
    iter_optimizer = torch.optim.SGD(model.parameters(), \
                                    lr=args.lrs[2], momentum=args.momentum, \
                                    dampening=args.dampening, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.StepLR(iter_optimizer, step_size=5, gamma=0.8) # step lr
    logger.info('====>Update pseudo labels.')
    centroids_norm = init_cluster(args, logger, cluster_loader, model)
    seghead = get_fixclassifier(args.feats_dim, args.primitive_num, centroids_norm).cuda()
    for epoch in range(args.max_epoch[1]+1, args.max_epoch[1]+args.max_epoch[2]+1):
        logger.info('Update Optimizer lr:{:.2e}'.format(scheduler.get_last_lr()[0]))
        train(train_loader, logger, model, iter_optimizer, loss, epoch, seghead, args.max_epoch[1]+args.max_epoch[2]) ### train
        scheduler.step()
        if epoch % 5 == 0:
            torch.save(model.state_dict(), join(args.save_path, 'svc', 'model_' + str(epoch) + '_checkpoint.pth'))
            torch.save(seghead.state_dict()['weight'], join(args.save_path, 'svc', 'cls_' + str(epoch) + '_checkpoint.pth'))
            with torch.no_grad():
                o_Acc, m_Acc, s = eval(epoch, args, mode='svc')
                logger.info('Iter--Eval Epoch{:02d}: oAcc {:.2f}  mAcc {:.2f} IoUs'.format(epoch, o_Acc, m_Acc) + s+'\n')
            ### Update pseudo labels
            if epoch != args.max_epoch[1]+args.max_epoch[2]+1:
                logger.info('Update pseudo labels')
                centroids_norm = init_cluster(args, logger, cluster_loader, model)
                seghead.weight.data = centroids_norm.requires_grad_(False)

    logger.info('====>End Super Voxel Clustering !!!\n')


def init_cluster(args, logger, cluster_loader, model, submodel=None):
    time_start = time.time()

    ## Extract Superpoints Feature
    sp_feats_list = init_get_sp_feature(args, cluster_loader, model, submodel)
    sp_feats = torch.cat(sp_feats_list, dim=0) ### will do Kmeans with l2 distance
    _, centroids_norm = faiss_cluster(args, sp_feats.cpu().numpy())
    centroids_norm = centroids_norm.cuda()

    ## Compute and Save Pseudo Labels
    all_pseudo, all_labels = init_get_pseudo(args, cluster_loader, model, centroids_norm, submodel)
    o_Acc, m_Acc, s = compute_seg_results(args, all_labels, all_pseudo)
    logger.info('clustering time: %.2fs', (time.time() - time_start))
    logger.info('Trainset: oAcc {:.2f}  mAcc {:.2f} IoUs'.format(o_Acc, m_Acc) + s+'\n')

    return centroids_norm

def distill(distill_loader, logger, model, submodel, optimizer, loss, epoch, maxepochs):
    distill_loader.dataset.mode = 'distill'
    model.train()
    submodel.train()
    loss_display = AverageMeter()

    trainloader_bar = tqdm(distill_loader)
    for batch_idx, data in enumerate(trainloader_bar):
        ## Prepare data
        trainloader_bar.set_description('Epoch {}'.format(epoch))
        coords, features, dinofeats, normals, labels, inverse_map, region, index, scenenames = data
        ## Forward
        in_field      = ME.TensorField(features, coords, device=0)
        feats         = model(in_field) 
        feats_aligned = submodel(feats)
        ## Loss
        mask = region.squeeze() >= 0
        loss_distill = loss(F.normalize(feats_aligned[mask]), F.normalize(dinofeats[mask].cuda().detach()))
        loss_display.update(loss_distill.item())
        optimizer.zero_grad()
        loss_distill.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))
        if batch_idx %5 == 0:
            trainloader_bar.set_postfix(trainloss='{:.3e}'.format(loss_display.avg))
    if epoch % 10 == 0:
        logger.info('Epoch: {}/{} Train loss: {:.3e}'.format(epoch, maxepochs, loss_display.avg))

def train(train_loader, logger, model, optimizer, loss, epoch, classifier, maxepochs):
    train_loader.dataset.mode = 'train'
    model.train()
    classifier.train()
    loss_display = AverageMeter()

    trainloader_bar = tqdm(train_loader)
    for batch_idx, data in enumerate(trainloader_bar):

        trainloader_bar.set_description('Epoch {}/{}'.format(epoch, maxepochs))
        coords, features, normals, labels, inverse_map, pseudo_labels, inds, region, index, scenenames = data

        in_field = ME.TensorField(features, coords, device=0)
        feats_nonorm = model(in_field)
        logits = classifier(feats_nonorm)

        ## loss
        pseudo_labels_comp = pseudo_labels.long().cuda()
        loss_sem = loss(logits, pseudo_labels_comp).mean()
        loss_display.update(loss_sem.item())
        optimizer.zero_grad()
        loss_sem.backward()
        optimizer.step()

        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))

        if batch_idx % 20 == 0:
            trainloader_bar.set_postfix(trainloss='{:.3e}'.format(loss_display.avg))

    logger.info('Epoch {}/{}: Train loss: {:.3e}'.format(epoch, maxepochs, loss_display.avg))

def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)

    return logger

if __name__ == '__main__':
    args = parse_args()
    
    args.save_path = os.path.join(args.save_path, args.expname)
    args.pseudo_path = os.path.join(args.save_path, 'pseudo_labels')

    '''Setup logger'''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        os.makedirs(args.pseudo_path)
        os.makedirs(join(args.save_path, 'cmd'))
        os.makedirs(join(args.save_path, 'svc'))
    logger = set_logger(os.path.join(args.save_path, 'train.log'))
    logger.info(args)

    '''Cache code'''
    cache_codes(args)

    '''Random Seed'''
    seed = args.seed
    set_seed(seed)

    main(args, logger)
