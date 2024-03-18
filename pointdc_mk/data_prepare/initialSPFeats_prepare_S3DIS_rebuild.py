import torch, os

import open3d as o3d
import numpy as np

from tqdm import tqdm
from glob import glob
from torch_scatter import scatter

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

spdir   = r'/home/bip/czs/codes/growsp_inf/data/S3DIS4growsp/initial_superpoints_rebuild'
sppaths = glob(os.path.join(spdir, '*rebuild_superpoint.npy'))

sppaths_bar = tqdm(sppaths)
for sppath in sppaths_bar:
    sppaths_bar.set_description(os.path.basename(sppath))
    savepath = sppath.replace('initial_superpoints_rebuild', 'input_spfeat_rebuild').replace('_superpoint.npy', '_sp.pt')
    # if os.path.exists(savepath): continue

    sp      = np.load(sppath)
    pcddict = torch.load(sppath.replace('initial_superpoints_rebuild', 'input_rebuild').replace('_superpoint.npy', '.pt'))
    coords, colors, feat, dis = pcddict['coord'].copy(), pcddict['color'].copy(), pcddict['feat'].copy(), pcddict['distance'].copy()

    sp       = torch.tensor(sp+1, dtype=torch.int64).cuda()
    feat     = torch.FloatTensor(feat).cuda()
    sp_feat  = scatter(feat, sp, dim=0, reduce='mean')

    point_spfeat       = sp_feat[sp]
    pcddict['spfeat'] = sp_feat.cpu().numpy()
    del pcddict['feat']
    
    torch.save(pcddict, savepath, pickle_protocol=4)