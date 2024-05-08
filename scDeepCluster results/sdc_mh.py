import h5py
import numpy as np
import scanpy as sc
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import math
import os
from sklearn import metrics
from single_cell_tools import *

from scDeepCluster import scDeepCluster

from preprocess import read_dataset, normalize
from reproducibility.utils import data_sample, data_preprocess, set_seed, calculate_metric, read_data

####################################  Read dataset  ####################################
mat, obs, var, uns = read_data('/gpfs/work/bio/jiayiliu20/scAce-code/reproducibility/data/Mouse_h.h5', sparsify=False, skip_exprs=False)
x = np.array(mat.toarray())
cell_name = np.array(obs["cell_type1"])
cell_type, y = np.unique(cell_name, return_inverse=True)

####################################  Set parameters  ####################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'

####################################  Run without sampling  ####################################
seed = 0
set_seed(seed)

adata = sc.AnnData(x)
adata.obs['celltype'] = y

adata = data_preprocess(adata, use_count=True)
print(adata)

model = scDeepCluster(input_dim=adata.n_vars, z_dim=32,
                      encodeLayer=[256, 64], decodeLayer=[64, 256],
                      device=device)
start = time.time()

model.pretrain_autoencoder(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.scale_factor)


class Args(object):
    def __init__(self):
        self.n_clusters = 0
        self.knn = 20
        self.resolution = 0.8
        self.batch_size = 256
        self.maxiter = 2000
        self.pretrain_epochs = 300
        self.gamma = 1.
        self.sigma = 2.5
        self.update_interval = 1
        self.tol = 0.001
        self.ae_weights = None

args = Args()

### estimate number of clusters by Louvain algorithm on the autoencoder latent representations
x_tensor = torch.tensor(adata.X, dtype=torch.float32)
pretrain_latent = model.encodeBatch(x_tensor).cpu().numpy()
adata_latent = sc.AnnData(pretrain_latent)
sc.pp.neighbors(adata_latent, n_neighbors=args.knn, use_rep="X")
sc.tl.louvain(adata_latent, resolution=args.resolution)
y_pred_init = np.asarray(adata_latent.obs['louvain'],dtype=int)
features = pd.DataFrame(adata_latent.X,index=np.arange(0,adata_latent.n_obs))
Group = pd.Series(y_pred_init,index=np.arange(0,adata_latent.n_obs),name="Group")
Mergefeature = pd.concat([features,Group],axis=1)
cluster_centers = np.asarray(Mergefeature.groupby("Group").mean())
n_clusters = cluster_centers.shape[0]
print('Estimated number of clusters: ', n_clusters)
y_pred, _, _, _, _ = model.fit(X=adata.X, X_raw=adata.raw.X, size_factor=adata.obs.scale_factor, n_clusters=n_clusters, init_centroid=cluster_centers,
                               y_pred_init=y_pred_init, y=y, batch_size=args.batch_size, num_epochs=args.maxiter, update_interval=args.update_interval, tol=args.tol)

end = time.time()
run_time = end - start
print(f'Total time: {end - start} seconds')

nmi, ari = calculate_metric(y, y_pred)
print('Evaluating cells: NMI= %.4f, ARI= %.4f' % (nmi, ari))

# Assuming `embedded` is meant to be `pretrain_latent` or a similar meaningful output
embedded = pretrain_latent

# np.savez("/gpfs/work/bio/jiayiliu20/scAce-code/scDeepCluster/results_new/hpbmc_scDeepcluster_wo_sample.npz", ARI=ari, NMI=nmi, Embedding=embedded,
#          Clusters=y_pred, Labels=y, Time_use=run_time)
np.savez("/gpfs/work/bio/jiayiliu20/scAce-code/scDeepCluster/results_new/mh_scDeepcluster_wo_sample.npz", ARI=ari, NMI=nmi,
         Clusters=y_pred, Labels=y, Time_use=run_time)

####################################  Plot!!!!!!  ####################################
from umap import UMAP
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from matplotlib.pyplot import plot,savefig
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

scd = np.load("/gpfs/work/bio/jiayiliu20/scAce-code/scDeepCluster/results_new/mh_scDeepcluster_wo_sample.npz")
y_pred = scd["Clusters"]
y = scd["Labels"]
print(adata)
print(y_pred)

# Adjust labels to start from 0
y = y - 1  # Subtract 1 from each element in the Labels array
print(y)

# Evaluate clustering performance
if y is not None:
    nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
    ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
    print('Evaluating cells: NMI= %.4f, ARI= %.4f' % (nmi, ari))

# Encode data and compute UMAP
x_tensor = torch.tensor(adata.X, dtype=torch.float32)
final_latent = model.encodeBatch(x_tensor).cpu().numpy()

# Store clustering results and labels in AnnData object
adata.obs['scDeepCluster_cluster'] = y_pred.astype(str)
if y is not None:
    adata.obs['Labels'] = y.astype(str)

# Create AnnData
adata = sc.AnnData(X=final_latent)  # Assuming final_latent is ready and correct
adata.obs['Labels'] = y.astype(str)
adata.obs['scDeepCluster_cluster'] = y_pred.astype(str)

print(adata)

sc.pp.neighbors(adata, use_rep='X')
sc.tl.umap(adata, random_state=0)

adata.obs['scDeepCluster_cluster'] = adata.obs['scDeepCluster_cluster'].astype('category')
adata.obs['Labels'] = adata.obs['Labels'].astype('category')

import os
output_dir = '/gpfs/work/bio/jiayiliu20/scAce-code/scDeepCluster/results_new/'
filename = 'mh_plot.png'
full_path = os.path.join(output_dir, filename)

os.makedirs(output_dir, exist_ok=True)

sc.pl.umap(adata, color=['scDeepCluster_cluster', 'Labels'],show=False)
plt.savefig(full_path)