import h5py
import numpy as np
import scanpy as sc
import torch
import random
import matplotlib.pyplot as plt

from reproducibility.utils import data_sample, data_preprocess, set_seed
from scace import run_scace

# ##################################  Set Seed  ####################################
# seed = 2023
# np.random.seed(seed)
# random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
#
# ####################################  Read dataset  ####################################
# data_mat = h5py.File('/gpfs/work/bio/jiayiliu20/scAce-code/reproducibility/data/Human_k.h5')
# x, y = np.array(data_mat['X']), np.array(data_mat['Y'])
# data_mat.close()
#
# adata = sc.AnnData(x)
# adata.obs['celltype'] = y
#
# ##################################  Preform Data Preprocessing ####################################
# sc.pp.filter_genes(adata, min_cells=3)
# sc.pp.filter_cells(adata, min_genes=200)
# adata.raw = adata.copy()
#
# sc.pp.normalize_per_cell(adata)
# adata.obs['scale_factor'] = adata.obs.n_counts / adata.obs.n_counts.mean()
#
# sc.pp.log1p(adata)
# sc.pp.scale(adata)

####################################  Read dataset  ####################################

data_mat = h5py.File('/gpfs/work/bio/jiayiliu20/scAce-code/reproducibility/data/Human_k.h5')
x, y = np.array(data_mat['X']), np.array(data_mat['Y'])
data_mat.close()

####################################  Run without sampling  ####################################

seed = 2023
set_seed(seed)

adata = sc.AnnData(x)
adata.obs['celltype'] = y

adata = data_preprocess(adata)

adata, nmi, ari, K, pred_all, emb_all, run_time = run_scace(
    adata,
    cl_type='celltype',
    return_all=True,
    save_pretrain=True,
    saved_ckpt='/gpfs/work/bio/jiayiliu20/scAce-code/pretraining/scace_hk_2.pth'
)

#print("ARI shape:", np.array(ari).shape)
#print("NMI shape:", np.array(nmi).shape)
#print("K shape:", np.array(K).shape)
#print("Embedding shapes:", [emb.shape for emb in emb_all])
#print("ARI = {}, NMI = {}".format(ari, nmi))

#np.savez("/gpfs/work/bio/jiayiliu20/scAce-code/results/scAce_wo_sample_hk.npz", ARI=ari, NMI=nmi, K=K, Embedding=emb_all,
#         Clusters=pred_all, Labels=y, Time=run_time)

# 创建一个字典来保存所有数据
# data_to_save = {
#     "ARI": np.array([ari]),
#     "NMI": np.array([nmi]),
#     "K": np.array([K]),
#     "Time": np.array([run_time]),
#     "Labels": y
# }
#
# # 添加 emb_all 和 pred_all 中的每个数组到字典
# for i, emb in enumerate(emb_all):
#     data_to_save[f"Embedding_{i}"] = emb
# for i, cluster in enumerate(pred_all):
#     data_to_save[f"Clusters_{i}"] = cluster
#
# # 保存数据到一个 .npz 文件
# np.savez("/gpfs/work/bio/jiayiliu20/scAce-code/results/scAce_wo_sample_hk.npz", **data_to_save)

#
# ####################################  Show Final Clustering results  ####################################
# sc.pp.neighbors(adata, use_rep='scace_emb')
# sc.tl.umap(adata)
# adata.obs['celltype'] = adata.obs['celltype'].astype(int).astype('category')
# adata.obs['scace_cluster'] = adata.obs['scace_cluster'].astype(int).astype('category')
# sc.pl.umap(adata, color=['scace_cluster', 'celltype'])
# plt.savefig('/gpfs/work/bio/jiayiliu20/scAce-code/visualization/hk_final_clustering.png')
#
# ####################################  Show Initial Clustering results  ####################################
# # emb_all[0] is the embedding after pre-training
# # pred_all[0] is the initial clustering result after pre-training
#
# adata_tmp = sc.AnnData(emb_all[0])
# adata_tmp.obs['celltype'] = adata.obs['celltype']
# adata_tmp.obs['scace_cluster'] = pred_all[0]
# adata_tmp.obs['scace_cluster'] = adata_tmp.obs['scace_cluster'].astype(int).astype('category')
#
# sc.pp.neighbors(adata_tmp)
# sc.tl.umap(adata_tmp)
# sc.pl.umap(adata_tmp, color=['scace_cluster', 'celltype'])
# plt.savefig('/gpfs/work/bio/jiayiliu20/scAce-code/visualization/hk_initial_clustering.png')
#
#
# ####################################  Show Clustering results before and after cluster merging ####################################
# # for the first time
# # emb_all[1] is the embedding of the first cluster merging.
# # pred_all[1] is the all clustering results from the first cluster merging, where pred_all[1][0] is the clustering
# # result before cluster merging, and pred_all[1][0] is the clustering result after cluster merging.
#
# adata_tmp = sc.AnnData(emb_all[1])
# adata_tmp.obs['celltype'] = adata.obs['celltype']
# adata_tmp.obs['scace_before'], adata_tmp.obs['scace_after'] = pred_all[1][0], pred_all[1][-1]
# adata_tmp.obs['scace_before'] = adata_tmp.obs['scace_before'].astype(int).astype('category')
# adata_tmp.obs['scace_after'] = adata_tmp.obs['scace_after'].astype(int).astype('category')
#
# sc.pp.neighbors(adata_tmp)
# sc.tl.umap(adata_tmp)
# sc.pl.umap(adata_tmp, color=['scace_before', 'scace_after', 'celltype'])
# plt.savefig('/gpfs/work/bio/jiayiliu20/scAce-code/visualization/hk_compare_initial.png')
#
# # for the final time
# adata_tmp = sc.AnnData(emb_all[-2])
# adata_tmp.obs['celltype'] = adata.obs['celltype']
# adata_tmp.obs['scace_before'], adata_tmp.obs['scace_after'] = pred_all[2][0], pred_all[2][-1]
# adata_tmp.obs['scace_before'] = adata_tmp.obs['scace_before'].astype(int).astype('category')
# adata_tmp.obs['scace_after'] = adata_tmp.obs['scace_after'].astype(int).astype('category')
#
# sc.pp.neighbors(adata_tmp)
# sc.tl.umap(adata_tmp)
# sc.pl.umap(adata_tmp, color=['scace_before', 'scace_after', 'celltype'])
# plt.savefig('/gpfs/work/bio/jiayiliu20/scAce-code/visualization/hk_compare_final.png')
