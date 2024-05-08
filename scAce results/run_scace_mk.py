import h5py
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt


from reproducibility.utils import data_sample, data_preprocess, set_seed, read_data
from scace import run_scace

####################################  Read dataset  ####################################

mat, obs, var, uns = read_data('/gpfs/work/bio/jiayiliu20/scAce-code/reproducibility/data/Mouse_k.h5', sparsify=False, skip_exprs=False)
x = np.array(mat.toarray())
cell_name = np.array(obs["cell_type1"])
cell_type, y = np.unique(cell_name, return_inverse=True)

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
    saved_ckpt='/gpfs/work/bio/jiayiliu20/scAce-code/pretraining/scace_mk.pth'
)


####################################  Show Final Clustering results  ####################################
sc.pp.neighbors(adata, use_rep='scace_emb')
sc.tl.umap(adata)
adata.obs['celltype'] = adata.obs['celltype'].astype(int).astype('category')
adata.obs['scace_cluster'] = adata.obs['scace_cluster'].astype(int).astype('category')
sc.pl.umap(adata, color=['scace_cluster', 'celltype'])
plt.savefig('/gpfs/work/bio/jiayiliu20/scAce-code/visualization/mk_final_clustering.svg', dpi=300, format='svg', bbox_inches='tight')

####################################  Show Initial Clustering results  ####################################
# emb_all[0] is the embedding after pre-training
# pred_all[0] is the initial clustering result after pre-training

adata_tmp = sc.AnnData(emb_all[0])
adata_tmp.obs['celltype'] = adata.obs['celltype']
adata_tmp.obs['scace_cluster'] = pred_all[0]
adata_tmp.obs['scace_cluster'] = adata_tmp.obs['scace_cluster'].astype(int).astype('category')

sc.pp.neighbors(adata_tmp)
sc.tl.umap(adata_tmp)
sc.pl.umap(adata_tmp, color=['scace_cluster', 'celltype'])
plt.savefig('/gpfs/work/bio/jiayiliu20/scAce-code/visualization/mk_initial_clustering.svg', dpi=300, format='svg', bbox_inches='tight')


####################################  Show Clustering results before and after cluster merging ####################################
# for the first time
# emb_all[1] is the embedding of the first cluster merging.
# pred_all[1] is the all clustering results from the first cluster merging, where pred_all[1][0] is the clustering
# result before cluster merging, and pred_all[1][0] is the clustering result after cluster merging.

adata_tmp = sc.AnnData(emb_all[1])
adata_tmp.obs['celltype'] = adata.obs['celltype']
adata_tmp.obs['scace_before'], adata_tmp.obs['scace_after'] = pred_all[1][0], pred_all[1][-1]
adata_tmp.obs['scace_before'] = adata_tmp.obs['scace_before'].astype(int).astype('category')
adata_tmp.obs['scace_after'] = adata_tmp.obs['scace_after'].astype(int).astype('category')

sc.pp.neighbors(adata_tmp)
sc.tl.umap(adata_tmp)
sc.pl.umap(adata_tmp, color=['scace_before', 'scace_after', 'celltype'])
plt.savefig('/gpfs/work/bio/jiayiliu20/scAce-code/visualization/mk_compare_initial.svg', dpi=300, format='svg', bbox_inches='tight')

# for the final time
adata_tmp = sc.AnnData(emb_all[-2])
adata_tmp.obs['celltype'] = adata.obs['celltype']
adata_tmp.obs['scace_before'], adata_tmp.obs['scace_after'] = pred_all[2][0], pred_all[2][-1]
adata_tmp.obs['scace_before'] = adata_tmp.obs['scace_before'].astype(int).astype('category')
adata_tmp.obs['scace_after'] = adata_tmp.obs['scace_after'].astype(int).astype('category')

sc.pp.neighbors(adata_tmp)
sc.tl.umap(adata_tmp)
sc.pl.umap(adata_tmp, color=['scace_before', 'scace_after', 'celltype'])
plt.savefig('/gpfs/work/bio/jiayiliu20/scAce-code/visualization/mk_compare_final.svg', dpi=300, format='svg', bbox_inches='tight')

