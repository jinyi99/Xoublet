# Xoublet

import scanpy as sc
from xoublet import Xoublet


adata = sc.read_h5ad("your_data.h5ad")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.pp.pca(adata)


xb = Xoublet(
    sim_doublet_ratio=1.0, 
    n_neighbors=25, 
    expected_doublet_rate=0.08
)

xb.run(adata)

sc.pl.embedding(adata, basis='umap', color='doublet_score')
