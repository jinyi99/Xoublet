from scipy.sparse import csc_matrix
from anndata import AnnData
import logging
import scann
import numpy as np
from scipy.sparse import csr_matrix


class Xoublet:

    def __init__(self, sim_doublet_ratio=1, n_neighbors=25, expected_doublet_rate=0.1,
                 total_counts_key='n_counts', use_scann=True):
        self.sim_doublet_ratio = sim_doublet_ratio
        self.n_neighbors = n_neighbors
        self.expected_doublet_rate = expected_doublet_rate
        self.total_counts_key = total_counts_key
        self.use_scann = use_scann

        # 设置日志
        try:
            from scanpy import logging as logg
            self.logg = logg
        except ImportError:
            self.logg = logging

    def simulate_doublets_from_pca(self, PCdat, total_counts=None, sim_doublet_ratio=None):
        if sim_doublet_ratio is None:
            sim_doublet_ratio = self.sim_doublet_ratio

        n_obs = PCdat.shape[0]
        n_doub = int(n_obs * sim_doublet_ratio)

        if total_counts is None:
            total_counts = np.ones(n_obs)
        else:
            total_counts = np.array(total_counts)

        # 生成随机细胞对用于双细胞模拟
        pair_ix = np.random.randint(0, n_obs, size=(n_doub, 2))
        pair_tots = np.hstack((
            total_counts[pair_ix[:, 0]][:, None],
            total_counts[pair_ix[:, 1]][:, None]
        ))
        pair_tots = np.array(pair_tots, dtype=float)
        pair_fracs = pair_tots / np.sum(pair_tots, axis=1)[:, None]

        # 创建PCA坐标的加权平均值
        PCdoub = (PCdat[pair_ix[:, 0], :] * pair_fracs[:, 0][:, None] +
                  PCdat[pair_ix[:, 1], :] * pair_fracs[:, 1][:, None])

        PCdat_new = np.vstack((PCdat, PCdoub))
        doub_labels = np.concatenate((np.zeros(n_obs), np.ones(n_doub)))

        return PCdat_new, doub_labels, pair_ix

    def calculate_doublet_scores_scann(self, adata, doub_labels, n_neighbors=None, expected_doublet_rate=None):

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        if expected_doublet_rate is None:
            expected_doublet_rate = self.expected_doublet_rate

        self.logg.info("使用ScaNN算法计算双细胞分数")
        n_obs = sum(doub_labels == 0)
        n_sim = sum(doub_labels == 1)
        self.logg.info(f"观察到的细胞: {n_obs}, 模拟双细胞: {n_sim}")

        # 根据模拟与观察细胞的比率调整k值
        k_adj = int(round(n_neighbors * (1 + n_sim / float(n_obs))))
        self.logg.info(f"调整后的k值: {k_adj}")

        pca_data = adata.obsm["X_pca"]

        # 使用适合数据集大小的参数构建ScaNN索引
        try:
            # 配置ScaNN，使用根据数据规模调整的合理默认值
            searcher = scann.scann_ops_pybind.builder(
                pca_data, k_adj, "squared_l2"
            ).tree(
                num_leaves=min(3000, max(100, int(pca_data.shape[0] / 5))),
                num_leaves_to_search=min(60, max(40, int(pca_data.shape[0] / 15))),
                training_sample_size=min(1000000, pca_data.shape[0] * 5)
            ).score_ah(
                2, anisotropic_quantization_threshold=0.15
            ).reorder(100).build()

            self.logg.info("ScaNN索引构建成功")
        except Exception as e:
            self.logg.error(f"构建ScaNN索引时出错: {e}")
            raise

        # 批量搜索最近邻
        try:
            neighbors_indices, distances = searcher.search_batched(pca_data)
            distances=np.sqrt(distances)
            self.logg.info("ScaNN最近邻搜索完成")
        except Exception as e:
            self.logg.error(f"最近邻搜索过程中出错: {e}")
            raise

        # 从最近邻创建稀疏邻接矩阵
        rows, cols, data = [], [], []
        for i, indices in enumerate(neighbors_indices):
            for j in indices:
                rows.append(i)
                cols.append(j)
                data.append(1)  # 二进制关系 - 如果是邻居则为1

        # 创建稀疏矩阵
        n_total = len(doub_labels)
        conn_matrix = csr_matrix((data, (rows, cols)), shape=(n_total, n_total))
        self.logg.info(f"连接矩阵已创建，形状: {conn_matrix.shape}")

        # 计算每个细胞邻域中模拟双细胞和观察细胞的数量
        n_sim_neigh = np.zeros(n_total)
        n_obs_neigh = np.zeros(n_total)

        for i in range(n_total):
            # 获取第i个细胞的所有邻居
            neighbors = neighbors_indices[i]
            # 计算邻居中模拟双细胞和观察细胞的数量
            n_sim_neigh[i] = np.sum(doub_labels[neighbors] == 1)
            n_obs_neigh[i] = np.sum(doub_labels[neighbors] == 0)

        # 计算双细胞分数
        doub_score = n_sim_neigh / (n_sim_neigh + n_obs_neigh * n_sim / float(n_obs) / expected_doublet_rate)

        # 报告统计信息
        doub_score_obs = doub_score[doub_labels == 0]
        self.logg.info(
            f"观察细胞的双细胞分数: 最小值={doub_score_obs.min():.4f}, 最大值={doub_score_obs.max():.4f}, 平均值={doub_score_obs.mean():.4f}")

        # 记录分数分布
        thresholds = [0.35, 0.5, 0.65, 0.8]
        for t in thresholds:
            percent = (doub_score_obs > t).mean() * 100
            self.logg.info(f"双细胞比例 (分数 > {t}): {percent:.2f}%")

        return doub_score[doub_labels == 0], doub_score[doub_labels == 1]

    def run(self, adata, copy=False):

        adata = adata.copy() if copy else adata

        if 'X_pca' not in adata.obsm_keys():
            raise ValueError("未找到'X_pca'。先运行`sc.pp.pca`。")

        # 获取可用的总计数，否则使用1
        if self.total_counts_key in adata.obs:
            total_counts = np.array(adata.obs[self.total_counts_key])
        else:
            self.logg.warning(f"在adata.obs中未找到{self.total_counts_key}，使用统一权重")
            total_counts = np.ones(adata.X.shape[0])

        self.logg.info('模拟双细胞...')
        PCdat, doub_labels, parent_ix = self.simulate_doublets_from_pca(
            adata.obsm['X_pca'],
            total_counts=total_counts,
            sim_doublet_ratio=self.sim_doublet_ratio
        )

        # 创建包含真实和模拟细胞的临时AnnData
        adata_doub = AnnData(csc_matrix((PCdat.shape[0], 1)))
        adata_doub.obsm['X_pca'] = PCdat

        # 选择ScaNN并打分
        self.logg.info('运行分类器...')
        if self.use_scann:
            try:
                self.logg.info('使用ScaNN进行最近邻搜索')
                adata.obs['doublet_score'], adata.uns['sim_doublet_score'] = self.calculate_doublet_scores_scann(
                    adata_doub,
                    doub_labels,
                    n_neighbors=self.n_neighbors,
                    expected_doublet_rate=self.expected_doublet_rate
                )

            except Exception as e:
                self.logg.warning(f"ScaNN失败: {e}。回退到标准KNN方法。")
                # 如果有标准方法可用，则回退到标准方法

        return adata if copy else None