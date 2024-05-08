import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.cluster import KMeans

from ..model import centroid_split
from ..model.scace_net import target_distribution

"""
tools.py 给出了一些用于实施聚类以及评估聚类结果的函数
"""

# compute_mu 返回每个聚类的中心
def compute_mu(scace_emb,
               pred):  # scace_emb是样本的embedding表示(二维数组 e.g., 1000个cell嵌入到dim=20的空间 那么scace_emb就是(1000, 20)的数组)
    #                                  pred是每个样本的聚类标签 (一维数组 长度等于样本数量)
    mu = []
    for idx in np.unique(pred):  # 遍历找出的所有的聚类标签
        mu.append(scace_emb[idx == pred, :].mean(axis=0))  # idx==pred 是布尔索引 找出所有聚类标签等于index的样本
        #                                                   mu.append(...) 最后将聚类中心添加到mu列表中

    return np.array(mu)


# cluster_acc 计算聚类准确率, 即预测的聚类标签和真实的聚类标签匹配的样本比例
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)  # 将y_true转换为int64类型
    y_pred = pd.Series(data=y_pred)  # 将y_pred转换为Pandas的Series对象

    assert y_pred.size == y_true.size  # 检查y_pred和y_true的大小是否一致
    D = max(y_pred.max(), y_true.max()) + 1  # 计算D 是y_pred和y_true中的最大标签值加1 (即预测标签包括0 1 2, 那么D=3, 构建出的w=3*3 符合逻辑)
    D = int(D)
    w = np.zeros((D, D), dtype=np.int64)  # 构建一个D*D的零矩阵w

    """
    w用于记录 预测的聚类标签y_pred和真实的聚类标签y_true之间的匹配情况
    w[i, j]的值表示: 预测为i且真实标签为j的样本数量
    
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [1, 1, 0, 0, 2, 2] 在这个例子中标签最大=2 所以首先创建w=3*3的零矩阵
    遍历所有的y_pred: 对于第一个样本, y_pred[0]=1, y_true[0]=0, 所以w[1, 0] += 1
                    对于第二个样本, y_pred[1]=1, y_true[1]=0, 所以w[1, 0] += 1
                    对于第三个样本, y_pred[2]=0, y_true[2]=1, 所以w[0, 1] += 1
                    .....
                    对于第六个样本, y_pred[5]=2, y_true[5]=2, 所以w[2, 2] += 1
    最终得到混淆矩阵 w = [[0, 2, 0],
                       [2, 0, 0],
                       [0, 0, 2]] 记录了预测标签和真实标签之间的匹配情况
    """
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    """
    cost = np.array([[4, 1, 3], 
                     [2, 0, 5], 
                     [3, 2, 2]])
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    可以得到 row_ind = [0, 1, 2]
            col_ind = [1, 0, 2]   
    这表示最优的匹配是: 第0行和第1列, 第1行和第0列, 第2行和第2列。这个匹配的总成本是cost[0, 1] + cost[1, 0] + cost[2, 2] 

    这里, 把 w.max() - w 代入到linear_sum_assignment()来计算最优的匹配, 因为linear_sum_assignment()是用来计算最小的匹配成本的, 所以我们需要将w中的每个元素取反, 这样最小的匹配就变成了最大的匹配
    """
    row_ind, col_ind = linear_sum_assignment(w.max() - w)  # w.max()是w矩阵中的最大值
    # w.max()-w 就是用w,max()的值减去w中的每一个元素
    # linear_sum_assignment()是Scipy中的函数 解决线性和分配的问题, 即在一个矩阵中找到一组元素，使得这些元素的和最小，并且每个行和列只能选择一个元素
    # 输出两个一维数组row_ind 和 col_ind

    """
    将匹配的样本数量除以预测标签的总数量, 得到预测的聚类标签和真实的聚类标签匹配的样本比例 (聚类准确率)
    注: * 1.0是将结果转换为浮点数, 以便进行浮点数除法
    """
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size
    # 注: 这种方法的一个假设是，每个预测的聚类都对应一个真实的聚类，且这种对应关系是一对一的
    # 这在很多聚类问题中都是合理的，但在某些情况下可能不适用


# clustering 用于聚类, 输入包括model、表达矩阵exp_mat，以及一些可选的参数，如初始聚类init_cluster、初始方法init_method 和分辨率resolution
def clustering(model, exp_mat, init_cluster=None, init_method=None, resolution=None):
    """
    PyTorch中, model.eval()和model.train()是用来切换模型的模式的
    在训练模型之前, 应该调用model.train(); 在评估模型之前, 应该调用model.eval()
    """
    model.eval()  # 将模型设置为评估模式
    scace_emb = model.EncodeAll(exp_mat)  # 将表达矩阵exp_mat送入模型的Encoder网络, 得到样本的embedding表示

    model.train()  # 将模型设置为训练模式

    if init_method == 'kmeans':
        scace_emb = scace_emb.cpu().numpy()  # 将scace_emb转移到CPU并转换为numpy数组
        max_score = -1
        k_init = 0
        for k in range(15, 31):

            kmeans = KMeans(k, n_init=50)
            y_pred = kmeans.fit_predict(scace_emb)
            s_score = metrics.silhouette_score(scace_emb, y_pred)
            if s_score > max_score:
                max_score = s_score
                k_init = k

        kmeans = KMeans(k_init, n_init=50)
        y_pred = kmeans.fit_predict(scace_emb)
        mu = kmeans.cluster_centers_
        return y_pred, mu, scace_emb


    elif init_method == 'leiden':
        adata_l = sc.AnnData(scace_emb.cpu().numpy())
        sc.pp.neighbors(adata_l, n_neighbors=10)
        sc.tl.leiden(adata_l, resolution=resolution, random_state=0)
        y_pred = np.asarray(adata_l.obs['leiden'], dtype=int)
        mu = compute_mu(scace_emb.cpu().numpy(), y_pred)

        return y_pred, mu, scace_emb.cpu().numpy()


    elif init_method == 'louvain':
        adata_l = sc.AnnData(scace_emb.cpu().numpy())
        sc.pp.neighbors(adata_l, n_neighbors=10)
        sc.tl.louvain(adata_l, resolution=resolution, random_state=0)
        y_pred = np.asarray(adata_l.obs['louvain'], dtype=int)
        mu = compute_mu(scace_emb.cpu().numpy(), y_pred)

        return y_pred, mu, scace_emb.cpu().numpy()

    # 如果不是指明用前面的三种聚类 也不是none, 即给定初始聚类
    if init_cluster is not None:
        cluster_centers = compute_mu(scace_emb.cpu().numpy(), init_cluster)  # 计算初始聚类的中心

        """
        将嵌入表示scace_emb (二维数组 每一行代表一个数据点的嵌入表示，每一列代表一个特征)
        和 初始聚类init_cluster(一维数组 每个数据点代表一个元素的聚类标签) 
        沿着列轴(axis=1)合并为一个数据集data_1, 也就是说, init_cluster的每个元素被添加到scace_emb的每一行的末尾
        """
        data_1 = np.concatenate([scace_emb.cpu().numpy(), np.array(init_cluster).reshape(-1, 1)], axis=1)

        mu, y_pred = centroid_split(scace_emb.cpu().numpy(), data_1, cluster_centers, np.array(init_cluster))

        return y_pred, mu, scace_emb.cpu().numpy()

    # Deep Embedded Clustering, DEC (最后一种可能 如果init_cluster=None)
    """
    DEC是一种基于深度学习的聚类方法, 它首先使用一个深度神经网络将数据映射到一个低维的嵌入空间, 然后在这个嵌入空间中进行聚类
    """
    q = model.soft_assign(scace_emb)
    p = target_distribution(q)

    y_pred = torch.argmax(q, dim=1).cpu().numpy()

    return y_pred, scace_emb.cpu().numpy(), q, p


def calculate_metric(pred, label):
    # 调用cluster_acc计算聚类准确率，并将结果四舍五入到小数点后5位, 得到聚类准确率acc
    # acc = np.round(cluster_acc(label, pred), 5) 
    """
    calculate_metric 关注的是 预测标签和真实标签之间的相似性, 计算归一化互信息(NMI) 和 调整兰德指数(ARI)
    nmi: 衡量两个标签分配(这里是pred和label)之间相似性的度量, 值范围为0到1, 1表示完全相同
    ari: 也是衡量两个标签分配之间相似性的度量, 值范围为-1到1, 1表示完全相同, 0表示随机分配, 负值表示比随机分配还差
    """
    nmi = np.round(metrics.normalized_mutual_info_score(label, pred), 5)
    ari = np.round(metrics.adjusted_rand_score(label, pred), 5)

    return nmi, ari
