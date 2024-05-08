import numpy as np


# cluster_intra_dis 返回一个列表，包含每个簇内的数据点到簇中心的平均距离
def cluster_intra_dis(X, cent, label):  # cent参数是一个数组，包含了每个簇的中心点, cent[i]是第i个簇的中心，它是一个向量，表示了簇中心在多维空间中的“位置”
#                                         label参数是一个与输入数据X同样长度的数组，其中的每个元素表示对应的数据点属于哪个簇, 如label[i]=j，表示第i个数据点属于第j个簇
    
    intra_dis = [] # 初始化一个空列表，用于存储每个簇的平均距离
    for i in range(len(cent)): # 对每个簇进行遍历
        data_cluster = X[label == i, :] # 取出属于当前簇的所有样本

        cluster_dis = np.linalg.norm(data_cluster - cent[i], axis=0) # 计算当前簇中每个样本与簇中心的距离
        intra_dis.append(np.sum(cluster_dis) / data_cluster.shape[0]) # 计算当前簇的平均距离并添加到列表中

    return intra_dis


def merge_compute(y_pred, mu, scace_emb): # 接受预测的簇标签y_pred、簇中心mu 和嵌入数据 scace_emb作为输入
    # Check if len(mu_prepare) == len(np.unique(y_pred)) (检查是否有空的 即没有数据点的簇)
    idx_cent = [] # 初始化一个空列表，用于存储没有数据点的簇的索引
    for i in range(len(mu)): # 对每个簇进行遍历
        if (y_pred == i).any() == False: # 如果当前簇没有数据点
            idx_cent.append(i) # 将当前簇的索引添加到列表中
    if len(idx_cent) == 0:  # 如果所有的簇都有数据点
        mu = mu # 簇中心不变
    else: # 否则
        mu = np.delete(mu, idx_cent, 0) # 删除没有数据点的簇的中心
    n_clusters = len(mu) # 计算合并后的簇的数量

    # Change label to 0 ~ len(np.unique(label))
    # 例如，如果我们有一个标签数组[0, 1, 2, 4, 5]，这段代码将其转换为[0, 1, 2, 3, 4]
    for i in range(len(np.unique(y_pred))):
        if np.unique(y_pred)[i] != i:
            y_pred[y_pred == np.unique(y_pred)[i]] = i

    Centroid = np.array(mu)  # 将mu转换为numpy数组并将其赋值给Centroid

    # Compute d_bar (加权平均距离)
    intra_dis = cluster_intra_dis(scace_emb, Centroid, y_pred) # 调用cluster_intra_dis每个元素表示一个簇内所有点到簇中心的平均距离
    #                                                            接收三个参数：scace_emb(数据点的嵌入向量), Centroid(每个簇的中心点), 以及y_pred(每个数据点的预测簇标签)
    d_ave = np.mean(intra_dis) # 计算各个簇内平均距离的平均值, 结果储存在d_ave中

    sum_1 = 0
    for i in range(0, n_clusters): # 遍历所有的簇对
        for j in range(i + 1, n_clusters): # i和j, 其中j大于i以避免重复计算和自我比较
            weight = d_ave / ((intra_dis[i] + intra_dis[j]) / 2) # 对于每一对簇, 计算一个权重, 该权重等于平均簇内距离d_ave除以当前簇对的平均簇内距离
            sum_1 += weight * np.linalg.norm(Centroid[i] - Centroid[j]) # 计算加权平均距离(每个簇对的欧式距离*weight的总和 加到sum_1上)

    d_bar = sum_1 / (n_clusters * (n_clusters - 1)) # 计算d_bar，即所有簇对的加权距离之和除以簇对的总数

    return y_pred, Centroid, d_bar, intra_dis, d_ave # 返回预测的簇标签、簇中心、加权平均距离、簇内距离和簇内距离的平均值


"""
centroid_merge 通过合并距离过近的簇中心来优化簇的数量和质量

输入: X: 数据点
cent_to_merge: 需要合并的簇中心
label: 每个数据点的簇标签
min_dis: 簇中心之间的最小距离, 如果两个簇中心的距离小于这个值, 它们就会被合并, 
         初始值并没有在这个函数中设定, 可能在函数被调用的地方
intra_dis: 每个簇的内部距离
d_ave: 簇内距离的平均值

返回: 最终的簇中心Final_Centroid_merge, 簇标签label、簇的数量n_clusters_t, 以及所有的簇标签历史记录pred_f
"""
def centroid_merge(X, cent_to_merge, label, min_dis, intra_dis, d_ave):  
    # 初始化一个空列表，用于存储每次迭代后的簇标签
    pred_f = [] 

    for t in range(200): # 迭代200次

        if t == 0: # 如果是第一次迭代
            pred_f.append(label) # 将初始的簇标签添加到列表中

        Cent_dis = [] # 建立空列表 存储每一对簇中心之间的加权距离
        Cent_i = [] # 存储计算距离的簇对中的第一个簇的索引
        Cent_e = [] # 存储计算距离的簇对中的第二个簇的索引
        Final_Centroid_merge = cent_to_merge # 初始化Final_Centroid_merge为需要合并的簇中心

        n_Clusters = len(Final_Centroid_merge) # 计算当前的簇数量

        # 对于每一对簇(i和e，其中e大于i以避免重复计算和自我比较), 计算它们的加权距离, 并将其和对应的簇索引添加到Cent_dis、Cent_i和Cent_e
        for i in range(n_Clusters):
            for e in range(i + 1, n_Clusters):
                weight = d_ave / ((intra_dis[i] + intra_dis[e]) / 2)
                dis = np.linalg.norm(Final_Centroid_merge[i] - Final_Centroid_merge[e])
                Cent_dis.append(weight * dis)
                Cent_i.append(i)
                Cent_e.append(e)

        for i in range(len(Cent_dis)): # 在这里，i会依次取值为0, 1, 2, ..., len(Cent_dis)-1
            # 检查当前的加权距离是否小于min_dis并且是否是Cent_dis中的最小值n (如果是，那么就执行以下的代码块，否则就跳过)
            if Cent_dis[i] < min_dis and Cent_dis[i] == min(Cent_dis):
            
                Cent_merge = (Final_Centroid_merge[Cent_i[i]] + Final_Centroid_merge[Cent_e[i]]) / 2 # 计算需要合并的两个簇中心的平均值作为新的簇中心

                Final_Centroid_merge = np.delete(Final_Centroid_merge, (Cent_i[i], Cent_e[i]), 0) # 从Final_Centroid_merge中删除了要合并的两个簇中心
                Final_Centroid_merge = np.insert(Final_Centroid_merge, Cent_i[i], Cent_merge, 0) # 在Final_Centroid_merge的Cent_i[i]位置插入了新的簇中心
                Final_Centroid_merge = np.insert(Final_Centroid_merge, Cent_e[i], 0, 0) # 在Final_Centroid_merge的Cent_e[i]位置插入了0

                Final_Centroid_merge = Final_Centroid_merge[~(Final_Centroid_merge == 0).all(axis=1)] # 删除了Final_Centroid_merge中所有为0的行

                cent_to_merge = Final_Centroid_merge # 更新了cent_to_merge

                label = np.array(label) # 将label转换为numpy数组
                label[label == Cent_e[i]] = Cent_i[i] # 将label中所有等于Cent_e[i]的元素替换为Cent_i[i], 即将被合并的簇的标签更新为新的簇的标签

                for i in range(len(np.unique(label))): # 遍历label中的每个唯一值
                    if np.unique(label)[i] != i: # 检查当前的唯一值是否等于其索引 (如果不等于, 那么就执行以下的代码块, 否则就跳过)
                        label[label == np.unique(label)[i]] = i
                    else:
                        continue

            else:
                pass

        n_clusters_t = len(np.unique(label))
        pred_f.append(label)

        intra_dis = cluster_intra_dis(X, cent_to_merge, label)
        d_ave = np.mean(intra_dis)

        sum_1 = 0
        for i in range(n_clusters_t):
            for j in range(i + 1, n_clusters_t):
                weight = d_ave / ((intra_dis[i] + intra_dis[j]) / 2)

                sum_1 += weight * (np.linalg.norm(Final_Centroid_merge[i] - Final_Centroid_merge[j]))

        d_bar = sum_1 / (n_clusters_t * (n_clusters_t - 1))


        """
        动态调整阈值来决定是否合并簇: (为什么在进行一遍合并之后又把min_dis设定为d_bar进行第二遍合并)
        因为d_bar在每次迭代中都会被重新计算, 代表了当前所有簇之间的平均距离. 将min_dis设定为d_bar意味着在下一次迭代中, 只有当两个簇之间的距离小于所有簇之间的平均距离时, 这两个簇才会被合并
        然后, 如果大于d_bar的簇对的数量count>=所有可能的簇对数量count_true(=int((n_clusters_t ** 2 - n_clusters_t) / 2)) 则停止迭代
        """
        min_dis = d_bar

        count = 0
        for i in range(0, n_clusters_t):
            for j in range(i + 1, n_clusters_t):
                weight = d_ave / ((intra_dis[i] + intra_dis[j]) / 2)
                d_inter = weight * np.linalg.norm(Final_Centroid_merge[i] - Final_Centroid_merge[j])

                if d_inter > d_bar:
                    count += 1
        # 上面计算出大于d_bar的簇对的数量count

        count_true = int((n_clusters_t ** 2 - n_clusters_t) / 2) # 这里(n_clusters_t ** 2 - n_clusters_t) / 2 一定是能整除的，但python计算中可能得到浮点数如3.00，因此int()换算成整数, 注意int(3.7)=3, int(-3.7)=-4

        print("-----------------iter: %d-----------------" % int(t + 1))
        print("n_clusters: %d" % n_clusters_t)
        print("count_true: %d" % count_true)
        print("count: %d" % count)

        """
        一般来说，聚类算法的终止条件可能包括：
        1. 达到预设的最大迭代次数
        2. 簇的数量达到预设的值
        3. 簇的变化（例如，簇的中心或者簇的成员）小于某个阈值
        4. 优化的目标函数（例如，簇内距离的和）的改变小于某个阈值
        """
        if count >= count_true:  ## 真的不懂这个终止条件??? count有可能>count_true吗; count=count_true就意味着所有簇对的距离全部等于d_bar, 这有合理性吗; 所以意味着几乎终止条件就是跑完200次迭代？
            print("Reach count!")
            break

        else:
            continue

    return Final_Centroid_merge, label, n_clusters_t, pred_f


# 执行聚类中心的分裂操作, 特别是通过分裂合并不合适的簇中心，从而提高聚类的性能
def centroid_split(X, X_1, Centroid_split, label):
    """
        Parameters
        ----------
        X
            Embedding after pre-training. Rows are cells and columns are genes.
        X_1
            Embedding + Column vectors of cell types (Label splicing in the last column 最后一列是标签).
        还有两个输入的参数: 当前的簇中心centroid_split, 当前的簇标签label
    """

    """
    函数的目标是根据某种条件对簇中心进行分裂，然后更新标签。分裂的条件是基于距离的，具体如下：

    1. 计算每个簇的平均距离 intra_dis 和平均距离 d_ave。
    2. 计算每对簇中心之间的权重距离，并选择距离最大的一对簇进行分裂。
    3. 选择两个簇中的样本并计算它们的距离，然后将它们分配到新的簇。
    4. 更新簇中心和标签。
    5. 重复上述步骤直到满足停止条件。
    函数返回分裂后的簇中心和更新后的标签
    """
    
    ### Compute weights
    intra_dis = cluster_intra_dis(X, Centroid_split, label)
    d_ave = np.mean(intra_dis)

    sum_1 = 0
    n_clusters = len(np.unique(label))
    for i in range(0, n_clusters):
        for j in range(i + 1, n_clusters):
            weight = d_ave / ((intra_dis[i] + intra_dis[j]) / 2)
            sum_1 += weight * np.linalg.norm(Centroid_split[i] - Centroid_split[j])

    d_bar = sum_1 / (n_clusters * (n_clusters - 1) / 2) 
    min_dis = d_bar / 2 

    X_copy = 1 * X_1

    Dis_tol = cluster_intra_dis(X, Centroid_split, label)

    """
    进行最多200次迭代。在每次迭代中:
    1. 找出距离大于min_dis且距离等于所有距离的最大值的簇, 将其分裂为两个新的簇
    2. 更新簇的质心Centroid_split 和簇的标签label
    3. 重新计算: intra_dis(包括每个簇内距离平均值)、d_ave (=np.mean(intra_dis))、所有簇的加权平均距离d_bar 和 min_dis (=d_bar / 2)
    4. 计算距离小于min_dis的簇的数量count
    5. 如果count大于等于簇的数量n_clusters, 则终止迭代
    """
    for t in range(200):

        idx_split = []
        for i in range(len(Centroid_split)):
            if Dis_tol[i] > min_dis and Dis_tol[i] == max(Dis_tol):
                idx_split.append(i)
                dis = []
                X_2 = np.delete(X_1[X_1[:, -1] == i], -1, 1)
                for m in range(len(X_2)):
                    dis_append = 0
                    for n in range(len(X_2)):
                        dis_append += np.linalg.norm(X_2[m] - X_2[n]) ** 2
                    dis.append(dis_append)
                idx_1 = np.argmin(dis)
                centriod_1 = X_2[idx_1]

                X_3 = np.delete(X_2, idx_1, 0)
                T_m = []
                for m in range(len(X_3)):
                    T_nm = 0
                    for n in range(len(X_3)):
                        D_n = np.linalg.norm(X_3[n] - centriod_1) ** 2
                        d_nm = np.linalg.norm(X_3[m] - X_3[n]) ** 2
                        T_nm += np.maximum(D_n - d_nm, 0)
                    T_m.append(T_nm)
                idx_2 = np.argmax(T_m)
                centriod_2 = X_3[idx_2]

                centroid = np.concatenate(
                    (centriod_1.reshape(len(centriod_1), 1).T, centriod_2.reshape(len(centriod_2), 1).T), axis=0)
                idx_1 = []
                for j in range(len(X_1[X_1[:, -1] == i])):
                    A = np.delete(X_1[X_1[:, -1] == i][j], -1)
                    distance = []
                    for e in range(2):
                        B = centroid[e]
                        D = np.linalg.norm(A - B)
                        distance.append(D)
                    idx = np.argmin(distance)
                    idx_1.append(idx)

                    if idx == 1:
                        if np.unique(label)[0] == 0:
                            idx_a = np.array(np.where(X_1[:, -1] == i))[0, j]
                            a = X_1[X_1[:, -1] == i]
                            a[j, -1] = len(Centroid_split) + i + 1
                            X_copy[idx_a, :] = a[j, :]

                        else:
                            idx_a = np.array(np.where(X_1[:, -1] == i))[0, j]
                            a = X_1[X_1[:, -1] == i]
                            a[j, -1] = len(Centroid_split) + 2 + i
                            X_copy[idx_a, :] = a[j, :]
                    else:
                        if np.unique(label)[0] == 0:
                            idx_a = np.array(np.where(X_1[:, -1] == i))[0, j]
                            a = X_1[X_1[:, -1] == i]
                            a[j, -1] = len(Centroid_split) + i
                            X_copy[idx_a, :] = a[j, :]

                        else:
                            idx_a = np.array(np.where(X_1[:, -1] == i))[0, j]
                            a = X_1[X_1[:, -1] == i]
                            a[j, -1] = len(Centroid_split) + 1 + i
                            X_copy[idx_a, :] = a[j, :]

                Centroid_split = np.concatenate(
                    (Centroid_split, centriod_1.reshape(1, len(centriod_1)), centriod_2.reshape(1, len(centriod_2))))

            else:
                continue

        if len(idx_split) == 0:
            Centroid_split = Centroid_split
            label = label
        else:
            Centroid_split = np.delete(Centroid_split, idx_split, 0)
            label = X_copy[:, -1]
            label = np.array(label)
            for i in range(len(np.unique(label))):
                if np.unique(label)[i] != i:
                    label[label == np.unique(label)[i]] = i
                else:
                    continue
            label = label.tolist()

        n_clusters = Centroid_split.shape[0]
        X_1 = np.concatenate([np.array(X), np.array(label).reshape(len(label), 1)], axis=1)
        X_copy = 1 * X_1

        ### Compute weights
        intra_dis = cluster_intra_dis(X, Centroid_split, np.array(label))
        d_ave = np.mean(intra_dis)

        sum_1 = 0
        for i in range(0, n_clusters):
            for j in range(i + 1, n_clusters):
                weight = d_ave / ((intra_dis[i] + intra_dis[j]) / 2)
                sum_1 += weight * np.linalg.norm(Centroid_split[i] - Centroid_split[j])

        d_bar = sum_1 / (n_clusters * (n_clusters - 1) / 2)
        min_dis = d_bar / 2

        Dis_tol = cluster_intra_dis(X, Centroid_split, np.array(label))

        # 计算簇内的算术平均距离Dis_tol 小于 min_dis(=d_var/2) 的簇的数量count
        # 也就是, 找出那些内部距离小于所有簇对之间距离平均值一半的簇。当所有簇都满足这个条件时，循环会停止
        count = 0
        for i in range(len(Dis_tol)):
            if Dis_tol[i] < min_dis:
                count += 1

        print("-----------------iter: %d-----------------" % int(t + 1))
        print("n_clusters: %d" % n_clusters)
        print("count_true: %d" % n_clusters)
        print("count: %d" % count)

        if count >= n_clusters:
            print("Reach count!")
            break

        else:
            continue

    # 返回簇的质心Centroid_split 和簇的标签label
    return Centroid_split, label 
