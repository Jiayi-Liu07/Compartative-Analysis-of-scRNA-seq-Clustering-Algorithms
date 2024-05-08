import numpy as np
from torch.utils.data import Dataset 
import scipy.sparse as sp

"""
scDataset is a class for handling and loading data, especially for training and testing PyTorch models.
It inherits from torch.utils.data.Dataset.
"""

class scDataset(Dataset):
    def __init__(self, raw_mat, exp_mat, scale_factor):
        super(scDataset).__init__()
        if sp.issparse(raw_mat): # 如果raw_mat是稀疏矩阵
            raw_mat = raw_mat.todense() 
            """
            二维数组是一个数组，其中每个元素本身也是一个数组。这种结构类似于一个表格，有行和列。例如：
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
            稀疏矩阵是一种特殊的二维数组 其中大部分元素都是0。稀疏矩阵通常以一种压缩的方式存储, 只保存非零元素的值和位置, 以节省存储空间
            .todense()方法将稀疏矩阵转换为普通的二维数组, 也就是将所有的元素（包括零和非零元素）都存储在内存中
            
            例如，假设我们有以下的稀疏矩阵:
            from scipy.sparse import csr_matrix
            sparse_mat = csr_matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
            使用.todense()方法, 将稀疏矩阵转换为了一个普通的二维数组，所有的元素（包括零和非零元素）都被存储在了内存中
            dense_mat = sparse_mat.todense()
            print(dense_mat) 得到:
            [[1 0 0]
             [0 2 0]
             [0 0 3]]
            """ 

        # 将raw_mat, exp_mat, scale_factor转换为np.float32格式, 并存储为类的属性
        self.raw_mat = raw_mat.astype(np.float32)
        self.exp_mat = exp_mat.astype(np.float32)
        self.scale_factor = scale_factor.astype(np.float32)

    # __len__方法返回数据集的大小
    def __len__(self):
        return len(self.scale_factor)

    # __getitem__方法接收一个索引idx, 返回索引本身, raw_mat, exp_mat和scale_factor在该索引处的值
    def __getitem__(self, idx):
        return idx, self.raw_mat[idx, :], self.exp_mat[idx, :], self.scale_factor[idx, :]
