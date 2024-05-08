import torch
from torch import nn

"""
ZINBLoss 中的forward方法用于度量VAE的 reconstruction loss
接收五个参数: x(输入数据), mean(预测的均值), disp(预测的离散度), pi(预测的零膨胀概率)和scale_factor(缩放因子) 然后计算并返回ZINB损失
"""
class ZINBLoss(nn.Module):
    def __init__(self, ridge_lambda=0):
        super(ZINBLoss, self).__init__()
        self.ridge_lambda = ridge_lambda

    def forward(self, x, mean, disp, pi, scale_factor):
        eps = 1e-10
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8 * torch.ones_like(x)), zero_case, nb_case)

        # 如果ridge_lambda>0, 还会添加一个岭回归正则项 (pi^2 * ridge_lambda) 用于防止过拟合
        if self.ridge_lambda > 0:
            ridge = self.ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)

        return result


"""
cluster_loss 用于计算聚类损失
接收两个参数: target(目标分布)和pred(预测分布), 然后计算并返回聚类损失(KL散度, 是一种衡量两个概率分布差异的方法(两个分布都需满足元素非负且整体=1), KL散度越小, 两个概率分布越接近)
"""
class ClusterLoss(nn.Module):

    def __init__(self):
        super(ClusterLoss, self).__init__()

    def forward(self, target, pred):
        """
        forward方法首先计算了target和pred的比值 然后取对数。这个结果再乘以target, 得到的是每个类别的KL散度
        然后, 这些每个类别的KL散度在最后一个维度(也就是类别维度 dim=-1)上求和, 得到的是每个样本的KL散度 [例如: 如果target和pred都是二维的, 那么我们就是在每一行(第一个维度是样本维度，第二个维度是类别维度)内部进行求和]
        最后,返回所有样本KL散度的均值, 作为聚类损失
        """
        # 为了防止除数为0，计算比值时给pred加了一个很小的常数1e-6
        return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=-1))

"""
ELOBkldLoss is a class that implements a loss based on the Gaussian distribution 
(正则化损失是KL散度, 度量了编码的潜在变量分布和先验分布(通常是标准正态分布)之间的差异)
ELOBkldLoss class的forward方法就是在计算正则化损失。这个方法接收两个参数: mu和logvar, 分别代表潜在变量的均值和对数方差。然后，它计算并返回正则化损失
"""
class ELOBkldLoss(nn.Module):
    def __init__(self):
        super(ELOBkldLoss, self).__init__()

    
    def forward(self, mu, logvar):
        result = -((0.5 * logvar) - (torch.exp(logvar) + mu ** 2) / 2. + 0.5)
        result = result.sum(dim=1).mean()

        return result
