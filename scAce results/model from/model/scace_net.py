import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
gene expression matrix -> encoder -> latent space(将数据映射到低维空间) -> decoder
这部分的主要包括 1)自定义的激活函数 Mish, 2)能够处理噪声的encoder和decoder, 3)用于聚类的软分配矩阵
"""


class Mish(nn.Module):  # class Mish, 继承自torch.nn.Module, 是自定义的激活函数
    def __init__(self):  # 定义Mish类的初始化函数，__init__ 在创建Mish对象时会被调用
        super().__init__()

    def forward(self, x):  # 定义Mish类的前向传播函数
        return x * torch.tanh(F.softplus(x))


def buildNetwork(layers, activation="relu"):  # buildNetwork用于构建fully connected neural networks
    # 参数layers是一个列表，表示每一层的神经元数量; activation是一个字符串，表示使用的激活函数类型, 默认为relu

    net = []  # net是一个列表，用于存储神经网络的每一层
    for i in range(1, len(layers)):  # 从1开始（因为第0层是输入层）创建fully connected neural network，len(layers)表示神经网络的层数，即全连接层的数量
        layer = nn.Linear(layers[i - 1],
                          layers[i])  # 首先根据layers列表创建全连接层: nn.Linear是PyTorch中的全连接层，第一个参数是输入的维度，第二个参数是输出的维度
        nn.init.kaiming_normal_(layer.weight)  # 使用kaiming_normal_方法初始化权重
        # nn.init.kaiming_normal_(layer.bias)
        nn.init.constant_(layer.bias, 0)  # 使用constant_方法将偏置初始化为0
        net.append(layer)  # 将全连接层添加到net列表中

        # 根据activation参数添加对应的激活函数
        # net.append(nn.BatchNorm1D(layers[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())
        elif activation == "mish":
            net.append(Mish())
        elif activation == "tanh":
            net.append(nn.Tanh())
    return nn.Sequential(*net)  # 使用nn.Sequential将所有的层连接起来，形成一个完整的神经网络


"""
Adding Gaussian noise to the input data during training is a form of data augmentation and regularization technique.  
It can help to prevent overfitting by providing a form of randomness to the model training process.  
This can make the model more robust and improve its generalization ability, as it learns to perform well not just on the exact training data, but also on slightly modified or noisy versions of the data.
"""


class GaussianNoise(nn.Module):  # 是继承自nn.Module的类，用于添加高斯噪声
    def __init__(self, sigma=0):  # 初始化函数，接受一个参数：sigma-是高斯噪声的标准差
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        # 在前向传播函数forward中，如果模型处于训练模式，那么就向输入x添加以0为均值、sigma为标准差的高斯噪声
        if self.training:
            x = x + self.sigma * torch.randn_like(x.shape)
        return x  # 返回添加了噪声的输入x


"""
MeanAct和DispAct是自定义的激活函数, 它们在计算过程中使用了torch.exp和F.softplus
但在结果上加了限制, 使用了torch.clamp来限制输出值的范围
MeanAct can be useful in scenarios where the model needs to produce positive outputs, as the exponential function ensures that the output is always positive.
DispAct applies a softplus function (a smooth approximation to the ReLU) and can help to mitigate the vanishing gradient problem, which can occur when training deep neural networks.
"""


class MeanAct(nn.Module):  # 是自定义的激活函数
    def __init__(self):
        super(MeanAct, self).__init__()

    # 在前向传播函数forward中对输入x进行指数运算，然后使用torch.clamp函数将结果限制在1e-5和1e6之间
    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)  #


class DispAct(nn.Module):  # 是自定义的激活函数
    def __init__(self):
        super(DispAct, self).__init__()

    # 在前向传播函数forward中对输入x进行softplus运算，然后使用torch.clamp函数将结果限制在1e-4和1e4之间
    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


"""
Encoder函数 和 reparameterize函数一起实现了编码过程
soft_assign函数 和 target_distribution函数一起帮助实现了聚类过程
VAE方法中 encoder 和 decoder是一起训练的。训练过程中, 我们希望解码器的输出尽可能接近原始输入数据，这样可以确保潜在表示能够捕捉到输入数据的关键信息
我们并不是直接优化编码器的输入数据，而是优化编码器和解码器的参数。通过最小化输入数据和解码器输出之间的差异（通常使用重构误差作为度量），我们可以间接地优化编码器的输出，使其能够生成更好的潜在表示
"""


# 用于计算目标分布 p (根据参数q, 这是一个软分配矩阵，表示每个样本属于每个聚类的概率)
def target_distribution(q):
    p = q ** 2 / q.sum(0)
    return (p.t() / p.sum(1)).t()


class scAce(nn.Module):

    #  初始化方法‘__init__’ 接受一些超参数包括：输入维度input_dim,(将数据映射到的低维空间的)嵌入维度z_dim, 编码器和解码器的层次结构encode_layers/decode_layers，激活函数activation
    def __init__(self, input_dim, device, z_dim=32,
                 encode_layers=[512, 256],  # encoder有两层，第一层512个神经元，第二层256个神经元
                 decode_layers=[256, 512],
                 activation='relu'):
        super(scAce, self).__init__()
        self.z_dim = z_dim
        self.activation = activation

        # self.mu = None
        self.pretrain = False
        self.device = device
        self.alpha = 1.
        self.sigma = 0  # 1.

        self.encoder = buildNetwork([input_dim] + encode_layers, activation=activation)
        self.decoder = buildNetwork([z_dim] + decode_layers, activation=activation)

        self.enc_mu = nn.Linear(encode_layers[-1], z_dim)
        self.enc_var = nn.Linear(encode_layers[-1], z_dim)

        self.dec_mean = nn.Sequential(nn.Linear(decode_layers[-1], input_dim), MeanAct())
        self.dec_disp = nn.Sequential(nn.Linear(decode_layers[-1], input_dim), DispAct())
        self.dec_pi = nn.Sequential(nn.Linear(decode_layers[-1], input_dim), nn.Sigmoid())

    # soft_assign 用于计算软分配矩阵q，即每个样本都被分配给所有的聚类中心，但是分配的概率是不同的，这个概率是通过计算样本与聚类中心的距离得到的
    # 这种软分配的策略可以使模型更加灵活，能够更好地处理数据的不确定性和噪声
    def soft_assign(self, z):

        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu) ** 2, dim=2))
        q = (q.t() / torch.sum(q, dim=1)).t()
        return q

    # reparameterize 将均值mu和对数方差logvar转换为正态分布中的随机样本,是VAE的一个关键步骤，用于使模型能够进行梯度下降优化
    def reparameterize(self, mu, logvar):

        # obtain standard deviation from log variance
        std = torch.exp(0.5 * logvar)
        # values are sampled from unit normal distribution
        eps = torch.randn(std.shape).to(self.device)
        return mu + eps * std

    """
    注意这里只是描述了怎么使用encoder和decoder, 而还没有给出具体的网络结构(self.encoder和self.decoder没有被定义)
    """

    def Encoder(self, x):
        # h被用于与训练阶段, 它的输出z被送入decoder, 进一步重构来计算重构误差来进行模型优化
        # h0被用于与聚类阶段，它的输出Z0被用于计算软软分配矩阵q，然后通过比较q和目标分布p来计算聚类损失，也是模型优化的另一个重要目标

        # h = self.encoder(x + torch.randn_like(x) * self.sigma)
        h = self.encoder(x)
        # self.enc_mu(h) 和 self.enc_var(h)是两个神经网络层,输入h (h是encoder的输出 dim=256)
        #                                                输出z_mu 和 z_logvar (dim=32)
        z_mu = self.enc_mu(h)
        z_logvar = self.enc_var(h)
        # 然后通过重参数化技巧生成32维的z
        z = self.reparameterize(z_mu, z_logvar)

        if self.pretrain:  # 如果模型处于预训练状态
            return z_mu, z_logvar, z  # 返回z_mu, z_logvar, z

        h0 = self.encoder(x)
        z_mu0 = self.enc_mu(h0)
        z_logvar0 = self.enc_var(h0)
        z0 = self.reparameterize(z_mu0, z_logvar0)
        return z_mu, z_logvar, z, z0  # 返回z_mu, z_logvar, z, z0

    def Decoder(self, z):

        h = self.decoder(z)  # 通过解码器网络将潜在表示z转换为隐藏表示h
        mu = self.dec_mean(h)  # 计算h的均值mu
        disp = self.dec_disp(h)  # 计算h的对数方差disp
        pi = self.dec_pi(h)  # 计算h的混合参数pi

        return mu, disp, pi

    def forward(self, x):
        if self.pretrain:  # 如果模型处于预训练状态
            # Encode
            z_mu, z_logvar, z = self.Encoder(x)

            # Decode
            mu, disp, pi = self.Decoder(z)

            return z_mu, z_logvar, mu, disp, pi

        # else

        # Encode
        z_mu, z_logvar, z, z0 = self.Encoder(x)

        # Decode
        mu, disp, pi = self.Decoder(z)

        # cluster
        q = self.soft_assign(z0)  # 对z0进行软分配，得到每个样本属于每个聚类的概率

        return z_mu, z_logvar, mu, disp, pi, q

    # 利用小批量梯度下降（Mini-batch Gradient Descent）的优化算法 batch_size=256
    # 通过在每次迭代中使用一部分样本，可以既利用GPU并行计算的优势，又能减少因样本噪声带来的优化不稳定性
    def EncodeAll(self, X, batch_size=256):
        all_z_mu = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0 * X.shape[0] / batch_size))  # 计算总的批次数量, 向上取整以确保所有样本都能被处理
        for batch_idx in range(num_batch):  # 对每个批次进行处理
            exp = X[batch_idx * batch_size: min((batch_idx + 1) * batch_size, num)]  # 取出一个批次的数据
            exp = torch.tensor(np.float32(exp))  # 将数据转换为torch.tensor格式, 并确保数据类型为float32
            with torch.no_grad():  # 在计算梯度时，不会对这部分的操作进行求导
                z_mu, _, _, _ = self.Encoder(exp.to(self.device))  # 将数据送入encoder, 得到z_mu

            all_z_mu.append(z_mu)  # 将z_mu添加到all_z_mu列表中

        all_z_mu = torch.cat(all_z_mu, dim=0)  # 将所有批次的编码结果拼接起来，dim=0表示按照第一个维度（即样本维度）进行拼接
        return all_z_mu


# 用于初始化神经网络模型的权重～模型的训练速度和最终的性能
def weight_init(m):  # 这里m是一个神经网络层, 用来对模型的所有层进行初始化
    nn.init.xavier_normal_(
        m.weight)  # 使用xavier_normal_方法初始化权重，xavier_normal_方法是一种常用的权重初始化方法，它可以使得每一层的输出方差尽可能相等, 从而避免了梯度消失和梯度爆炸的问题
    # nn.init.kaiming_normal_(m.bias)
    nn.init.constant_(m.bias, 0)  # 将偏置项（bias）初始化为0, 这是一种常见的偏置项初始化方法
