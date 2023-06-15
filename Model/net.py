# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 10:50
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : net.py
# @Software: PyCharm
from torch import nn
import torch
from .stgcn import sgcn
from .wavemodule import DWTCompose, IDWT


class EmbeddingGraph(nn.Module):
    def __init__(self, nodes, dim, top_k, alpha, device):
        super(EmbeddingGraph,self).__init__()
        self.embedding1 = nn.Embedding(nodes, dim)
        self.embedding2 = nn.Embedding(nodes, dim)
        self.top_k = top_k
        self.alpha = alpha
        self.device = device
        self.LeReLu = nn.LeakyReLU()

    def forward(self, index_input):
        vec1 = self.embedding1(index_input)
        vec2 = self.embedding2(index_input)

        # matrix 1:
        matrix = torch.mm(vec1, vec2.transpose(1, 0)) - torch.mm(vec2, vec1.transpose(1, 0))
        adj = self.LeReLu(torch.tanh(self.alpha*matrix))
        # matrix 2:
        mask = torch.zeros(index_input.size(0), index_input.size(0)).to(self.device)
        mask.fill_(float('0'))
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.top_k, 1) # rand_like由区间[0,1)上均匀分布的随机数填充，topk：沿给定dim维度返回输入张量input中 k 个最大值，返回一个元组 (values,indices)
        mask.scatter_(1, t1, s1.fill_(1)) # scatter_修改mask矩阵，简单说就是通过一个张量 src  来修改另一个张量，哪个元素需要修改、用 src 中的哪个元素来修改由 dim 和 index 决定
        adj = adj * mask
        return adj


class WaveDecompose(nn.Module):
    def __init__(self, input_channel):
        super(WaveDecompose, self).__init__()
        self.dwt = DWTCompose(input_channel)

    def forward(self, x):
        dwt = self.dwt(x)
        return dwt


class IDWTLayers(nn.Module):
    def __init__(self):
        super(IDWTLayers, self).__init__()
        self.idwt = IDWT()

    def forward(self, low, high):
        return self.idwt(low, high)


class MutiLevelWaveGCN(nn.Module):
    """
    按层GCN，每一层的GCN返回结果，作为Latent表示，后续Latent再进行重构回不同的频率分量进行重构损失训练
    """
    def __init__(self, input_channel=8, gcn_depth=1, hopalpha=0.05):
        super(MutiLevelWaveGCN, self).__init__()
        self.gcn = sgcn(input_channel, input_channel, gcn_depth, hopalpha)

    def forward(self, wave_feature, adj):
        gc_output = self.gcn(wave_feature, adj)
        return gc_output
