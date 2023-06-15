# -*- coding: utf-8 -*-
# @Time    : 2021/11/25 15:36
# @Author  : SSK
# @Email   : skshao@bjtu.edu.cn
# @File    : wavemodule.py
# @Software: PyCharm
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse


class DWTCompose(nn.Module):
    def __init__(self, node):
        super(DWTCompose, self).__init__()
        self.dwt1 = DWT1DForward(J=1, wave='db1', mode='zero')
        self.dwt2 = DWT1DForward(J=1, wave='db1', mode='zero')
        self.dwt3 = DWT1DForward(J=1, wave='db1', mode='zero')
        self.bn = nn.BatchNorm1d(node)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        dwt1_low, dwt1_high = self.dwt1(x)
        dwt2_low, dwt2_high = self.dwt1(dwt1_low)
        dwt3_low, dwt3_high = self.dwt1(dwt2_low)

        dwt1_low = dwt1_low
        dwt1_high = dwt1_high[0]

        dwt2_low = dwt2_low
        dwt2_high = dwt2_high[0]

        dwt3_low = dwt3_low
        dwt3_high = dwt3_high[0]

        dwt1_low = dwt1_low.permute(0, 2, 1)
        dwt1_high = dwt1_high.permute(0, 2, 1)
        dwt2_low = dwt2_low.permute(0, 2, 1)
        dwt2_high = dwt2_high.permute(0, 2, 1)
        dwt3_low = dwt3_low.permute(0, 2, 1)
        dwt3_high = dwt3_high.permute(0, 2, 1)

        return [(dwt1_low, dwt1_high), (dwt2_low, dwt2_high), (dwt3_low, dwt3_high)]


class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.idwt1 = DWT1DInverse(wave='db1', mode='zero')

    def forward(self, low, high):
        return self.idwt1((low, high))
