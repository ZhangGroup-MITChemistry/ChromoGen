import torch
from torch import nn, einsum
from einops.layers.torch import Rearrange
import numpy as np
import torch.nn.functional as F

class Convblock(nn.Module):
    def __init__(self,in_channel,kernel_size,dilate_size,dropout=0.1):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(
                in_channel, in_channel,
                kernel_size, padding=self.pad(kernel_size, dilate_size),
                dilation=dilate_size),
            nn.BatchNorm2d(in_channel),
            nn.Dropout(dropout)
        )
    def pad(self,kernelsize, dialte_size):
        return (kernelsize - 1) * dialte_size // 2
    def symmetric(self,x):
        return (x + x.permute(0,1,3,2)) / 2
    def forward(self,x):
        identity=x
        out=self.conv(x)
        x=out+identity
        x=self.symmetric(x)
        return F.relu(x)

class dilated_tower(nn.Module):
    def __init__(self,embed_dim,in_channel=48,kernel_size=9,dilate_rate=4):
        super().__init__()
        dilate_convs=[]
        for i in range(dilate_rate+1):
            dilate_convs.append(
                Convblock(in_channel,kernel_size=kernel_size,dilate_size=2**i))

        self.cnn=nn.Sequential(
            Rearrange('b l n d -> b d l n'),
            nn.Conv2d(embed_dim, in_channel, kernel_size=1),
            *dilate_convs,
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            Rearrange('b d l n -> b l n d'),
        )
    def forward(self,x,crop):
        x=self.cnn(x)
        x=x[:,crop:-crop,crop:-crop,:]
        return x

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool_fn = Rearrange('b (n p) d-> b n p d', n=1)
        self.to_attn_logits = nn.Parameter(torch.eye(dim))

    def forward(self, x):
        attn_logits = einsum('b n d, d e -> b n e', x, self.to_attn_logits)
        x = self.pool_fn(x)
        logits = self.pool_fn(attn_logits)

        attn = logits.softmax(dim = -2)
        return (x * attn).sum(dim = -2).squeeze()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        conv_kernel_size1 = 10
        conv_kernel_size2 = 8
        pool_kernel_size1 = 5
        pool_kernel_size2 = 4
        self.conv_net = nn.Sequential(
            nn.Conv1d(5, 256, kernel_size=conv_kernel_size1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv1d(256, 256, kernel_size=conv_kernel_size1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size1, stride=pool_kernel_size1),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.1),
            nn.Conv1d(256, 360, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv1d(360, 360, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size2, stride=pool_kernel_size2),
            nn.BatchNorm1d(360),
            nn.Dropout(p=0.1),
            nn.Conv1d(360, 512, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv1d(512, 512, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2))
        self.num_channels = 512
    def forward(self, x):
        out = self.conv_net(x)
        return out

