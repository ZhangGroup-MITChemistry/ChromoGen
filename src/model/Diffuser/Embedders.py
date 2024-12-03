'''
Greg Schuette 2023
'''
import torch
from torch import nn

# Support functions
class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

