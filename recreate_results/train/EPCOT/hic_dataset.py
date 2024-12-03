import numpy as np
from torch.utils.data import Dataset
import torch,pickle

class hic_dataset(Dataset):
    def __init__(self,train_chrs):
        with open('input_region_1Mb.pickle', 'rb') as f:
            input_locis = pickle.load(f)
        chrs = [str(i) for i in range(1, 23)]
        indices=[]
        for chr in train_chrs:
            locs = input_locis[chr]
            idx = chrs.index(chr)
            pad = idx*np.ones((locs.shape[0],1))
            indices.append(np.hstack((pad,locs)))
        self.indices = np.vstack(indices)
        self.num = self.indices.shape[0]
        print(self.indices.shape, self.num)
    def __getitem__(self, index):
        return self.indices[index]
    def __len__(self):
        return self.num


