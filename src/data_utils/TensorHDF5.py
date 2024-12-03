'''
Greg Schuette 2024
'''
import numpy as np
import torch
from pathlib import Path
import h5py
import sys
import os

class TensorHDF5:

    def __init__(
        self,
        filepath,
        data_shape=None, # Tuple, list, etc., (must be convertible to a list) of integers. Only required when creating a new file.
        compression='gzip'
    ):
        self.__filepath = Path(filepath)

        if not self.filepath.exists():
            
            assert data_shape is not None, "When a file doesn't already exist, data_shape must be specified!"
            
            d = self.filepath.parent
            if d.is_file():
                raise Exception(f'The provided parent directory, {d}, is a file!')
            d.mkdir(parents=True, exist_ok=True)
            try:
                with h5py.File(self.filepath,'w') as hf:
                    hf['save_shape'] = list(data_shape)
                    hf.create_dataset('data', data=np.empty((0,*data_shape),dtype=np.float32), compression=compression, chunks=True, maxshape=(None,*data_shape))
            except:
                if self.filepath.is_file():
                    self.filepath.unlink()
                raise
                
        with h5py.File(self.filepath, 'r') as hf:
            data_shape1 = hf['save_shape'][:]
            if data_shape is not None:
                assert all(data_shape == data_shape1), f'The specified data_shape ({data_shape}) is different from those used in the specified file ({data_shape1}).'
            data_shape = data_shape1
            self.__idx = np.arange(hf['data'].shape[0])

        self.__data_shape = data_shape

    @property
    def filepath(self):
        return self.__filepath

    @property
    def data_shape(self):
        return self.__data_shape
    
    @property
    def n_bins(self):
        return self.__n_bins

    def __len__(self):
        return self.__idx.size

    def __getitem__(self,index):

        with h5py.File(self.filepath,'r') as hf:
            data = torch.from_numpy(hf['data'][index,...])
        if type(index) == int:
            return data.squeeze(0)
        return data

    def __format_input(self,value):
        if not isinstance(value,list):
            value = [value]

        ds = self.data_shape
        lds = len(ds)
        for k,val in enumerate(value):
            match val:
                case np.ndarray():
                    value[k] = val = torch.from_numpy(val).float()
                case torch.Tensor():
                    value[k] = val = val.detach().float().cpu()
                case _:
                    value[k] = val = torch.tensor(val).float()
            assert (
                val.ndim in [lds, lds+1] and
                all(val.shape[-k]==ds[-k] for k in range(1,lds+1))
            ), f'Expected save data with dimensions {self.data_shape}, with an optional batch dimension in front, but received {val.shape}.'
            if val.ndim == lds:
                value[k] = val = val.unsqueeze(0)

        value = torch.cat(value,dim=0)
        return value.numpy()
    
    def __setitem__(self,key,value,value_unprepared=True):
        
        if value_unprepared:
            value = self.__format_input(value)

        if isinstance(key,slice):
            key = self.__idx[key]
        
        max_idx = max(max(key),len(self)-1)
        with h5py.File(self.filepath,'a') as hf:
            if max_idx >= len(self):
                hf['data'].resize(max_idx+1,axis=0)
                self.__idx = np.arange(max_idx+1)
            hf['data'][key,...] = value

    def append(self,value):
        value = self.__format_input(value)
        key = np.arange(len(self),len(self) + value.shape[0])
        self.__setitem__(key,value,value_unprepared=False)