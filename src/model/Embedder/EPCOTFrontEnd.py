'''
Rewrite of the EPCOT "Tranmodel" class in https://github.com/liu-bioinfo-lab/EPCOT/blob/main/COP/hic/model.py
Greg Schuette 2024
'''

import torch
from torch import nn
from collections import OrderedDict
from pathlib import Path
from einops.layers.torch import Rearrange
from .CNNBackbone import CNNBackbone
from .Transformer import Transformer

class EPCOTFrontEnd(nn.Sequential):

    def __init__(
        self,
        CNNBackbone_kwargs={},
        Transformer_kwargs={
            'd_model':512,
            'dropout':0.2,
            'nhead':4,
            'dim_feedforward':1024,
            'num_encoder_layers':1,
            'num_decoder_layers':1
        }
    ):
        super().__init__()
        self.extend([
            CNNBackbone(**CNNBackbone_kwargs),
            Transformer(**Transformer_kwargs)
        ])
        
        self.insert(
            1,
            nn.Conv1d(
                in_channels=self[0].config['out_channels'][-1],
                out_channels=self[1].d_model,
                kernel_size=1
            )
        )

        self.cnn_config = self[0].config
        self.transformer_config = Transformer_kwargs

    @property
    def dtype(self):
        for p in self.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.parameters():
            return p.device

    @staticmethod
    def __rename_param_keys(params: OrderedDict, desired_keys: list):
        # This allows us to load parameters from files that 
        # were created using the original EPCOT code. 
        params1 = params.copy()
        for key in params:
            if key in desired_keys:
                continue
            elif key[:14] != 'pretrain_model':
                # These parameters are most likely from 
                # the decoder part. If not, then we were likely
                # given the wrong file. 
                params1.pop(key)
            else:
                # These parameters correspond to Tranmodel, 
                # now renamed SequenceEmbedder. Rename they keys
                # to match the new convention. 
                if 'conv_net' in key:
                    prefix = key[:key.rfind('.')]
                    n = int(prefix[prefix.rfind('.')+1:])
                    key1 = key.replace(prefix,f'0.{n//8}.{n%8}')
                else:
                    key1 = key.replace(
                        'pretrain_model.input_proj','1'
                    ).replace(
                        'pretrain_model.transformer','2'
                    )
                params1[key1] = params1.pop(key)
        return params1
    
    def load(self,filepath_or_data):
        if isinstance(filepath_or_data,OrderedDict):
            # Given the parameters. Likely loaded from file generated 
            # with original EPCOT code, so map the keys to those present
            # in the new code. 
            od = self.__rename_param_keys(filepath_or_data, list(self.state_dict().keys()))
        elif isinstance(filepath_or_data,dict):
            od = filepath_or_data['params']
        else:
            return self.load(torch.load(filepath_or_data,map_location=self.device))

        return self.load_state_dict(od)

    def save(self,filepath):
        f = Path(filepath)
        assert not f.is_dir(), f'{filepath} is a directory!'
        f.parent.mkdir(exist_ok=True,parents=True)
        f = f.with_suffix('.pt')
        torch.save(
            {
                'params':self.state_dict(),
                'cnn_config':self.cnn_config,
                'transformer_config':self.transformer_config
            },
            f
        )
    
    @staticmethod
    def from_file(filepath,dtype=None,device=None):
        data = torch.load(filepath,map_location='cpu')
        if isinstance(data,OrderedDict):
            # Assume default configuration since it isn't specified. 
            # This will work for files generated with the original EPCOT code
            # with our chosen hyperparameters since those are set as default. 
            sequence_embedder = SequenceEmbedder()
        else:
            # The config info should be in the save file. 
            sequence_embedder = SequenceEmbedder(
                CNNBackbone_kwargs = data['cnn_config'], 
                Transformer_kwargs = data['transformer_config']
            )

        # Pass the full dict to avoid the computational cost involved with
        # SequenceEmbedder.load() calling SequenceEmbedder.__rename_param_keys()
        sequence_embedder.load(data)
        
        return sequence_embedder.to(dtype=dtype,device=device) 

    #############################################################################
    # Overriding the forward function because I missed the errors arising from 
    # improper shapes AFTER I wrote the __rename_param_keys method. Will likely 
    # fix later, but for now, need to wrap this up... 
    def forward(self,x):
        assert isinstance(x,torch.Tensor), f'Input to {__class__.__name__}.forward() should be torch.Tensor object. Received {type(x)}.'
        ndim = x.ndim
        assert ndim >= 2, f'Input to {__class__.__name__}.forward() should have at least two dimensions, while your input has {ndim}.'
        n = self.cnn_config['in_channels']
        assert x.shape[-2] == n, f'Input to {__class__.__name__}.forward() should have size {n} in the second-to-last dimension, while your input has {x.shape[-2]}.'

        if ndim == 2:
            x = x.unsqueeze()
        elif ndim > 3:
            x_shape = x.shape
            x = x.reshape(torch.tensor(x_shape[:-2]).prod(), *x_shape[-2:])
                              
        x = super().forward(x)
        if ndim > 3:
            x = x.reshape(*x_shape[:-2],*x.shape[-2:])

        return x
            
    