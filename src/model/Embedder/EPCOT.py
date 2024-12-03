'''
~Essentially~ a rewrite of the "finetunemodel" class in https://github.com/liu-bioinfo-lab/EPCOT/blob/main/COP/hic/model.py

Greg Schuette 2024
'''

import torch
from torch import nn, einsum
import torch.nn.functional as F
from types import NoneType
import warnings
from collections import OrderedDict
from pathlib import Path
import inspect
from einops.layers.torch import Rearrange
from .EPCOTFrontEnd import EPCOTFrontEnd

############################################################################################################
# Some minor support functions
def verify_int(x,arg_name,minimum=None,maximum=None):
    
    if isinstance(x,float) and int(x) == x:
        x = int(x)
    assert isinstance(x,int), f'Expected {arg_name} to be an integer. Receieved {type(x)}.'
    
    if minimum is not None:
        minimum = verify_int(minimum,'minimum')
        if maximum is not None:
            maximum = verify_int(maximum,'maximum')
            assert minimum <= x <= maximum, f'{arg_name} must lie in the range [{minimum},{maximum}]. Received {x}.'
        assert minimum <= x, f'{arg_name} must be greater than or equal to {minimum}. Received {x}.'
        
    elif maximum is not None:
        maximum = verify_int(maximum,'maximum')
        assert x <= maximum, f'{arg_name} must be less than or equal to {maximum}. Received {x}.'
        
    return x

def verify_activation(x,calling_class,None_ok=True):
    if x is None:
        if None_ok:
            return lambda x: x
        else:
            raise Exception(f'{calling_class} requires activation function to be defined.')

    if inspect.isfunction(x) or isinstance(x,nn.Module):
        return x

    if inspect.isclass(x):
        try:
            return x()
        except:
            raise Exception(
                f'Activation function passed to {calling_class}, {x}, cannot be initialized without arguments. Please initialize before passing.'
            )

    if isinstance(x,str):
        if hasattr(nn,x):
            try:
                return getattr(nn,x)()
            except:
                raise Exception(
                    f'The activation function passed to {calling_class}, {x}, seems to require arguments to be initialized. Please initialize before passing.'
                )
        if hassattr(F,x):
            return getattr(F,x)
        raise Exception(f'When activations are passed as a string, they should be an attribute of torch.nn or torch.nn.Functional. The passed {x} is not.')

    raise Exception(
        f'Activation functions must be passed to {calling_class} as a function, an nn.Module instance, '
        f'a class that can be initialized without input arguments, {"" if None_ok else "or "}a string denoting an attribute of '
        'torch.nn or torch.nn.functional' + (', or a NoneType object.' if None_ok else '.') + 
        f' Received {type(x)}.'
    )
        

############################################################################################################
# Supporting classes 

##########
# This class was taken directly from the EPCOT repository. 
# The only modification is replacing the batch dimension with ellipses to generalize the number of batch
# dimensions permitted
class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool_fn = Rearrange('... (n p) d -> ... n p d', n=1)
        self.to_attn_logits = nn.Parameter(torch.eye(dim))

    def forward(self, x):
        #attn_logits = einsum('... n d, d e -> ... n e', x, self.to_attn_logits)
        attn_logits = x @ self.to_attn_logits
        x = self.pool_fn(x)
        logits = self.pool_fn(attn_logits)

        attn = logits.softmax(dim = -2)
        return (x * attn).sum(dim = -2).squeeze()

##########
# Creating an nn.Module to replace the output_head method used in the original EPCOT repository.
# Also replaces the dist_embed layer and , 
class OutputHead(nn.Module):

    def __init__(self, bins_per_sample, num_embeddings, embedding_dim, dropout_prob):
        super().__init__()

        # Verify inputs
        bins_per_sample = verify_int(bins_per_sample,'bins_per_sample',minimum=1)
        num_embeddings = verify_int(num_embeddings,'num_embeddings',minimum=1)
        embedding_dim = verify_int(embedding_dim,'embedding_dim')
        assert isinstance(dropout_prob, float), f'dropout_prob passed to {__class__.__name__} must be a float. Received {type(dropout_prob)}.'
        assert 0<=dropout_prob<=1, f'dropout_prob passed to {__class__.__name__} must lie in the range [0,1]. Received {dropout_prob}.'

        # The original EPCOT repo created a distance index via the position_matrix method 
        # with every batch, then used that to index the embedding produced by nn.Embedding. 
        # Both of these are entirely unnecessary, but I'll only deal with the indexing here
        # to simplify backward compatibility. This also saves some memory by keeping the 
        # distance embeddings in its smaller representation until its expanded form is needed.
        # The new approach also saves a lot of memory transiently by broadcasting the embedding
        # (and, indirectly, its index) rather than creating an object the same size as the
        # embedded batch. 
        idx = torch.arange(bins_per_sample).reshape(1,1,bins_per_sample)
        self.distance_embedding_index = (idx.mT - idx).abs_().clamp_(0,num_embeddings-1) # Replaces the finetunemodel.position_matrix method
        self.distance_embedding = nn.Embedding(num_embeddings,embedding_dim)             # Renaming of the finetunemodel.distance_embed attribute
        self.distance_embedding_dropout = nn.Dropout(dropout_prob)                       # Renaming of the finetunemodel.dist_dropout attribute

    def requires_grad_(self,requires_grad):
        # Otherwise, PyTorch keeps trying to apply requires_grad_ to self.distance_embedding_index 
        for p in self.parameters():
            try:
                p.requires_grad_(requires_grad)
            except:
                pass
    
    def output_head(self,x,dist_embed,bins):
        x1=torch.tile(x.unsqueeze(1),(1,bins,1,1))
        x2 = x1.permute(0,2,1,3)
        mean_out=(x1+x2)/2
        dot_out=x1*x2
        return mean_out+dot_out+dist_embed
    
    def forward(self, sequence_embeddings):
        
        # Add a dimension & expand to the number of bins, essentially repeating the embeddings matrices in the new dimension
        # while keeping memory usage low until the next operation is performed. 
        # Equivalent to the following from the EPCOT repo: x1=torch.tile(x.unsqueeze(1),(1,bins,1,1))
        n = sequence_embeddings.ndim - 2  # number of batch dimensions
        b = sequence_embeddings.shape[-1] # Number of bins per sample
        x1 = sequence_embeddings.unsqueeze(-3).expand(*[-1]*n, b, -1, -1)

        # Get a new view with the new/expanded and prior sequencing dimensions transposed. 
        # Equivalent to `x2 = x1.permute(0,2,1,3)` from the EPCOT repository. 
        x2 = x1.transpose(-2,-3)

        # Get the distance embedding, expand to x1 & x2's size, and apply the dropout layer.
        distance_embedding = self.distance_embedding_dropout(
            self.distance_embedding(
                self.distance_embedding_index.to(x1.device)
            ).expand_as(x1)
        )
        
        # 1. Symmetrize sequence embedding object in the new/expanded and prior sequencing dimension.
            # Equivalent to `mean_out=(x1+x2)/2` in the EPCOT repo
        # 2. Get the Hadamard (element-wise) product between x1 & x2 (NOT dot product, as was errantly 
            # implied in the EPCOT repo). 
            # Equivalent to `dot_out=x1*x2` in the EPCOT repo
        # 3. Add (1) and (2) to the distance embedding and return.
            # Equivalent to `return mean_out+dot_out+dist_embed` in the EPCOT repo
        return (x1+x2)/2 + x1*x2 + distance_embedding

##########
# DilationTower and its support classes. 
# Minor revision of the 'dilated_tower' class in the EPCOT repository, now named DilationTower. 
# This includes slight revisions to their 'Convblock' class, now named SymmetricResNetBlock. 

# New crop module
class Crop(nn.Module):

    def __init__(self,amount_to_crop):
        super().__init__()
        self.n = verify_int(amount_to_crop,'amount_to_crop',minimum=0)
    
    def forward(self,x):
        if n:=self.n:
            # Don't do anything if crop is set to 0
            x = x[...,n:-n,n:-n,:] 
        return x

# Adaptation of the 'Convblock' class in the original EPCOT repository
class SymmetricResNetBlock(nn.Module):
    # I haven't ~personally~ encountered a ResNet with just one layer before, but 
    # I'm calling it that anyway to denote the residual connection made at the end. 
    
    def __init__(
        self,
        n_channels,
        kernel_size,
        dilation,
        dropout_prob=0.1,
        activation = nn.ReLU(),
        **kwargs
    ):
        super().__init__()
        
        # Verify inputs
        n_channels = verify_int(n_channels,'n_channels',minimum=1)
        kernel_size = verify_int(kernel_size,'kernel_size',minimum=1)
        dilation = verify_int(dilation,'dilation',minimum=1)
        assert isinstance(dropout_prob,float), f'Expected dropout_prob to be a float. Received {type(dropout_prob)}.'

        # Build the block
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels  = n_channels, 
                out_channels = n_channels,
                kernel_size  = kernel_size,
                padding  = 'same',
                dilation = dilation
            ),
            nn.BatchNorm2d(n_channels),
            nn.Dropout(dropout_prob)
        )

        self.activation = verify_activation(activation,SymmetricResNetBlock)
        
        
    def forward(self,x):
        # Create a residual connection between input and output of convolutional block
        x = x + self.conv_block(x)

        # Force symmetry in the sequence-indexed dimensions
        x = (x + x.mT)/2

        # Apply the activation function. 
        # In ChromoGen, as in EPCOT, we use a ReLU activation to force all values to be >= 0
        return self.activation(x)

# Rewrite of the 'dilated_tower' class in the EPCOT repository
class DilationTower(nn.Sequential):
    
    def __init__(
        self,
        in_channels=256,
        out_channels=48,
        kernel_size=1,
        symmetric_resnet_kernel_size=9,
        dilations = None,
        n_symmetric_resnet_blocks=5,
        amount_to_crop=4,
        resnet_block_activation=nn.ReLU(),
        resnet_block_dropout_prob=0.1
    ):
        super().__init__()
        
        ###############
        # Verify inputs 
        in_channels = verify_int(in_channels,'in_channels',minimum=1)
        out_channels = verify_int(out_channels,'out_channels',minimum=1)
        kernel_size = verify_int(kernel_size,'kernel_size',minimum=1)
        symmetric_resnet_kernel_size = verify_int(symmetric_resnet_kernel_size,'symmetric_resnet_kernel_size',minimum=1)
        n_symmetric_resnet_blocks = verify_int(n_symmetric_resnet_blocks,'n_symmetric_resnet_blocks',minimum=0)
        amount_to_crop = verify_int(amount_to_crop,'amount_to_crop',minimum=0)
        assert isinstance(resnet_block_dropout_prob,float) and 0<=resnet_block_dropout_prob<=1, \
        f'resnet_block_dropout_prob must be a float in the range [0,1]. Received {resnet_block_dropout_prob}.'
        resnet_block_activation = verify_activation(resnet_block_activation,f'{__class__.__name__} as resnet_block_activation') 
        match dilations:
            case None:
                dilations = tuple(2**i for i in range(n_symmetric_resnet_blocks))
            case int():
                dilations = (dilations,)
            case list() | tuple():
                dilations = tuple(dilations)
            case _:
                raise Exception(
                    'dilations argument DilationTower to must be None, a positive integer, '
                    'a tuple of positive integers, or a list of positive integers. '
                    f'Received {type(dilations)}.'
                )
        invalid_dilations = {d for d in dilations if not isinstance(d,int) or d<1}
        assert not invalid_dilations, (
            'All dilation values must be positive integers, '
            f'but the passed dilations argument includes invalid value(s) {invalid_dilations}.'
        )
        if len(dilations) == 1:
            dilations*= n_symmetric_resnet_blocks
        assert len(dilations) == n_symmetric_resnet_blocks, (
            f'The number of dilation values passed in the dilations argument to {DilationTower} '
            f'must be either 1 or equal to n_symmetric_resnet_blocks ({n_symmetric_resnet_blocks}), '
            f'but dilations contains {len(dilations)} entries.'
        )

        ###############
        # Initialize the object

        # Add the pre-ResNet blocks
        self.extend([
            # Change dimensions so that the Conv2d layers operate on genomic index/sequence embedding dimensions
            Rearrange('b l n d -> b d l n'),
            # Use a convolutional layer to set the number of channels. 
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding='same')
        ])

        # Add the symmetric ResNet blocks
        for dilation in dilations:
            self.append(
                SymmetricResNetBlock(
                    n_channels=out_channels,
                    kernel_size=symmetric_resnet_kernel_size,
                    dilation=dilation,
                    dropout_prob=resnet_block_dropout_prob,
                    activation=resnet_block_activation
                )
            )

        # Add the output layers
        self.extend([
            # Include one final Conv2d layer to mix the data. 
            # (EPCOT and ChromoGen use kernel_size 1, so this just mixes the channels.)
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same'),
            # Reorganize dimensions to their original meaning. 
            Rearrange('b d l n -> b l n d'),
            # Crop the data, as desired. 
            Crop(amount_to_crop)
        ])

############################################################################################################
# The main event, adapted from the 'finetunemodel' class in the original EPCOT repository. 

class EPCOT(nn.Module):

    def __init__(
        self,
        # EPCOTFrontEnd instance, which is used in this model as part of the transfer learning procedure. 
        front_end=None,
        *,
        bins_per_sample=256,            # Number of bins per sample. Renamed from bins in the original implementation
        num_distance_embeddings=11,     # Renamed from max_bin in the original implementation (and redefined as max_bin+1)
        hidden_dim=512,      
        embed_dim=256,
        trunk='transformer',
        trunk_kwargs={},
        mode='embed_sequences',
        amount_to_crop=4,
        distance_embedding_dropout_prob=0.1,
        dilation_tower_out_channels=64, # Known as 'in_dim' in the original EPCOT implementation
        dilation_tower_first_last_kernel_size=1,
        symmetric_resnet_blocks_kernel_size=9,
        n_symmetric_resnet_blocks=5,
        **kwargs
    ):
        super().__init__()

        ########
        # Verify inputs, name basic attributes
        if front_end is None:
            front_end = EPCOTFrontEnd() # Default
        else:
            assert isinstance(front_end,EPCOTFrontEnd), (
                f'Expected front_end to be a {EPCOTFrontEnd} instance but received {type(front_end).__name__}. '
                'Don\'t pass any front_end value if you\'d like to use the default front_end architecture. '
            )
        
        self.bins_per_sample = bins_per_sample = verify_int(bins_per_sample,'bins_per_sample',minimum=1)
        num_distance_embeddings = verify_int(num_distance_embeddings,'num_distance_embeddings',minimum=1)
        self.hidden_dim = hidden_dim = verify_int(hidden_dim,'hidden_dim',minimum=1)
        self.embed_dim = embed_dim = verify_int(embed_dim,'embed_dim',minimum=1)
        self.amount_to_crop = amount_to_crop = verify_int(amount_to_crop,'amount_to_crop',minimum=0)
        assert isinstance(trunk,str), f'trunk must be a string. Received {type(trunk)}.'
        assert trunk in ['transformer','LSTM'], f'trunk must be "transformer" or "LSTM". Received "{trunk}".'

        # Preserve the config details so they can be placed in a save file. This attribute 
        # is a pre-requisite for the final check performed in this __init__ function
        self.__config = {
            'bins_per_sample':bins_per_sample,
            'num_distance_embeddings':num_distance_embeddings,
            'hidden_dim':hidden_dim,
            'embed_dim':embed_dim,
            'trunk':trunk,
            'trunk_kwargs':trunk_kwargs,
            'mode':mode,
            'amount_to_crop':amount_to_crop,
            'distance_embedding_dropout_prob':distance_embedding_dropout_prob,
            'dilation_tower_out_channels':dilation_tower_out_channels,
            'dilation_tower_first_last_kernel_size':dilation_tower_first_last_kernel_size,
            'symmetric_resnet_blocks_kernel_size':symmetric_resnet_blocks_kernel_size,
            'kwargs':kwargs
        }

        # Let the mode property's setter verify that input
        self.__forwards_by_mode = {
            'predict_hic':self.predict_hic,
            'embed_sequences':self.embed_sequences,
            'predict_hic_vector':self.predict_hic_vector
        }
        self.mode = mode
        
        ########
        # Now actually initialize EPCOT

        ####
        # Front components needed in embed_sequences mode
        # Just going to the project/cnn values hard-coded for now... 
        self.front_end = front_end
        self.attention_pool = AttentionPool(hidden_dim)
        self.project=nn.Sequential(
            Rearrange('(b n) c -> b c n', n=bins_per_sample*5), # Will coarsen to desired resolution
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=15, padding=7,groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        self.cnn=nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=15, padding=7),
            nn.BatchNorm1d(embed_dim),
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.Dropout(0.2),
            Rearrange('b c n -> b n c')
        )
        if trunk == 'transformer':
            # Default values
            encoder_kwargs = {
                'nhead':4,
                'dim_feedforward':2 * embed_dim,
                'batch_first':True,
                'norm_first':True
            }
            encoder_kwargs.update(trunk_kwargs.get('encoder',{}))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                **encoder_kwargs
            )
            
            transformer_kwargs = {'num_layers':3}
            transformer_kwargs.update(trunk_kwargs.get('transformer',{}))
            self.trunk = nn.TransformerEncoder(encoder_layer, **transformer_kwargs)
            
        else:
            lstm_kwargs = {
                'hidden_size':embed_dim//2, 
                'num_layers':4,
                'batch_first':True,
                'dropout':0.2,
                'bidirectional':True
            }
            lstm_kwargs.update(trunk_kwargs)
            # Already verified that it's either transformer or LSTM
            self.trunk = nn.LSTM(
                input_size=embed_dim, 
                **lstm_kwargs
            )

        ####
        # Additional components used to predict Hi-C maps
    
        self.output_head = OutputHead(
            bins_per_sample=bins_per_sample,
            num_embeddings=num_distance_embeddings,
            embedding_dim=embed_dim,
            dropout_prob=distance_embedding_dropout_prob
        )
        self.dilation_tower = DilationTower(
            in_channels=embed_dim,
            out_channels=dilation_tower_out_channels, 
            kernel_size=dilation_tower_first_last_kernel_size,
            symmetric_resnet_kernel_size=symmetric_resnet_blocks_kernel_size,
            dilations = kwargs.get('dilations',None),
            n_symmetric_resnet_blocks = n_symmetric_resnet_blocks,
            amount_to_crop=amount_to_crop,
            resnet_block_activation=kwargs.get('resnet_block_activation',nn.ReLU()),
            resnet_block_dropout_prob=kwargs.get('resnet_block_dropout_prob',0.1)
        )
        
        self.to_contacts = nn.Linear(dilation_tower_out_channels,1) # Previously named 'prediction_head' 

    #############
    # Allow the prediction mode to be changed on the fly
    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self,mode):
        assert isinstance(mode,str), f'EPCOT mode must be set by a string. Received {type(mode)}.'
        assert (forward:=self.__forwards_by_mode.get(mode)), \
        f'The selected mode, {mode}, is invalid. Valid options are: {list(self.__forwards_by_mode.keys())}.'
        self.__forward = forward
        self.__mode = self.__config['mode'] = mode

    def as_hic_predictor(self):
        self.mode = 'predict_hic'
        return self

    def as_hic_vector_predictor(self):
        self.mode = 'predict_hic_vector'
        return self
    
    def as_sequence_embedder(self):
        self.mode = 'embed_sequences'
        return self

    #############
    # Basic lookup properties
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    @property
    def dtype(self):
        for p in self.parameters():
            return p.dtype

    @property
    def config(self):
        return self.__config.copy()

    #############
    # Interact with files

    def save(self,filepath):
        fp = Path(filepath)
        assert not fp.is_dir(), f'Provided filepath, {filepath}, is a directory.'
        fp.parent.mkdir(exist_ok=True,parents=True)
        if (sfx:=fp.suffix) != '.pt':
            if sfx:
                warnings.warn(
                    f'The provided filepath uses the extension "{sfx}". This is '
                    'being changed to ".pt".'
                )
            else:
                warnings.warn('The provided filepath has no extension. ".pt" is being automatically added.')
            fp = fp.with_suffix('.pt')
            
        data = {
            'cnn_config':self.front_end.cnn_config,
            'transformer_config':self.front_end.transformer_config,
            'epcot_hic_config':self.config,
            'state_dict':self.state_dict()
        }
        
        torch.save(data,fp)

    # Convert the OrderedDict keys from a save file originating from the original EPCOT code 
    # to they keys needed in this class
    @staticmethod
    def __rename_param_keys(params: OrderedDict, desired_keys: list):
        desired_keys1 = [key.replace('front_end.','') for key in desired_keys if key[:10]=='front_end.']
    
        # EPCOTFrontEnd has a method to handle the key conversion for parameters within it, so just go ahead
        # and recycle that method. 
        params1 = EPCOTFrontEnd._EPCOTFrontEnd__rename_param_keys(params, desired_keys=desired_keys1)
    
        # We must add the 'front_end.' prefix to each of these keys to load the parameters within this instance. 
        for key in tuple(params1.keys()):
            params1[f'front_end.{key}'] = params1.pop(key)
    
        # Deal with the keys specific to the downstream model.
        for key,param in params.items():
            if key[:10] == 'pretrain_model.':
                # Handled by EPCOTFrontEnd.__rename_param_keys()
                continue
    
            # A number of parameters' keys are unchanged, so simply add them to params1 and continue
            #if any(key[:len(prefix)] for prefix in ['cnn.','project.','attention_pool.']):
            if key in desired_keys:
                params1[key] = param
                continue
    
            # Change a few that are easily fixed systematically
            for (prefix, desired_prefix, additional_changes) in [
                ('dilate_tower.cnn.','dilation_tower.',[('.conv.','.conv_block.')]),
                ('transformer.','trunk.',[]), # NOTE: Conversion not set up for the BiLSTM option
                ('prediction_head.','to_contacts.',[]),
                ('distance_embed.','output_head.distance_embedding.',[])
            ]:
                n = len(prefix)
                if key[:n] == prefix:
                    key = desired_prefix + key[n:]
                    for old, new in additional_changes:
                        key = key.replace(old, new)
                    params1[key] = param
        
        return params1
    
    def load(self,filepath_dict_or_ordered_dict):
        od = filepath_dict_or_ordered_dict
        sd = self.state_dict()
        match od:
            case OrderedDict():
                # Given the parameters ONLY. Likely loaded from file generated 
                # with original EPCOT code, so map the keys to those present
                # in the new code. 
                od = self.__rename_param_keys(od, list(sd.keys()))
            case dict():
                od = od['state_dict']
            case _:
                return self.load(torch.load(od),map_location=self.device)

        # This wasn't tracked as a parameter in the original EPCOT implementation, so slap a band-aid on it here
        # for backward compatibility. It isn't an optimized parameter, so it doesn't really matter. 
        key = 'output_head.distance_embedding_index'
        if any(key1 == key for key1 in sd.keys()):
            od[key] = od.get(key, sd[key])
        
        return self.load_state_dict(od)

    @staticmethod
    def from_dict(data,err_msg=None,mode=None):
        if not err_msg:
            err_msg = 'The provided dictionary appears to be malformed for an EPCOT instance.'

        od = data
        if isinstance(od, OrderedDict):
            # Should correspond to a file saved from the original EPCOT code, which should only include models
            # using the default __init__ arguments. 
            epcot = EPCOT(front_end = EPCOTFrontEnd())
        else:
            assert isinstance(od, dict), err_msg
            assert all(key in od for key in ['cnn_config','transformer_config','epcot_hic_config','state_dict']), err_msg
            epcot = EPCOT(
                front_end = EPCOTFrontEnd(
                    CNNBackbone_kwargs = od['cnn_config'],
                    Transformer_kwargs = od['transformer_config']
                ),
                **od['epcot_hic_config']
            )

        # By not changing od from dict in the first case, we skip self.__rename_param_keys() within epcot.load,
        # thereby saving a ~bit~ of computation time. 
        epcot.load(od)

        if mode is not None:
            try:
                epcot.mode = mode
            except Exception as e:
                warnings.warn(
                    f"Attempting to set EPCOT's mode to the selected value, {mode}, yielded the following error:"'\n'
                    '\t'f'{e}''\n'
                    f'Therefore, I am returning EPCOT in {epcot.mode} mode, which was determined using the information inside {filepath}.'
                )

        return epcot
    
    @staticmethod
    def from_file(filepath,mode=None):
        od = torch.load(filepath, map_location='cpu')
        err_msg = f'The provided file, {filepath}, appears to be malformed for an EPCOT instance.'
        return EPCOT.from_dict(od,err_msg,mode=mode)
    
    #############
    # Forward functions
    
    def forward(self,*args,**kwargs):
        # Set in __init__ (and possibly reset by self.mode's setter) as one of the
        # following methods. 
        return self.__forward(*args,**kwargs)

    def embed_sequences(self, x):
        # As in the front_end, the sequence embedding layers specific to EPCOT.embed_sequences
        # require an object with 2 or 3 dimensions (b/c Conv1d layers). 
        # Rather than flipping back-and-forth repeatedly, we'll just fix that here (and the operations
        # will be skipped in EPCOTFrontEnd.forward because it won't be relevant). 
        # ALSO, the 
        #'''
        assert isinstance(x,torch.Tensor), f'Input to {__class__.__name__}.forward() should be torch.Tensor object. Received {type(x)}.'
        ndim = x.ndim
        assert ndim >= 2, f'Input to {__class__.__name__}.forward() should have at least two dimensions, while your input has {ndim}.'
        n = self.front_end.cnn_config['in_channels']
        assert x.shape[-2] == n, f'Input to {__class__.__name__}.forward() should have size {n} in the second-to-last dimension, while your input has {x.shape[-2]}.'

        if ndim == 2:
            x = x.unsqueeze()
        elif ndim > 3:
            x_shape = x.shape
            x = x.reshape(torch.tensor(x_shape[:-2]).prod(), *x_shape[-2:])
        '''
        ...
        if ndim > 3:
            x = x.reshape(*x_shape[:-2],*x.shape[-2:])
        '''
        # Now actually run the layers. 
        x = self.front_end(x)
        x = self.attention_pool(x)
        x = self.project(x)
        x = self.cnn(x)
        return self.trunk(x)

    def predict_hic_vector(self, x):
        ###
        # Start by embedding the sequences to the extend done in ChromoGen
        x = self.embed_sequences(x)

        ###
        # Downstream layers to predict the Hi-C map 
        x = self.output_head(x)
        x = self.dilation_tower(x)
        
        # The following three lines reduce dimensions from (...batch dimensions..., N, N, M) to
        # (...batch dimensions..., N*(N+1)/2, M) by taking the upper triangle values, where
        # N = bins_per_sample - 2*amount_to_crop and M = dilation_tower_out_channels.
        # This replaces the following lines from the EPCOT repository:
        # x=rearrange(x,'b l n d -> b (l n) d')
        # x=self.upper_tri(x,self.bins-2*self.crop)
        n = x.shape[-2]
        i,j = torch.triu_indices(n,n,device=x.device)
        x = x[...,i,j,:]

        # Combine the channels from the DilationTower to predict the normalized contacts.  
        x = self.to_contacts(x)

        # Finally, 
        return x

    # This function isn't used for training or anything, so always run it in inference mode. 
    @torch.inference_mode()
    def predict_hic(self, x):
        # Temporarily place model in eval mode (turning off dropout layers, etc.) for the same reason this method runs in inference_mode
        was_training = self.training
        self.eval()
        x = self.predict_hic_vector(x)
        if was_training:
            self.train()

        # Place the vectorized result (which contains upper triangle == lower triangle data) in a square matrix and
        # simultaneously remove the pesky dimension left over from the last Linear layer, self.to_contacts. 
        n = round( ( 1 + (1 + 8*x.shape[-2])**.5 )/2 ) # Using round instead of // for cases where we may end up at x.999997 or something
        y = torch.empty(*x.shape[:-2],n,n)
        i,j = torch.triu_indices(n,n,device=x.device)
        y[...,i,j] = y[...,j,i] = x[...,:,0]
        return y
