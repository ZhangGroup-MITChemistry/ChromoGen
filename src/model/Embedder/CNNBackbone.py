'''
Rewrite of the EPCOT CNN "backbone" to remove all the hardcoding, simplify hyperparameter modifications.

Greg Schuette 2024
'''

from torch import nn
from types import NoneType

class CNNBlock(nn.Sequential):

    def __init__(
        self,
        # Conv1d layer
        in_channels,
        out_channels,
        kernel_size,
        # Activation function. Always comes immediately after the Conv1d layer in EPCOT
        activation,
        activate_inplace,
        # Max pool layer, if relevant 
        max_pool_kernel_size = None, # If None, no max pooling layer. Otherwise, need an integer. 
        # Batch norm layer, if relevant
        include_batch_norm=True,
        # Dropout layer
        dropout_prob=.1, 
    ):
        super().__init__()
        activation = activation if isinstance(activation,nn.Module) else getattr(nn,activation)
        self.extend([
            nn.Conv1d(in_channels, out_channels, kernel_size = kernel_size),
            activation(inplace=activate_inplace),
            nn.Dropout(p=dropout_prob),
            nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size),
            activation(inplace=activate_inplace)
        ])
        
        if max_pool_kernel_size:
            self.append(
                nn.MaxPool1d(kernel_size=max_pool_kernel_size,stride=max_pool_kernel_size)
            )
        if include_batch_norm:
            self.append(nn.BatchNorm1d(out_channels))
        self.append(nn.Dropout(p=dropout_prob))
        

class CNNBackbone(nn.Sequential):

    def __init__(
        self,
        in_channels=5,
        out_channels=(256,360,512),       # int would mean one layer
        conv_kernel_sizes=(10,8,8),       # int would mean "use same value in each block"
        activation = 'ReLU', 
        activate_inplace=True,
        max_pool_kernel_sizes=(5,4,None), # None means no max pooling layer in a given block. One value means "use same throughout"
        include_batch_norm=True,
        dropout_probs=(.1,.1,.2),         # float would mean "use same dropout probability in each block
    ):
        super().__init__()

        #################
        # Validate inputs

        # in_channels
        assert isinstance(in_channels,int), f'in_channels must be an integer. Recieved {type(in_channels)}.'
        assert in_channels > 0, f'in_channels must be positive-valued. Received {in_channels}.'

        # Validate/format all the tuple inputs 
        args = {
            'out_channels':(out_channels,int),
            'conv_kernel_sizes':(conv_kernel_sizes,int),
            'activation':(activation,(str,nn.Module)),
            'activate_inplace':(activate_inplace,bool),
            'max_pool_kernel_sizes':(max_pool_kernel_sizes,(int,NoneType)),
            'include_batch_norm':(include_batch_norm,bool),
            'dropout_probs':(dropout_probs,float)
        }

        for arg_name, (value, valid_dtypes) in args.items():
            err_msg = f'{arg_name} must be '
            if isinstance(valid_dtypes,tuple):
                err_msg+= f'{valid_dtypes[0]}, {valid_dtypes[1]}, '
            else:
                err_msg+= f'{valid_dtypes}, '
            err_msg+= 'a tuple of them, or a list of them. '

            if isinstance(value,valid_dtypes):
                value = (value,)
            elif isinstance(value,(tuple,list)):
                assert value, err_msg + f'However, the provided {arg_name} {type(value)} is empty.'
                value = tuple(value)
                invalid_types = {type(v) for v in value if not isinstance(v,valid_dtypes)}
                assert not invalid_types, err_msg + f'Provided value contains invalid type(s) {invalid_types}.'
            else:
                raise Exception(err_msg + f'Received {type(out_channels)}.')

            args[arg_name] = value
        
        # Validate the length of all other tuple inputs
        out_channels = args['out_channels']
        nblocks = len(out_channels)
        for arg_name, value in args.items():
            if len(value) == 1:
                args[arg_name]*= nblocks
            else:
                assert len(value) == nblocks, (
                    'The number of blocks, inferred as number of elements in out_channels, is '
                    f'{nblocks}, but {arg_name} has {len(value)} elements. This should either be 1 -- '
                    f'indicating that the same value should be used throughout -- or {nblocks}.'
                )

        #################
        # Place the formatted info in a config dict
        self.config = args.copy()
        self.config['in_channels'] = in_channels

        #################
        # Build the convolutional frontend
        c_in = in_channels
        for k,co in enumerate(out_channels):
            self.append(
                CNNBlock(
                    in_channels=c_in,
                    out_channels=co,
                    kernel_size=args['conv_kernel_sizes'][k],
                    activation=args['activation'][k],
                    activate_inplace=args['activate_inplace'][k],
                    max_pool_kernel_size=args['max_pool_kernel_sizes'][k],
                    include_batch_norm=args['include_batch_norm'][k], 
                    dropout_prob=args['dropout_probs'][k]
                )
            )
            c_in = co