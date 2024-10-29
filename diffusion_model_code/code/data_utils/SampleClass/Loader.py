from mdtraj import load as md_load
from Sample import Sample 

########################################################
# Support functionality
########################################################

# To initialize objects
def format_and_infer_class(input,*args,**kwargs):

    if type(input) in [str,md.Trajectory] or (type(input) == list and len(input)>0 and type(input[0])==str):
        return Trajectory(input,*args,**kwargs)
    if type(input) != torch.Tensor:
        try:
            input = torch.tensor(input)
        except:
            raise Exception('The input must be convertible to a torch.tensor '+\
                            f'object, but input of type {type(input)} is not!')
    if input.ndim == 0:
        raise Exception('The input must have non-zero dimensionality!')

    err_msg = f'Invalid input dimensions ({input.shape}). '+\
    'Should be square in final two dimensions for distance maps or have '+\
    'input.shape[-1]==3 for coordinates.'
    
    if input.ndim == 1:
        assert input.numel() in [1,3], err_msg
    
    while input.ndim < 3:
        input = input.unsqueeze(0)

    if input.shape[-1] != input.shape[-2]:
        assert input.shape[-1] == 3, err_msg
        return Coordinates(input)
    elif input.shape[-1] == 3:
        # Ambiguous case. 
        # Check the diagonal. If all zeros, assume distance map
        diag = input[...,torch.arange(3),torch.arange(3)]
        if torch.allclose(diag,torch.zeros_like(diag)):
            return Distances(input)
        return Coordinates(input)
    return Distances(input)

class Loader:

    def __init__(
        
    ):

    