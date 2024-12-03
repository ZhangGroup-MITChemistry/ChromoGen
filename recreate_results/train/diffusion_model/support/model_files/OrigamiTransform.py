import torch

def transpose_minor_axis(tensor):
    return tensor.flip(-1).transpose(-2,-1).flip(-1)
    
class OrigamiTransform:

    #def __init__(self,nbeads,includes_self_interaction=False):
    '''
    Input size: (optional, batch etc)xCxHxW
    H == W, always

    Full map (Input to __call__, output of inverse): 
    C == 1
    H == nbeads

    Condensed map (output of __call__, input to inverse): 
    C == 2
    H == nbeads//2 

    NOTE: The first half of the main diagonal is lost in the submap. Upon reconstruction, 
    the second half of the main diagonal is reflected to fill in the first half. As such, 
    the main diagonal must contain unimportant values to avoid issues; this is generally true 
    when speaking of self-interaction values, so it's ok for our purposes. 
    '''
        
    def __call__(self,full_map):
        '''
        Input size: (optional dims, batch, channels, etc)x(im_size 1 == full_map.shape[-1]//2 (nbeads-1))
        '''
        ####
        # Verify input 
        assert len(full_map.shape) > 1, f'full_maps must have at least 2 dimensions since OrigamiTransform processes 2D data.'
        assert full_map.shape[-2] == full_map.shape[-1], f'Final two dimensions of full_maps should be equal. Receieved {full_map.shape}'
        assert full_map.shape[-1]%2 == 0, f'OrigamiTransform only operates on maps with even dimensions. Received shape {full_map.shape}'

        ####
        # Add dimension for channels if one wasn't provided
        if len(full_map.shape) == 2: 
            full_map = full_map.unsqueeze(0)
        if full_map.shape[-3] != 1: 
            full_map = full_map.unsqueeze(-3)

        ####
        # Perform the transformation. 

        # Infer useful information 
        n = full_map.shape[-1]//2 # Starting position for further elements.
        optional_dims = full_map.shape[:-3]
        i,j = torch.triu_indices(n,n,0)

        submap = torch.empty(*optional_dims,2,n,n,dtype=full_map.dtype,device=full_map.device)

        # Data folded under
        submap[...,1,i,j] = full_map[...,0,i,j] # Top left of preserved portion of the full map, folded right
        submap[...,1,j,i] = transpose_minor_axis(full_map)[...,0,i,j] # Bottom of preserved portion, folded up
        # Note: the digonal from the bottom region overwrites the diagonal from the upper region. 

        # Flip the last dimension so that the final arrangement resembles the triangular regions folding behind the square region
        submap = submap.flip([-1]) 

        # Top data
        submap[...,0,:,:] = full_map[...,0,:n,-n:]

        return submap

    def inverse(self,submap):
        assert len(submap.shape) > 2, f'Must have at least 3 dimensions in submap. Received {submap.shape}'
        assert submap.shape[-2]==submap.shape[-1], f'Last two dimensions of submap must be equal in size. Received {submap.shape}'
        assert submap.shape[-3]==2, f'Third-to-last dimension in submap should be size 2 (2 channels), but receieved {submap.shape[-3]}'

        # Useful info 
        n = submap.shape[-1]
        N = 2*n
        optional_dims = submap.shape[:-3]

        # Initialize the full map 
        full_map = torch.zeros(*optional_dims,1,N,N,dtype=submap.dtype,device=submap.device)

        # Flip the last dimension to simplify indexing when handling the folded regions
        submap = submap.flip([-1])

        # Fill in the triangular edge regions
        i,j = torch.triu_indices(n,n,0) 
        full_map[...,0,i,j] = submap[...,1,j,i] # Lower right portion
        full_map = transpose_minor_axis(full_map) # Place it in the lower right
        full_map[...,0,i,j] = submap[...,1,i,j]

        # Flip the last dimension back to its original position
        submap = submap.flip([-1])

        # Upper right quadrant
        full_map[...,0,:n,-n:] = submap[...,0,:,:]

        # Symmetrize the matrices
        full_map+= full_map.clone().transpose(-2,-1)

        # The diagonal was double during symmetrization, so divide it by 2
        i = torch.arange(N)
        full_map[...,i,i]/= 2
        
        return full_map 