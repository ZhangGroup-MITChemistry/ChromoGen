'''
Greg Schuette 2024
'''

from torch import Tensor, cdist, tensor
from torch import load as t_load
from torch.cuda import is_available as cuda_is_available
from ._ConformationsABC import ConformationsABC

# Initialization
def format_coords(coords):
    '''
    Also handle filepaths, trajectories
    '''
    t = type(coords)
    if issubclass(t,ConformationsABC):
        return coords.coordinates.values
    if t != Tensor:
        try:
            coords = tensor(coords)
        except:
            raise Exception('Data must be convertible to torch.Tensor object, '+\
                            f'but the provided {t} object cannot be!')

    dimension_msg = (
        f'Input shape {tuple(coords.shape)} is not valid!\n'
        'Dimensions should be one of the following:\n'
        '\t(3): One monomer\'s xyz coordinates;\n'
        '\t(num_atoms,3): One homopolymer\'s coordinate vector; or\n'
        '\t(batch_dim1,...,batch_dimN,num_atoms,3): '
            'Multiple equal-length homopolyers\' coordinate vectors.'
    )
    assert coords.ndim > 0, dimension_msg
    assert coords.shape[-1] == 3, dimension_msg
    while coords.ndim < 3:
        coords = coords.unsqueeze(0)
    
    return coords

# Converting between subclasses
def coords_to_dists(coords,use_cuda=False,high_precision=True):

    # Handle device, precision details
    return_device = coords.device
    return_dtype = coords.dtype
    if use_cuda and cuda_is_available():
        coords = coords.cuda()
    if high_precision:
        coords = coords.double()

    # Compute the distances
    dists = cdist(coords,coords)

    # cdist has slight numerical errors making it slightly asymmetric and often placing
    # errant, small-but-nonzero values along the diagonal. Fix that here. 
    dists = (dists + dists.transpose(-2,-1))/2
    i = range(dists.shape[-1])
    dists[...,i,i] = 0

    return dists.to(dtype=return_dtype,device=return_device)

class Coordinates(ConformationsABC):

    def __init__(self,coords,drop_invalid_conformations=False,map_location='cpu'):

        if type(coords) == str: # Assume it's a filepath
            self.__coords = format_coords(t_load(coords,map_location=map_location))
        else:
            self.__coords = format_coords(coords)

        if drop_invalid_conformations:
            self.drop_invalid_conformations_()
        
    
    ########################################################
    # Properties 

    @property
    def _values(self):
        return self.__coords

    @_values.setter
    def _values(self,c):
        self.__coords = c

    ########################################################
    # Operations
    def drop_invalid_conformations_(self):
        self.__coords = self.values[self.values.isfinite().all(-1).all(-1)]
        return self

    def drop_invalid_conformations(self):
        return self.clone().drop_invalid_conformations_()
        

    ########################################################
    # Some coordinate-specific functionality
    def compute_rg(
        self,
        *args,            # positional arguments for MDtraj's function
        use_mdtraj=False, # Whether to use MDTraj's compute_rg function
        optimization_safe=False, # If using this torch implementation, 
                                 # whether to ensure .backward will work later
        **kwargs          # Keyword arguments for MDtraj's function
    ):
        assert isinstance(use_mdtraj,bool), \
        f'use_mdtraj must be a bool. Received {type(use_mdtraj).__name__}'
        if use_mdtraj:
            # arugments are for mdtraj's compute_rg function
            return self.trajectory.compute_rg(*args,**kwargs)
        else:
            assert isinstance(optimization_safe,bool), \
            f'optimization_safe must be a bool. Received {type(optimization_safe).__name__}'
            
            v = self.values
            n = self.num_beads
            if optimization_safe:
                return ((v - v.mean(-2,keepdim=True)).square().sum((-1,-2)) / n).sqrt()
            else:
                # Otherwise, reduce memory usage/improve speed with some in-place operations
                return ((v - v.mean(-2,keepdim=True)).square_().sum((-1,-2)) / n).sqrt_()


    def center_coordinates_(self):
        self._values-= self.values.mean(-2,keepdim=True)
        return self
    
    def center_coordinates(self):
        return self.clone().center_coordinates_()

    def save_dcd(self,*args,**kwargs):
        self.trajectory.save_dcd(*args,**kwargs)

    def superpose(self,reference,*args,use_mdtraj=False,**kwargs):
        assert isinstance(use_mdtraj,bool), \
        f'use_mdtraj must be a bool. Received {type(use_mdtraj).__name__}.'

        if use_mdtraj:
            return self.trajectory.superpose(reference,*args,**kwargs).coordinates
        else:
            raise Exception('My PyTorch superpose implementation is not yet complete. Please pass use_mdtraj=True.')

    def rmsd(self,reference,*args,use_mdtraj=False,**kwargs):
        assert isinstance(use_mdtraj,bool), \
        f'use_mdtraj must be a bool. Received {type(use_mdtraj).__name__}'

        if use_mdtraj:
            return self.trajectory.rmsd(reference,*args,**kwargs)

        v = self.values
        v1 = Coordinates(reference,drop_invalid_conformations=False).values.to(device=v.device,dtype=v.dtype)
        while v.ndim < v1.ndim:
            v = v.unsqueeze(0)
        while v1.ndim < v.ndim:
            v1 = v1.unsqueeze(0)
        return (v-v1).square().sum(-1).mean(-1).sqrt()
    
    ########################################################
    # Converting between sample subclasses
    
    # Always be able to return coordinates object
    @property
    def coordinates(self): 
        return self

    # Always be able to return trajectory object
    @property
    def trajectory(self): 
        return Trajectory(self.values)

    # Always be able to return distance maps 
    @property
    def distances(self): 
        return Distances(coords_to_dists(self.values))

from ._Distances import Distances
from ._Trajectory import Trajectory
