from torch import Tensor, cdist
from torch import load as t_load
from torch.cuda import is_available as cuda_is_available
from ConformationsABC import ConformationsABC

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
            coords = Tensor(coords)
        except:
            raise Exception('Data must be convertible to torch.Tensor object, '+\
                            f'but the provided {t} object cannot be!')

    dimension_msg = f'Input shape {coords.shape} is not valid!\n'
    dimension_msg+= 'Dimensions should be one of the following:\n'
    dimension_msg+= '\t(3): One atom\'s xyz coordinates;\n'
    dimension_msg+= '\t(num_atoms,3); or\n'
    dimension_msg+= '\t(batch_dim1,...,batch_dimN,num_atoms,3): One atom\'s xyz coordinates;\n'
    assert coords.ndim > 0, dimension_msg
    assert coords.shape[-1] == 3, dimension_msg
    while coords.ndim < 3:
        coords = coords.unsqueeze(0)
    
    return coords

# Converting between subclasses
def coords_to_dists(coords,use_cuda=True,high_precision=True):

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

    def __init__(self,coords,drop_invalid_conformations=True,map_location='cpu'):

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
    def compute_rg(self,*args,**kwargs): 
        # arugments are for mdtraj's compute_rg function
        return self.trajectory.compute_rg(*args,**kwargs)
        
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

from Distances import Distances
from Trajectory import Trajectory
