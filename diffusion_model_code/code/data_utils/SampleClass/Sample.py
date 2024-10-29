'''
TODO: 
- to_dcd (& other formats)
- Accept filepath input during initialization
- from_data in Sample class
- to_mdtraj -- in the superclass AND Trajectory classes
- HiC
- scHiC
- from_file is probably better for loading (actually... load is fine... but have it call the md filepath function if needed)
- There's more, but I'm short on time rn
'''

################################################################################################################
# Sample ABC
################################################################################################################

import torch
import numpy as np
import copy
from abc import ABC, abstractmethod
import mdtraj as md

########################################################
# Support functionality

# Basics
def prod(input):
    return torch.tensor([*input]).prod()

def get_type(obj):
    if type(obj) == type:
        pass
    return str(type(obj)).split('.')[-1].strip("'>").lower()

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

########################################################
# Main abstract class
class Sample(ABC):

    __version__ = '0.0.0'

    ########################################################
    # Equivalent of initialization
    @classmethod
    def from_data(cls,input):

        if issubclass(type(input),Sample):
            return input

        return format_and_infer_class(input)
        
    ########################################################
    # Properties 

    # Version info
    @property
    def version(self):
        return self.__version__
    
    # Access the configuration data
    @property
    def values(self): 
        return self._values

    @property
    @abstractmethod
    def _values(self):
        pass

    @_values.setter
    @abstractmethod
    def _values(self):
        pass

    # Get the name of the class type
    @property
    def __type(self,obj=None):
        return get_type(self)
    
    # Report the number of beads
    @property
    def num_beads(self): 
        return self.values.shape[-2]

    # Report device & dtype
    @property
    def dtype(self):
        return self.values.dtype

    @property
    def device(self):
        return self.values.device

    @property
    def shape(self):
        return self.values.shape
    
    ########################################################
    # Standard functionality 

    def __getitem__(self,i):
        return type(self)(self.values[i])

    def __repr__(self):
        #t = str(type(self)).split('.')[-1].strip("'>").lower()
        t = self.__type
        return str(self.values).replace('tensor',t).replace('array',t)

    def __len__(self):
        '''
        Number of unique conformations... so, product of all batch dimensions!
        '''
        return prod(self.shape[:-2])

    def __eq__(self,obj):
        t = get_type(obj)
        self_vals = self.values
        if issubclass(type(obj),Sample):
            if obj.num_beads != self.num_beads:
                return False

            # If one is a distance map but the other is not, we should convert
            # both samples to a distances representation to remove effects of 
            # translation, rotation, etc. 
            # If both are distances maps, we obviously also want to use distances... 
            if t == 'distances' or self.__type == 'distances':
                self_vals = self.distances.values
                obj = obj.distances.values.to(dtype=self.dtype,device=self.device)

                # Try torch.allclose, which is fastest if it works & provides True
                if torch.allclose(self_vals,obj):
                    return True
                # In this case, torch.allclose doesn't work as expected due to numerical challenges
                # during the conversion of coordinates into 
            
            else:
                # Both Trajectory and Coordinates objects return coordinates by default. 
                # If coordinates were provided, assume that translations, rotations, etc.,
                # are relevant for comparison, so don't convert to distances (which removes this information)
                self_vals = self.values
                obj = obj.values
        elif t == 'ndarray':
            obj = torch.from_numpy(obj)
        elif t != 'tensor':
            return False

        # Object MUST be a tensor if this point is reached. 
        obj = obj.to(dtype=self.dtype,device=self.device)
        return torch.allclose(obj,self_vals)

    
    # Change the datatype & device types, etc., as in pytorch 
    def to(self,*args,**kwargs):
        return type(self)( self.values.to(*args,**kwargs) )

    def cuda(self):
        return type(self)( self.values.cuda() )

    def cpu(self):
        return type(self)( self.values.cpu() )

    def float(self):
        return type(self)( self.values.float() )

    def double(self):
        return type(self)( self.values.double() )
    
    def to_(self,*args,**kwargs):
        self._values = self._values.to(*args,**kwargs)

    def cuda_(self):
        self._values = self._values.cuda()

    def cpu_(self):
        self._values = self._values.cpu()

    def float_(self):
        self._values = self._values.float()

    def double_(self):
        self._values = self._values.double()
        
    def clone(self):
        return copy.deepcopy(self)

    ########################################################
    # Combining data from multiple objects
    def __parse_trajectory_kwargs(self,**kwargs):
        cat_stack_args = {}
        trajectory_args = {}
        for key,item in kwargs.items():
            if key in ['num_beads','topology_file','topology']:
                trajectory_args[key] = item
            else:
                cat_stack_args[key] = item
        return cat_stack_args, trajectory_args
    
    def __format_other_object(self,other_object,**kwargs):
        return_kwargs,kwargs = self.__parse_trajectory_kwargs(**kwargs)
        
        if not issubclass(type(other_object),Sample):
            # will raise exception if invalid non-Sample input
            other_object = format_and_infer_class(other_object,**kwargs)
        t = self.__type
        t1 = get_type(other_object)
        if t!=t1:
            if t in ['trajectory','coordinates']:
                # Only used for the stack/cat functions, which 
                # would otherwise cause trajectory conversion to be 
                # converted back and forth a few times. 
                other_object = other_object.coordinates
            else:
                other_object = getattr(other_object,t)
        return return_kwargs, other_object.to(dtype=self.dtype,device=self.device)

    def stack(self,other_object,*args,**kwargs):
        kwargs,other_object = self.__format_other_object(other_object,**other_kwargs)
        data = torch.stack(
            [
                self.values,
                other_object.values
            ],
            *args,
            **kwargs
        )
        return type(self)(data)   

    def __cat(self,other_object,*args,**kwargs):
        kwargs,other_object = self.__format_other_object(other_object,**kwargs)
        data = torch.cat(
            [
                self.values,
                other_object.values
            ],
            *args,
            **kwargs
        )
        return type(self)(data)  
    
    def append(self,other_object,*args,**kwargs):
        return self.__cat(other_object,dim=0,*args,**kwargs)

    ########################################################
    # Converting between sample subclasses
    
    # Always be able to return coordinates object
    @property
    @abstractmethod
    def coordinates(self): 
        pass

    # Always be able to return trajectory object
    @property
    @abstractmethod
    def trajectory(self): 
        pass

    # Always be able to return distance maps 
    @property
    @abstractmethod
    def distances(self): 
        pass
    
    ########################################################
    # Preserving data
    def save(self,filepath,*args,**kwargs):

        # Note to know which class to place the data in when loaded
        load_type = self.__type
        
        # To save memory, save in coordinates format if reasonable. 
        # For Distances objects, this means reducing from NxN maps to Nx3 coordinates. 
        # For Trajectory objects, this means dropping the topology data, which can be easily
        # reconstructed. 
        if (load_type == 'distances' and self.is_exact()) or load_type == 'trajectory':
            # 
            save_obj = self.coordinates()
        else:
            save_obj = self

        save_data = {
            '__version__':self.__version__,
            'load_type':load_type,
            'save_type':get_type(save_obj),
            'data':save_obj.values
        }

        torch.save(save_data,filepath,*args,**kwargs)
    
    @classmethod
    def load(cls,filepath,return_class=None,map_location=None,num_beads=None,topology=None,*args,**kwargs):
        '''
        ::filepath:: Path to a file containing either a tensor recognizable as coordinates/distance maps, 
            a filepath recognized by a molecular filepath type recognized by mdtraj, or 
        
        ::return_class:: Class to use when returning the data. 
            Either directly a class or a string denoting it as 'distances', 'trajectory', or 'coordinates'.

        
        ::map_location:: Same as in PyTorch's torch.load()
        :::args:: For torch.load()
        ::kwargs:: for torch.load()
        '''
        ###########
        # User selections
        
        # Ensure we won't run into device clashes
        if map_location is None and not torch.cuda.is_available():
            map_location = 'cpu'

        # Determine which class to return
        if return_class is not None:
            return_class = get_type(str(return_class))
            
        # Load data
        data = torch.load(filepath,map_location=map_location,*args,**kwargs)

        # Determine the format in which data was saved and the format to return to the user. 
        load_type = data['load_type'] if return_class is None else return_class
        version,load_type1,save_type,data = data['__version__'],data['load_type'],data['save_type'],data['data']

        # At present, data may only be saved as distances or 
        if save_type == 'coordinates':
            obj = Coordinates(data)
        else:
            obj = Distances(data)

        # Convert to the class type requested by the user
        if load_type != save_type:
            obj = getattr(obj,load_type)()

        return obj

    @classmethod
    def from_md_filetypes(cls,dcd_filepath,*,num_beads=None,topology_file=None,topology=None):
        return Trajectory.from_dcd(dcd_filepath,num_beads=num_beads,topology_file=topology_file,topology=topology)

################################################################################################################
# Coordinates
################################################################################################################

# Initialization
def format_coords(coords):
    '''
    Also handle filepaths, trajectories
    '''
    t = type(coords)
    if issubclass(t,Sample):
        return coords.coordinates.values
    if t != torch.Tensor:
        try:
            coords = torch.Tensor(coords)
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
    if use_cuda and torch.cuda.is_available():
        coords = coords.cuda()
    if high_precision:
        coords = coords.double()

    # Compute the distances
    dists = torch.cdist(coords,coords)

    # cdist has slight numerical errors making it slightly asymmetric and often placing
    # errant, small-but-nonzero values along the diagonal. Fix that here. 
    dists = (dists + dists.transpose(-2,-1))/2
    i = range(dists.shape[-1])
    dists[...,i,i] = 0

    return dists.to(dtype=return_dtype,device=return_device)

class Coordinates(Sample):

    def __init__(self,coords):

        self.__coords = format_coords(coords)
    
    ########################################################
    # Properties 

    @property
    def _values(self):
        return self.__coords

    @_values.setter
    def _values(self,c):
        self.__coords = c
        
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
    

################################################################################################################
# Trajectory
################################################################################################################

import pandas as pd
import mdtraj as md
import numpy as np
def homopolymer_topology(num_monomers):
    n = num_monomers
    atoms = pd.DataFrame.from_dict({
        'serial':n*[None],
        'name':n*['C'],
        'element':n*['VS'],
        'resSeq':n*[1],
        'resName':n*['POL'],
        'chainID':n*[0],
        'segmentID':n*['POL']
    })
    bonds = np.zeros((n-1,4))
    bonds[:,0] = range(n-1)
    bonds[:,1] = range(1,n)
    return md.Topology.from_dataframe(atoms,bonds)

def coords_to_traj(coords):

    if type(coords) == torch.Tensor:
        coords = coords.cpu().numpy()
    elif type(coords) != np.ndarray:
        try:
            coords = np.array(coords)
        except:
            raise Exception('Invalid input! Must be convertible to an np.ndarray object, '+\
                            f'but the provided {type(coords)} object is not!')
    if coords.ndim > 3:
        return_shape = coords.shape
        coords = coords.reshape(np.prod(coords.shape[:-2]),*coords.shape[-2:])
    else:
        return_shape = None
    
    topology = homopolymer_topology(coords.shape[-2])

    return md.Trajectory(coords,topology), return_shape

def file_to_trajectory(filepath,*,num_beads=None,topology_file=None,topology=None):
    '''
    Topology takes precedence over topology_file takes precedence over num_beads if more than 
        one is provided. 

    (following sentence needs editing)
    topology may be either mdtraj.Topology file, a path to a topology-containing file, or
    an integer indicating the number of beads, where we'll assume a homopolymer structure. 

    filepath is EITHER a string filepath to .dcd (or similar) files containing molecular structures
    OR a list of multiple such strings. 
    '''
    topology = choose_topology(num_beads,topology_file,topology)
    return md.load(filepath,top=topology)

def choose_topology(num_beads=None,topology_file=None,topology=None):
    '''
    Topology takes precedence over topology_file takes precedence over num_beads if more than 
        one is provided. 
    
    '''
    if topology is not None:
        assert type(topology) == md.Topology, f"topology argument should be of type md.Topology. Received {type(topology)}."
        return topology
    if topology_file is not None:
        assert type(topology_file) == str, f"topology_file argument should be of type md.Topology. Received {type(topology)}."
        return topology_file
    assert num_beads is not None, "Must provide num_beads, topology_file, or topology input!"
    if type(num_beads) != int:
        try:
            n = int(num_beads)
            if n == num_beads:
                num_beads == n
            else:
                asdf
        except:
            raise Exception(f"num_beads argument should be an integer, but received type {type(num_beads)}.")
    return homopolymer_topology(num_beads)
    

class Trajectory(Sample):

    def __init__(
        self,
        input,
        *,
        num_beads=None,
        topology_file=None,
        topology=None,
        dtype=None,
        device=None
    ):
        '''
        Topology takes precedence over topology_file takes precedence over num_beads if more than 
        one is provided. 
        
        num_beads is used to construct a topology object if a dcd (or similar) filepath is provided
        but no topology object or topology_file is provided. Otherwise, ignored. 

        If no keyword arguments are provided and input has xyz coordinates, then assume homopolymer
        to construct topology object of our own. 

        FOR FUTURE, should catch the exception output from called functions and compile a message
        that refers to what this initializer can accept rather than what those other functions can accept. 
        '''

        ########################################################
        # Format dtype, device for returns via values property
        if dtype is None:
            if type(input) == torch.Tensor:
                dtype = input.dtype
            elif type(input) == np.ndarray:
                dtype = torch.from_numpy(input.flatten()[:1]).dtype
            else:
                dtype = torch.float
        elif type(dtype) != torch.dtype:
            # Ensures a valid dtype was passed, as well as transforms it
            # into a torch dtype if a non-torch dtype was passed
            dtype = torch.empty(1).to(dtype=dtype).dtype
        self.__dtype = dtype

        if device is None:
            if type(input) == torch.Tensor:
                device = input.device
            else:
                device = torch.device('cpu')
        elif type(device) != torch.device:
            # Ensures a valid device type was passed, as well as transforms it
            # into a torch device object if something else was passed
            device = torch.empty(1).to(device=device).device
        self.__device = device

        # Used when returning the data values to maintain consistency with other classes
        self.__return_shape = None
        
        ########################################################
        # Construct the trajectory
        if type(input) == md.Trajectory:
            self.__trajectory = input
            return

        if type(input) == str or (type(input)==list and len(input)>0 and type(input[0])==str):
            self.__trajectory = file_to_trajectory(
                input,num_beads=num_beads,topology_file=topology_file,topology=topology
            )
            return

        self.__trajectory,self.__return_shape = coords_to_traj(input)

    '''
    Should remove...
    '''
    @classmethod
    def from_dcd(
        cls,
        input,
        *,
        num_beads=None,
        topology_file=None,
        topology=None,
        dtype=None,
        device=None
    ):
        return Trajectory(
            input,
            num_beads=num_beads,
            topology_file=topology_file,
            topology=topology,
            dtype=dtype,
            device=device
        )

    ########################################################
    # Properties 
    # Most of these affect the _values object, which returns a tensor
    # even though the trajectory is based on NumPy arrays
    
    # Access the configuration data
    @property
    def _values(self):
        return torch.from_numpy(
            self.__trajectory.xyz.reshape(self.__return_shape)
        ).to(
            device=self.__device,
            dtype=self.__dtype
        )

    @_values.setter
    def _values(self,c):
        raise Exception('Trajectory object\'s _values cannot be altered!')

    ########################################################
    # Overriding some functions from the superclass for efficiency. 

    # Report the number of beads
    @property
    def num_beads(self): 
        return self.__trajectory.n_atoms

    # Report device & dtype
    @property
    def dtype(self):
        return self.__dtype

    @property
    def device(self):
        return self.__device

    @property
    def shape(self):
        return self.__trajectory.xyz.shape
    
    # Change the datatype & device types, etc., as in pytorch 
    def to(self,*,dtype=None,device=None):
        if dtype is None and device is None:
            return self
        return self.clone().to_(dtype=dtype,device=device)

    def cuda(self):
        if self.dtype.type == 'cuda':
            return self
        return self.clone().cuda_()

    def cpu(self):
        if self.dtype.type == 'cpu':
            return self
        return self.clone().cpu_()

    def float(self):
        if self.dtype == torch.float:
            return self
        return self.clone().float_()

    def double(self):
        if self.dtype == torch.double:
            return self
        return self.clone().double_()
    
    def to_(self,*,dtype=None,device=None):
        if dtype is not None:
            self.__dtype = torch.empty(1).to(dtype=dtype).dtype
        if device is not None:
            self.__device = torch.empty(1).to(device=device).device
        return self
        
    def cuda_(self):
        self.__device = torch.device('cuda')
        return self

    def cpu_(self):
        self.__device = torch.device('cpu')
        return self

    def float_(self):
        self.__dtype = torch.float

    def double_(self):
        self.__dtype = torch.double
        
    ########################################################
    # Functionality that actually uses the mdtraj package
    ########################################################
    '''
    These functions take args & kwargs from the corresponding mdtraj function
    AND kwargs relevant to this Trajectory class's initializer (for handling input).
    '''
    # Obtain mdtraj Trajectory from provided reference data
    def __prepare_trajectory(self,reference,**kwargs):
        
        if type(reference) == md.Trajectory:
            return reference, kwargs
            
        if issubclass(type(reference),Sample):
            return reference.trajectory.__trajectory, kwargs
            
        kwargs1 = {}
        kwargs2 = {}
        for key,item in kwargs.items():
            if key in ['num_beads','trajectory','trajectory_filepath']:
                kwargs1[key] = item
            else:
                kwargs2[key] = items
                
        reference = Trajectory(reference,**kwargs1).__trajectory
        return reference, kwargs2

    # The following are named the same as in the mdtraj documentation
    def superpose(self,reference,*args,**kwargs):
        reference, kwargs = self.__prepare_trajectory(reference,**kwargs)
        self.__trajectory.superpose(reference,*args,**kwargs)
        return self

    def rmsd(self,reference,*args,**kwargs):
        reference, kwargs = self.__prepare_trajectory(reference,**kwargs)
        return md.rmsd(self.__trajectory,reference,*args,**kwargs)

    ########
    # The following use the above mdtraj functions to align 3D conformations
    # with reference data & select the conformations that best match the 
    # reference data for visualization purposes. 
    def __best_superposition(self,reference_frame):
        self.superpose(reference_frame)
        rmsds = self.rmsd(reference_frame)
        best_score = rmsds.min()
        best_idx = np.where(rmsds == best_score)[0]
        return self.__trajectory[best_idx].xyz.copy(),best_idx,best_score
        

    def get_best_superpositions(self,reference,**kwargs):
        '''
        ONLY provide kwargs for the initializer. 
        '''
        reference,_ = self.__prepare_trajectory(reference,**kwargs)
        best_scores = []
        best_indices = []
        best_frames = []
        for i,frame in enumerate(reference):
            fr,idx,score = self.__best_superposition(frame)
            best_frames.append(fr)
            best_indices.append(idx)
            best_scores.append(score)
        best_info = {
            'RMSD':torch.tensor(best_scores),
            'Index':torch.tensor(best_indices)
        }
        best_frames = Trajectory(np.concatenate(best_frames,axis=0))
        return best_info,best_frames
    
    ########################################################
    # Converting between sample subclasses
    
    # Always be able to return coordinates object
    @property
    def coordinates(self): 
        return Coordinates(self.values)

    # Always be able to return trajectory object
    @property
    def trajectory(self): 
        return self

    # Always be able to return distance maps 
    @property
    def distances(self):
        return self.coordinates.distances


################################################################################################################
# Distances
################################################################################################################

## NEED is_exact function for the saving function! 

########################################################
# Analytically convert distance maps into coordinates
def dists_to_coords(dists,use_gpu=True,high_precision=True,num_dimensions=3):

    # Use high-precision values throughout calculation, but return same dtype as provided
    # Same for device
    return_dtype = dists.dtype
    return_device = dists.device
    
    if high_precision:
        dists = dists.double()
    if use_gpu and torch.cuda.is_available():
        dists = dists.cuda()
    ''' # Specified device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    '''

    # Initialize the object to hold coordinates
    coords = torch.zeros_like(dists[...,:num_dimensions])

    # Keep track of reference indices
    reference_indices = []

    for _ in range(num_dimensions): 
        compute_new_dimension(coords,dists,reference_indices)

    return coords.to(dtype=return_dtype,device=return_device)
    
def x_dot_y(x_norm,y_norm,x_minus_y_norm):
    # From known vector norms
    y_norm = y_norm.expand_as(x_norm)
    return (x_norm**2 + y_norm**2 - x_minus_y_norm**2) / 2

def select_new_indices(dist_from_origin,coords):
    dist_not_accounted_for = (dist_from_origin.square() - coords.square().sum(-1,keepdim=True))

    # Numerical precision occasionally causes small negative values to appear... to avoid NaN results, set those to 0!
    dist_not_accounted_for[dist_not_accounted_for<0] = 0
    dist_not_accounted_for.sqrt_()
    # Select maximum value to minimize numerical errors with division on this value later
    #return dist_not_accounted_for.max(-2,keepdim=True)
    return dist_not_accounted_for.median(-2,keepdim=True)
    
def compute_new_dimension(coords,dists,reference_indices):
    # Everything operates in-place
    if len(reference_indices) == 0:
        # Set a central bead at the origin
        idx = torch.tensor(dists.shape[-1]//2).expand_as(dists[...,:1]).to(dists.device)
        reference_indices.append(idx)

    ri = reference_indices # ease of notation
    x_norm = dists.gather(-1,ri[0]) # Distance from origin
    
    coord_value, idx = select_new_indices(x_norm,coords)
    idx = idx.expand_as(dists[...,:1])
    dim = len(ri) - 1
    y_norm = x_norm.gather(-2,idx) # Distance from origin for new reference bead
    x_minus_y_norm = dists.gather(-1,idx) # Distance between all beads and the new reference bead
    
    new_coord_values = x_dot_y(x_norm,y_norm,x_minus_y_norm)
    #print(new_coord_values)
    if dim > 0:
        selected_coord_prior_values = coords[...,:dim].gather(-2,idx.expand_as(coords[...,:dim]))
        new_coord_values-= (selected_coord_prior_values * coords[...,:dim]).sum(-1,keepdim=True) # Dot product
    new_coord_values/= coord_value
    coords[...,dim:dim+1] = new_coord_values
    
    ri.append(idx)

########################################################
# Optimize coordinates to a reference distance map

## NEED is_exact function for the saving function! 
########################################################
# Initialization
def format_dists(dists):
    '''
    Also handle filepaths, trajectories
    '''
    t = type(dists)
    if issubclass(t,Sample):
        return coords.coordinates.values
    if t != torch.Tensor:
        try:
            dists = torch.Tensor(dists)
        except:
            raise Exception('Data must be convertible to torch.Tensor object, '+\
                            f'but the provided {t} object cannot be!')

    dimension_msg = f'Input shape {dists.shape} is not valid!\n'
    dimension_msg+= 'Dimensions should be one of the following:\n'
    dimension_msg+= '\t(num_atoms,num_atoms): Dimensions of a distance map; or\n'
    dimension_msg+= '\t(batch_dim1,...,batch_dimN,num_atoms,num_atoms).\n'
    assert dists.ndim > 1, dimension_msg
    assert dists.shape[-2]==dists.shape[-1], dimension_msg
    if dists.ndim == 2:
        dists = dists.unsqueeze(0)
    
    return dists

########################################################
# Analytically convert distance maps into coordinates
def dists_to_coords(dists,device=None,num_dimensions=3):

    # Use high-precision values throughout calculation, but return same dtype as provided
    # Same for device
    return_dtype = dists.dtype
    return_device = dists.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dists = dists.double().to(device)

    # Initialize the object to hold coordinates
    coords = torch.zeros_like(dists[...,:num_dimensions])

    # Keep track of reference indices
    reference_indices = []

    for _ in range(num_dimensions): 
        compute_new_dimension(coords,dists,reference_indices)

    return coords.to(dtype=return_dtype,device=return_device)
    
def x_dot_y(x_norm,y_norm,x_minus_y_norm):
    # From known vector norms
    y_norm = y_norm.expand_as(x_norm)
    return (x_norm**2 + y_norm**2 - x_minus_y_norm**2) / 2

def select_new_indices(dist_from_origin,coords):
    dist_not_accounted_for = (dist_from_origin.square() - coords.square().sum(-1,keepdim=True))

    # Numerical precision occasionally causes small negative values to appear... to avoid NaN results, set those to 0!
    dist_not_accounted_for[dist_not_accounted_for<0] = 0
    dist_not_accounted_for.sqrt_()
    # Select maximum value to minimize numerical errors with division on this value later
    #return dist_not_accounted_for.max(-2,keepdim=True)
    return dist_not_accounted_for.median(-2,keepdim=True)
    
def compute_new_dimension(coords,dists,reference_indices):
    # Everything operates in-place
    if len(reference_indices) == 0:
        # Set a central bead at the origin
        idx = torch.tensor(dists.shape[-1]//2).expand_as(dists[...,:1]).to(dists.device)
        reference_indices.append(idx)

    ri = reference_indices # ease of notation
    x_norm = dists.gather(-1,ri[0]) # Distance from origin
    
    coord_value, idx = select_new_indices(x_norm,coords)
    idx = idx.expand_as(dists[...,:1])
    dim = len(ri) - 1
    y_norm = x_norm.gather(-2,idx) # Distance from origin for new reference bead
    x_minus_y_norm = dists.gather(-1,idx) # Distance between all beads and the new reference bead
    
    new_coord_values = x_dot_y(x_norm,y_norm,x_minus_y_norm)
    #print(new_coord_values)
    if dim > 0:
        selected_coord_prior_values = coords[...,:dim].gather(-2,idx.expand_as(coords[...,:dim]))
        new_coord_values-= (selected_coord_prior_values * coords[...,:dim]).sum(-1,keepdim=True) # Dot product
    new_coord_values/= coord_value
    coords[...,dim:dim+1] = new_coord_values
    
    ri.append(idx)

########################################################
# Optimize coordinates to a reference distance map

def smooth_transition_loss(
    output,
    target,
    r_c=1.0, # Transition distance from x**2 -> x**(long_scale)
    long_scale=1
):
    '''
    Reduces to smooth L1 loss if  long_scale == 1
    '''
    # Scale to ensure the two functions have the same slope at r_c
    m = 2 / long_scale
    # Shift to ensure the two functions have the same value at r_c
    b = 1 - m
    
    loss = 0
    difference = (output - target).abs() / r_c
    mask = difference < 1
    if mask.any():
        #loss = loss + difference[mask].square().sum()
        loss = loss + torch.nansum(difference[mask].square())
    mask = ~mask
    if mask.any():
        #loss = loss + (m*difference[mask]**long_scale + b).sum()
        loss = loss + torch.nansum( m*difference[mask]**long_scale + b )

    return loss

def loss_fcn(coords,target_dists,r_c=1.0,long_scale=1/8,proportional=False):
    dists = torch.cdist(coords,coords)
    i,j = torch.triu_indices(dists.shape[-1],dists.shape[-1],1)
    output,target = dists[...,i,j],target_dists[...,i,j]
    if proportional:
        # Adding .0001 for numerical stability where VERY small values appear
        return smooth_transition_loss((output+.0001)/(target+.0001),torch.ones_like(output))
    else:
        return smooth_transition_loss(output,target)

def correct_coords(
    coords,
    target_dists,
    *,
    min_loss_change=1e-6,
    #num_iterations=10_000,#1000,
    num_iterations_absolute=1_000,
    num_iterations_relative=9_000,
    lr=.1,
    lr_decay=0,
    weight_decay=0,
    r_c=1.0,
    long_scale=1/8
):

    return_dtype = coords.dtype
    return_device = coords.device
    coords = coords.double()
    target_dists = target_dists.double()
    if torch.cuda.is_available():
        coords = coords.cuda()
        target_dists = target_dists.cuda()

    coords.requires_grad_(True)
    
    optimizer = torch.optim.Adagrad(
        [coords],
        lr=lr,
        lr_decay=lr_decay,
        weight_decay=weight_decay
    )
    
    prior_loss = loss_fcn(coords,target_dists).detach()

    num_iterations = num_iterations_absolute + num_iterations_relative
    with tqdm(initial = 0, total = num_iterations, leave=None) as pbar:
        for i in range(num_iterations):
            optimizer.zero_grad()
            #loss = loss_fcn(coords,target_dists,r_c=r_c,long_scale=long_scale)
            loss = loss_fcn(coords,target_dists,r_c=r_c,long_scale=long_scale,proportional=i>=num_iterations_absolute)
            loss.backward()
            optimizer.step()

            loss = loss.detach()
            if i > 0 and abs(prior_loss - loss) < min_loss_change:
                print(f'Change in loss value less than tolerance: {loss - prior_loss}')
                break
            prior_loss = loss

            if i%100 == 0:
                pbar.set_description(f'Correcting Distance Maps. loss per distance value: {loss/target_dists.numel():.4f}')
                pbar.update(100)
    
    return coords.detach().to(dtype=return_dtype,device=return_device)

def smooth_transition_loss_by_sample(
    output,
    target,
    r_c=1.0, # Transition distance from x**2 -> x**(long_scale)
    long_scale=1/8,
    use_gpu=True,
    high_precision=True
):
    '''
    Reduces to smooth L1 loss if  long_scale == 1.
    
    Rather than summing over ALL data, sum over the final two 
    dimensions (corresponding to individual distance maps). 
    '''
    # Scale to ensure the two functions have the same slope at r_c
    m = 2 / long_scale
    # Shift to ensure the two functions have the same value at r_c
    b = 1 - m
    
    return_device = output.device
    return_dtype = output.dtype
    if use_gpu and torch.cuda.is_available():
        output = output.cuda()
        target = target.cuda()
    if high_precision:
        output = output.double()
        target = target.double()

    losses = (output - target).abs_()
    del output, target
    losses/= r_c
    
    if losses.is_cuda:
        '''
        This is slower than using masking, but it avoid memory issues associated
        with mask indexing (torch turns bool masks into int64 indexing arrays) while
        remaining faster than some alternative low-memory options I tried. 
        '''
    
        #losses = torch.where(
        #    losses < 1,
        #    losses.square(),
        #    m*losses.pow(long_scale)+b,
        #    out=losses
        #)
        torch.where(
            losses < 1,
            losses.square(),
            m*losses.pow(long_scale)+b,
            out=losses
        )
        
    else:
        '''
        Assume that these memory issues don't arise on the CPU
        '''
        mask = losses < 1
        if mask.any():
            losses[mask] = losses[mask]**2
        mask^= True
        if mask.any():
            losses[mask] = m*losses[mask]**long_scale + b
        del mask
    

    return losses.sum((-1,-2)).to(dtype=return_dtype,device=return_device)

class Distances(Sample):

    def __init__(
        self,
        input
    ):
        self.__dist_maps = format_dists(input)

    ########################################################
    # Needed for much of the functionality in Sample superclass
    @property
    def _values(self):
        return self.__dist_maps

    @_values.setter
    def _values(self,c):
        self.__dist_maps = c

    ########################################################
    # Helps compare these samples to samples from other classes
    #@property
    def __is_exact(self,other_distances):
        n = self.num_beads
        i,j = torch.triu_indices(n,n,1)
        corrcoef = torch.corrcoef(
            torch.stack(
                [
                    self.values[...,i,j].flatten(),
                    other_distances[...,i,j].flatten()
                ],
                dim=0
            )
        )[0,1]
        return corrcoef == 1
        
    @property
    def is_exact(self):
        '''
        Convert to coordinates -- without optimization -- and back
        to distances. Even when using an exact solution, torch.allclose
        doesn't work due to numerical precision issues. However, using
        the pearson correlation coefficient seems to work
        '''
        reconstructed_dists = coords_to_dists(dists_to_coords(self.values))
        return self.__is_exact(reconstructed_dists)

    ########################################################
    # Optimize a set of coordinates to best match the given
    # distance maps
    def correct_coordinates(
        self,
        coords=None,
        *,
        min_loss_change=1e-4,
        num_iterations_absolute=1_000,
        num_iterations_relative=9_000,
        lr=.1,
        lr_decay=0,
        weight_decay=0,
        r_c=1.0,
        long_scale=1/8
    ):
        coords = self if coords is None else coords
        if issubclass(type(coords),Sample):
            if type(coords) == Distances: # To avoid recursion
                coords = dists_to_coords(coords.values)
            else:
                coords = coords.coordinates.values

        if type(coords) != torch.Tensor:
            coords = torch.tensor(coords)
        coords = coords.to(dtype=self.dtype,device=self.device)

        coords = correct_coords(
            coords,
            self.values,
            min_loss_change=min_loss_change,
            num_iterations_absolute=num_iterations_absolute,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            r_c=r_c,
            long_scale=long_scale
        )

        return coords

    ########################################################
    # Converting between sample subclasses
    
    # Always be able to return coordinates object
    @property
    def coordinates(self):
        '''
        Optimize coordinates with default optimization values if necessary
        '''
        coords = dists_to_coords(self.values)
        reconstructed_dists = coords_to_dists(coords)
        if not self.__is_exact(reconstructed_dists):
            coords = self.correct_coordinates(coords)
        return Coordinates(coords)

    # Always be able to return trajectory object
    @property
    def trajectory(self): 
        return self.coordinates.trajectory

    # Always be able to return distance maps 
    @property
    def distances(self):
        return self
        
        
        
        