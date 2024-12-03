'''
Greg Schuette 2024
'''
import torch
import pandas as pd
import mdtraj as md
import numpy as np
from ._ConformationsABC import ConformationsABC

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
    t = md.load(filepath,top=topology)
    t.xyz*= 10
    return t

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
    

class Trajectory(ConformationsABC):

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

    # Get the mdtraj objects to use any desired functionality there that isn't
    # shortcutted directly in this class
    @property
    def mdtraj(self):
        return self.__trajectory

    @property
    def topology(self):
        return self.__trajectory.topology
    
    ########################################################
    # Overriding some functions from the superclass for efficiency. 

    def __len__(self):
        return self.__trajectory.n_frames
    
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
            
        if issubclass(type(reference),ConformationsABC):
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
        return torch.from_numpy(md.rmsd(self.__trajectory,reference,*args,**kwargs)).to(dtype=self.dtype,device=self.device)

    def compute_rg(self,*args,**kwargs):
        return torch.from_numpy(md.compute_rg(self.__trajectory),*args,**kwargs).to(dtype=self.dtype,device=self.device)
        
    ########
    # The following use the above mdtraj functions to align 3D conformations
    # with reference data & select the conformations that best match the 
    # reference data for visualization purposes. 

    def __best_superposition(self,reference_frame):
        self.superpose(reference_frame)
        scores = self.rmsd(reference_frame)
        scores/= (
            self.compute_rg() * 
            torch.from_numpy(
                md.compute_rg(reference_frame)
            ).to(dtype=self.dtype,device=self.device)
        ).sqrt()
        best_score = scores.min()
        best_idx = torch.where(scores == best_score)[0][0]
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
            'Normed RMSD':torch.tensor(best_scores),
            'Index':torch.tensor(best_indices)
        }
        best_frames = Trajectory(np.concatenate(best_frames,axis=0))
        return best_info,best_frames
    
    ########################################################
    # Converting between ComformationsABC subclasses
    
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

    ########################################################
    # Save data
    def save_dcd(self,*args,**kwargs):
        self.__trajectory.xyz/=10
        self.__trajectory.save_dcd(*args,**kwargs)
        self.__trajectory.xyz*=10
        

from ._Coordinates import Coordinates
