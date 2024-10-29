'''
TODO: 
1. Define format_and_infer_class() 
2. Define a generic save/load class method
'''

import torch
from abc import ABC, abstractmethod
import copy

########################################################
# Support functionality
########################################################

# Basics
def prod(input):
    return torch.tensor([*input]).prod()

def get_type(obj):
    if type(obj) == type:
        pass
    return str(type(obj)).split('.')[-1].strip("'>").lower()

# Support for __eq__
def distances_are_equal(distmap1,distmap2):
    # NOTE: This is only called if we've already verified that the tensor
    # shapes are compatible 
    
    # Check if the maps happen to be exactly equal
    if torch.allclose(distmap1,distmap2):
        return True
        
    # Otherwise, numerical imprecision when switching between coordinates
    # and distance maps can cause torch.allclose to fail. 
    # Thus, if that method failed, use the following correlation
    # analysis as a band-aid.
    n = distmap1.shape[-1]
    i,j = torch.triu_indices(n,n,1)
    corrcoef = torch.corrcoef(
        torch.stack(
            [
                distmap1[...,i,j].flatten(),
                distmap2[...,i,j].flatten()
            ],
            dim=0
        )
    )[0,1]
    return corrcoef == 1
    
########################################################
# Main abstract class for the module
########################################################

class ConformationsABC(ABC):

    __version__ = '0.0.0'

    ########################################################
    # Equivalent of initialization
    '''
    @classmethod
    def from_data(cls,input):

        if issubclass(type(input),ConformationsABC):
            return input

        return format_and_infer_class(input)
    '''
        
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

    @property
    def is_flat(self):
        return self.values.ndim==3
    
    ########################################################
    # Standard functionality 

    def __getitem__(self,i):
        if type(i) == float:
            assert int(i) == i, f'Floating-point value {i} is an invalid index for {type(self)} objects'
            i = int(i)
        if type(i) == int:
            assert i < len(self) and i > -len(self)-1, f'Index {i} is out of bounds for {type(self)} of length {i}'
        return type(self)(self.values[i])

    def __iter__(self):
        flat_version = self.flatten()
        return ( flat_version[i] for i in range(len(self)) )
            
    
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
        if issubclass(type(obj),ConformationsABC):
            if obj.num_beads != self.num_beads or len(obj) != len(self):
                return False

            # If one is a distance map but the other is not, we should convert
            # both samples to a distances representation to remove effects of 
            # translation, rotation, etc. 
            # If both are distances maps, we obviously also want to use distances... 
            # Due to numerical inconsistencies arising from converting between coordinates
            # and distance maps, use distances_are_equal comparison instead of the 
            # 
            if t == 'distances' or self.__type == 'distances':
                return distances_are_equal(
                    self.distances.values,
                    obj.distances.values.to(dtype=self.dtype,device=self.device)
                )
            
            else:
                # Both Trajectory and Coordinates objects return coordinates by default. 
                # If coordinates were provided, assume that translations, rotations, etc.,
                # are relevant for comparison, so don't convert to distances (which removes this information)
                obj = obj.values.to(dtype=self.dtype,device=self.device)
        else:
            if t != 'tensor':
                try:
                    obj = torch.tensor(obj)
                except:
                    # If it can't be converted to a torch Tensor, then it isn't a compatible type => not equal!
                    return False
            # If there are a different number of elements in the two objects, then return False.
            # Note: This means that coordinates can only be compared to distances if using ConformationABC subclasses.
            # Similarly, the numerical imprecision from converting between coordinates & distances won't be accounted for
            # unless the data is held within ConformationsABC subclasses. 
            if obj.numel() != self.values.numel():
                return False

            obj = obj.reshape(self.shape).to(dtype=self.dtype,device=self.device)

        return torch.allclose(obj,self.values)

    
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
        return self

    def cuda_(self):
        self._values = self._values.cuda()
        return self

    def cpu_(self):
        self._values = self._values.cpu()
        return self

    def float_(self):
        self._values = self._values.float()
        return self

    def double_(self):
        self._values = self._values.double()
        return self
        
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
    
    def __format_other_object(self,other_object):#,**kwargs):
        #return_kwargs,kwargs = self.__parse_trajectory_kwargs(**kwargs)
        
        #if not issubclass(type(other_object),ConformationsABC):
        #    # will raise exception if invalid non-Sample input
        #    other_object = format_and_infer_class(other_object,**kwargs)
        if issubclass(type(other_object),ConformationsABC):
            other_object = other_object.values
        elif type(other_object) != torch.Tensor:
            try:
                other_object = torch.tensor(other_object)
            except:
                raise Exception('Objects to be appended must be convertible to torch.Tensor, '+\
                                f'but input of type {type(other_object)} is not!')
        other_object = other_object.to(dtype=self.dtype,device=self.device)
        return other_object
        '''
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
        '''

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
        if type(other_object) != list:
            other_object = [other_object]
        other_conformations = []
        for i,obj in enumerate(other_object):
            #kwargs1,other_object = self.__format_other_object(obj,**kwargs)
            other_conformations.append(self.__format_other_object(obj))
            
        data = torch.cat(
            [
                self.values,
                *other_conformations
            ],
            *args,
            **kwargs
        )
        return type(self)(data)  
    
    def append(self,other_object,*args,**kwargs):
        return self.__cat(other_object,dim=0,*args,**kwargs)

    def flatten_(self):
        if self.is_flat:
            return self
        self._values = self.values.flatten(0,-3)

    def flatten(self):
        # NOTE: Should be a way to make a shallow copy, THEN flatten the copy, but this'll work for now
        if self.is_flat:
            return self
        return self.clone().flatten_()

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

    ''' Should make these module functions
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
            save_obj = self.coordinates
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
        '#''
        ::filepath:: Path to a file containing either a tensor recognizable as coordinates/distance maps, 
            a filepath recognized by a molecular filepath type recognized by mdtraj, or 
        
        ::return_class:: Class to use when returning the data. 
            Either directly a class or a string denoting it as 'distances', 'trajectory', or 'coordinates'.

        
        ::map_location:: Same as in PyTorch's torch.load()
        :::args:: For torch.load()
        ::kwargs:: for torch.load()
        '#''
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
    '''

#from Coordinates import Coordinates
#from Trajectory import Trajectory
#from Distances import Distances