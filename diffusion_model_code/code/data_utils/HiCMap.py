'''
Convenient class for visualizing/comparing Hi-C maps
'''

import torch
import copy 
import numpy as np

# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import cooltools.lib.plotting # provides 'fall' colormap

def add_diagonal(mat): 

    # Initialize the new matrix
    n = mat.shape[0] + 1
    mat1 = torch.empty(n,n,dtype=mat.dtype,device=mat.device)

    # Fill in the upper/lower matrices
    i,j = torch.triu_indices(n,n,1)
    j2 = j - 1
    mat1[i,j] = mat[i,j2] 
    mat1[j,i] = mat[j2,i] 

    # Fill in the diagonal with ones (assume self-interaction probability==1, essentially) 
    mat1.fill_diagonal_(1)

    return mat1 

def pcc(mat1,mat2,ignore_diagonal=True,lower=False): 
    assert mat1.shape==mat2.shape

    n = mat1.shape[0] 
    i,j = torch.triu_indices(n,n,int(ignore_diagonal))
    if lower: 
        i,j = j,i

    mat = torch.stack((mat1[i,j],mat2[i,j]),dim=0)

    return torch.corrcoef(mat)[0,1] 

from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_hic_map(
        sample,
        *,
        vmin=None,
        vmax=1,
        cmap='fall',
        fig=None,
        ax=None,
        cbar_orientation=None,
        #title=None, # Add functionality for these later. 
        #colorbar_label=None,
        #xlabel=None,
        #ylabel=None
    ):

        norm = LogNorm(vmin=vmin,vmax=vmax)
        n,hic_map = sample.nbins, sample.prob_map
        extent = (0,n,n,0)
        
        if fig is None or ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        im = ax.matshow(
            hic_map.cpu().numpy(),
            norm=norm,
            cmap=cmap,
            extent=extent
        );
        #ax.xaxis.set_visible(False) # don't show the arbitrary ticks
        #ax.yaxis.set_visible(False) # don't show the arbitrary ticks 
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('Genomic index')

        # Ensure colorbar is the same size
        divider = make_axes_locatable(ax)
        if cbar_orientation == 'horizontal':
            cax = divider.append_axes("bottom", size="5%", pad=0.05)
        else:
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax.set_xlabel('Genomic index')
            
        if cbar_orientation is not None:
            cbar = fig.colorbar(im,cax=cax,label='Interaction Frequency',orientation=cbar_orientation)
        else:
            cbar = fig.colorbar(im,cax=cax,label='Interaction Frequency')
        #cbar = fig.colorbar(im, label='Interaction Frequencies',location='right');
    
        return fig,ax,im,cbar

class HiCMap: 
    '''
    hic class that can be obtained from either the Sample class or the Hi-C dataset class
    Should have attributes for easy comparison, e.g. hic_map1.pcc(hic_map2) 
    Should also have a plot function 
    '''
    def __init__(
        self,
        prob_map,
        *,
        chrom=None,
        start=None,
        stop=None,
        includes_self_interaction=True,
        device='cuda' if torch.cuda.is_available() else None, #None,
        dtype=torch.double
    ):

        # Ensure probability map is a torch Tensor
        if type(prob_map) == np.ndarray: 
            prob_map = torch.from_numpy(prob_map)

        # Ensure the probability map is square. Remove any dimensions of 1
        p = prob_map.squeeze()
        assert len(p.shape) == 2, f'The provided probability map should have exactly two nonsingleton dimensions, but has dimensions {prob_map.shape}'
        assert prob_map.shape[0]==prob_map.shape[1], f'The provided probability map should be square, but has dimensions {prob_map.shape}'
        prob_map = p 

        # Satisfy device requirements & dtype requirements
        #if device is None: 
        #    # By default, use the GPU (if available) 
        #    device = 'cuda' if torch.cuda.is_available() else None
        self.prob_map = prob_map.to(device=device,dtype=dtype)

        # If the diagonal is not included, add it
        if not includes_self_interaction: 
            self.prob_map = add_diagonal(self.prob_map)

        # Set the properties of the given region. 
        # **** not used for now, but should be used in self.plot later for label/tick generation ****
        self.chrom = chrom
        self.start = start
        self.stop = stop 
        
    #####################################################
    # Basic properties & functionality 
    @property
    def device(self):
        return self.prob_map.device

    @property
    def dtype(self):
        return self.prob_map.dtype

    @property
    def shape(self):
        return self.prob_map.shape

    @property
    def nbins(self): 
        return self.shape[0] 

    def __getitem__(self,i): 
        return self.prob_map[i]

    def clone(self):
        # Return a deep copy of the object
        return copy.deepcopy(self) 

    def to_(self,*args):
        # In-place
        self.prob_map = self.prob_map.to(*args)

    def to(self,*args):
        # Returns copy with desired attributes
        out = self.clone()
        out.to_(*args)
        return out

    def T_(self):
        # transpose
        self.prob_map = self.prob_map.transpose(-2,-1)

    def T(self):
        copied_map = self.clone()
        copied_map.T_()
        return copied_map

    def replace_lower_triangle(self,other_map,*,replace_diagonal=False):
        n = self.nbins
        i,j = torch.tril_indices(n,n,int(replace_diagonal)-1)
        self.prob_map[i,j] = other_map.prob_map[i,j]

    #####################################################
    # Operations
    def pcc(self,hic_map,*args):
        # Pearson correlation coefficient
        if type(hic_map) in [torch.Tensor, np.ndarray]: 
            # This will ensure the matrix is square, etc etc
            hic_map = HiCMap(hic_map)
    
        assert type(hic_map) == type(self), f"The hic_map argument should be of type {type(self)}, torch.Tensor, or np.ndarray. Received {type(hic_map)}"

        if hic_map.dtype != self.dtype or hic_map.device != self.device: 
            hic_map = hic_map.to(device=self.device,dtype=self.dtype) 

        return pcc(self,hic_map,*args)

    #####################################################
    # Visualization 
    def plot(
        self,
        **kwargs # keyword arguments for plot_hic_map()
    ):
        return plot_hic_map(self,**kwargs)
        '''
        norm = LogNorm(vmin=vmin,vmax=vmax)
        n,hic_map = self.nbins, self.prob_map
        extent = (0,n,n,0)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        im = ax.matshow(
            hic_map.cpu().numpy(),
            norm=norm,
            cmap=cmap,
            extent=extent
        );
        ax.xaxis.set_visible(False) # don't show the arbitrary ticks
        ax.yaxis.set_visible(False) # don't show the arbitrary ticks 
        
        cbar = fig.colorbar(im, label='Interaction Frequencies',location='right');
    
        return fig,ax,im,cbar
        '''

    def plot_with(
        self,
        other_map,
        *,
        self_on_upper=True,
        **kwargs # keyword arguments for plot_hic_map()
    ):
        # Don't want to permanently alter either map, so clone one
        temp_map = self.clone()

        # Place the other contacts on the lower triangle
        temp_map.replace_lower_triangle(other_map)

        # Flip the other map to the top if desired
        if not self_on_upper:
            temp_map.T_()

        # Plot the map
        return temp_map.plot(**kwargs)
        
        

        
