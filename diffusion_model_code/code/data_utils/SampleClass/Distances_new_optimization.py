'''
TODO: 
1. Fix num_beads, etc., functions for when in folded state.
2. Add functionality to fold more than once. 
'''

import torch
from ConformationsABC import ConformationsABC
from OrigamiTransform import OrigamiTransform
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Initialization
def format_dists(dists):
    '''
    Also handle filepaths, trajectories
    '''
    t = type(dists)
    if issubclass(t,ConformationsABC):
        return dists.distances.values
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
# Normalizer functions
def is_normalized(dist_maps):
    return (dist_maps <= 1).all()
    
class Normalizer:

    def __init__(
        self,
        mean_dist_fp='../../data/mean_dists.pt',
        mean_sq_dist_fp='../../data/squares.pt',
    ):

        mean_dist = torch.load(mean_dist_fp).flatten().double()
        mean_square_dist = torch.load(mean_sq_dist_fp).flatten().double()
            
        self.dist_std = (mean_square_dist - mean_dist**2).sqrt()
        self.inv_beta = torch.sqrt( 2*mean_square_dist/3 )
        self.inv_beta_sigmoid = torch.sigmoid( -self.inv_beta/self.dist_std )
        self.complement_inv_beta_sigmoid = 1 - self.inv_beta_sigmoid
    
    def to(self,*args,**kwargs):
        '''
        Primarily for moving to the device of whichever object is being worked on
        '''
        self.dist_std = self.dist_std.to(*args,**kwargs)
        self.inv_beta = self.inv_beta.to(*args,**kwargs)
        self.inv_beta_sigmoid = self.inv_beta_sigmoid.to(*args,**kwargs)
        self.complement_inv_beta_sigmoid = self.complement_inv_beta_sigmoid.to(*args,**kwargs)

    def __prep_for_comp(self,dists):
        n = dists.shape[-1]
        assert dists.ndim > 1 and dists.shape[-2] == n, \
        'Expected square distance matrices in the final two dimensions, '+\
        'but received object with shape {dists.shape}'

        # Always perform computations with high precision to avoid numerical issues with 
        # the logit function.
        dists1 = dists.double()

        # Also use the GPU for performance... transferring data back and forth may actually
        # slow things down on small batches, but that's a problem to think about later. 
        if torch.cuda.is_available():
            dists1 = dists1.cuda()

        # Diagonal values should always equal 0, but are occasionally non-zero due to 
        # numerical imprecision with the U-Net. Fix that here. 
        dists1[...,range(n),range(n)] = 0

        # If the matrices are symmetric, just perform the computation once
        # and broadcast the results from the upper triangle to the lower triangle
        broadcast = torch.allclose(dists1,dists1.transpose(-2,-1))

        # Indices to grab upper/lower triangle
        i,j = torch.triu_indices(n,n,1)
        
        # Separation index. Subtract 1 since the mean distances/square distances
        # objects were saved without self interaction (so index 0 corresponds to 
        # objects with 1 bond between them)
        sep = j - i - 1

        # Move this object's tensors to the same device as the distances object
        self.to(dists1.device)
        
        return dists1, broadcast, i, j, sep

    def __normalize(self,dists,sep):
        '''
        Normalize distances. This takes data ALREADY
        indexed with torch.triu_indices indexing, as in self.unnormalize()
        '''

        dists-= self.inv_beta[sep].expand(*dists.shape[:-1],-1)
        dists/= self.dist_std[sep].expand(*dists.shape[:-1],-1)
        dists.sigmoid_()
        dists-= self.inv_beta_sigmoid[sep].expand(*dists.shape[:-1],-1)
        dists/= self.complement_inv_beta_sigmoid[sep].expand(*dists.shape[:-1],-1)
        
        return dists
    
    def normalize(self,dists):

        if is_normalized(dists):
            return dists

        # This fucntion moves data to the GPU (if available) and uses double precision,
        # but we want to return an object with the original dtype and on the original device, 
        # so track them
        return_dtype = dists.dtype
        return_device = dists.device

        # Placed some data prep code into another function since it's used
        # exactly the same in both normalize and unnormalize functions
        dists, broadcast, i, j, sep = self.__prep_for_comp(dists)

        # Normalize the values
        dists[...,i,j] = self.__normalize(dists[...,i,j],sep)
        if broadcast:
            dists[...,j,i] = dists[...,i,j]
        else:
            dists[...,j,i] = self.__normalize(dists[...,j,i],sep)

        return dists.to(dtype=return_dtype,device=return_device)
        
    def __unnormalize(self,dists,sep):
        '''
        Unnormalize distances. This takes data ALREADY
        indexed with torch.triu_indices indexing, as in self.unnormalize()
        '''
        dists*= self.complement_inv_beta_sigmoid[sep].expand(*dists.shape[:-1],-1)
        dists+= self.inv_beta_sigmoid[sep].expand(*dists.shape[:-1],-1)
        dists.logit_()
        dists*= self.dist_std[sep].expand(*dists.shape[:-1],-1)
        dists+= self.inv_beta[sep].expand(*dists.shape[:-1],-1)

        # The logit function causes values at 1 to become infinite-valued, an
        # artifact handled during the conversion to coordinates. 
        # However, the -infinity values (which actually end up near 0 due to the 
        # other conversions performed) are easier to handle here
        dists[dists<1e-8] = .01
        return dists
    
    def unnormalize(self,dists):

        if not is_normalized(dists):
            return dists

        # Placed some data prep code into another function since it's used
        # exactly the same in both normalize and unnormalize functions
        dists1, broadcast, i, j, sep = self.__prep_for_comp(dists)

        # Unnormalize the values
        dists[...,i,j] = self.__unnormalize(dists1[...,i,j],sep).to(dtype=dists.dtype,device=dists.device)
        if broadcast:
            dists[...,j,i] = dists[...,i,j]
        else:
            dists[...,j,i] = self.__unnormalize(dists1[...,j,i],sep).to(dtype=dists.dtype,device=dists.device)

        return dists

########################################################
# Analytically convert distance maps into coordinates
def dists_to_coords(dists,device=None,num_dimensions=3,num_attempts=None,error_threshold=1e-4):

    # Use high-precision values throughout calculation, but return same dtype as provided
    # Same for device
    return_dtype = dists.dtype
    return_device = dists.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dists = dists.double().to(device)

    # Initialize the object to hold coordinates
    coords = torch.empty_like(dists[...,:num_dimensions])

    # Choose beads to set at the origin when computing conformations, which 
    # occasionally fails due to numerical issues with the distance maps generated
    # by the diffusion model 
    n = dists.shape[-1]
    num_attempts = n if num_attempts is None else min(num_attempts,n)
    origins = [i*dists.shape[-1]//num_attempts for i in range(num_attempts)]

    # We'll ignore infinite values in the target distance maps, an artifact discussed further in correct_coords()
    inf_mask = torch.where(dists.isinf())

    # Convert distances to coordinates
    # Keep track of the set of coordinates that best agrees with the target distance map
    # to reduce the amount of optimization needed when correcting coordinates
    zero = torch.tensor(0.).to(dtype=dists.dtype,device=dists.device)
    for i,o in enumerate(origins):
        # Fill coords with zeros so the compute_new_dimension function operates as desired
        coords.fill_(0)
        
        # Keep track of reference indices as compute_new_dimension is called repeatedly
        reference_indices = []

        # Compute the new dimensions
        for _ in range(num_dimensions):
            compute_new_dimension(coords,dists,reference_indices,initial_index=o)

        # If any of these freshly-computed coordinates are closer to the desired result than
        # the best already encountered, update the best_coords/best_errors object
        errors = (torch.cdist(coords,coords)-dists).square_()
        errors[inf_mask] = 0 # Send errors at interactions with infinite distance in the target map to 0
        errors[errors.isnan()] = torch.inf # If NaN values appear, treat that conformation as infinitely wrong
        errors = errors.mean((-2,-1))
        if i == 0:
            best_coords = coords.clone()
            best_errors = errors.clone()
        else:
            mask = errors < best_errors
            if mask.any():
                best_coords[mask] = coords[mask]
                best_errors[mask] = errors[mask]
        if (best_errors < error_threshold).all():
            break
    return best_coords.to(dtype=return_dtype,device=return_device)
    
    '''
    # Convert distances to coordinates
    mask = torch.ones(*coords.shape[:-2],dtype=bool,device=coords.device)
    for o in origins:
        # The coordinates that still need to be computed due to failures in prior loop iterations
        temp_coords = coords[mask]
        temp_coords.fill_(0)
        temp_dists = dists[mask]
        
        # Keep track of reference indices
        reference_indices = []

        for _ in range(num_dimensions):
            compute_new_dimension(temp_coords,temp_dists,reference_indices,initial_index=o)
        coords[mask] = temp_coords
        mask = ~coords.isfinite().all(-1).all(-1)
        if not mask.any():
            break
    return coords.to(dtype=return_dtype,device=return_device)
    '''
    
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
    
def compute_new_dimension(coords,dists,reference_indices,initial_index=None):
    # Everything operates in-place
    if len(reference_indices) == 0:
        # Set a bead at the origin
        if initial_index is None: # Choose the central bead if none specified. 
            idx = torch.tensor(dists.shape[-1]//2).expand_as(dists[...,:1]).to(dists.device)
        else: 
            idx = torch.tensor(initial_index).expand_as(dists[...,:1]).to(dists.device)
        reference_indices.append(idx)

    ri = reference_indices # ease of notation
    x_norm = dists.gather(-1,ri[0]) # Distance from origin
    
    coord_value, idx = select_new_indices(x_norm,coords)
    idx = idx.expand_as(dists[...,:1])
    dim = len(ri) - 1
    y_norm = x_norm.gather(-2,idx) # Distance from origin for new reference bead
    x_minus_y_norm = dists.gather(-1,idx) # Distance between all beads and the new reference bead
    
    new_coord_values = x_dot_y(x_norm,y_norm,x_minus_y_norm)
    if dim > 0:
        selected_coord_prior_values = coords[...,:dim].gather(-2,idx.expand_as(coords[...,:dim]))
        new_coord_values-= (selected_coord_prior_values * coords[...,:dim]).sum(-1,keepdim=True) # Dot product
    new_coord_values/= coord_value
    coords[...,dim:dim+1] = new_coord_values
    
    ri.append(idx)

########################################################
# Optimize coordinates to a reference distance map
'''
def smooth_transition_loss(
    output,
    target,
    r_c=1.0, # Transition distance from x**2 -> x**(long_scale)
    long_scale=1
):
    '#''
    Reduces to smooth L1 loss if  long_scale == 1
    '#''
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
'''

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
    
    # 
    difference = torch.fmax(
        (output - target).abs() / r_c,
        ((output+1e-6)/(target+1e-6)-1).abs()
    )

    # Replace infinite values with NaN's, which are ignored below. 
    # Infinite values are an artifact of applying the logit function to values of 
    # 1 from the normalized distance maps produced by the U-Net. 
    # This causes these points to have no impact on the optimization, which is valid 
    # since we don't know the true distance intended at these points. 
    difference[difference.isinf()] = torch.nan

    # Remove explicit outliers, which are likely a result of numerical imprecision
    difference[difference > 25*torch.nanmean(difference)] = torch.nan

    #loss = 0
    loss = difference.clone()
    mask = difference < 1
    if mask.any(): # SSE for errors < 1
        #loss = loss + torch.nansum( difference[mask].square(), dim=-1)
        loss[mask] = loss[mask] + difference[mask].square()
    mask = ~mask
    if mask.any(): # Decaying slope for errors > 1, minimizing the impact of outliers
        #loss = loss + torch.nansum( m*difference[mask]**long_scale + b, dim=-1)
        loss[mask] = loss[mask] + m*difference[mask]**long_scale + b

    return torch.nansum(loss,dim=-1)

def loss_fcn(coords,target_dists,r_c=1.0,long_scale=1/8,near_neighbors_scales=[10,8,6,4,2]):
    dists = torch.cdist(coords,coords)
    i,j = torch.triu_indices(dists.shape[-1],dists.shape[-1],1)
    output,target = dists[...,i,j],target_dists[...,i,j]

    bond_dists = j - i
    for k,scale in enumerate(near_neighbors_scales):
        if scale != 1:
            # Provide additional weight to distances between sequentially proximal interactions, which  
            # otherwise end up being a worse match to the target than the larger-distance interactions
            idx = torch.where(bond_dists == k+1)[0]
            output[...,idx] = output[...,idx] * scale
            target[...,idx] = target[...,idx] * scale

    return smooth_transition_loss(output,target)

def correct_coords(
    coords,
    target_dists,
    *,
    min_loss_change_per_sample=1e-6,
    max_iterations=torch.inf,
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

    prior_per_sample_loss = loss_fcn(coords,target_dists).detach()
    running_mask = prior_per_sample_loss.isfinite()

    # Get the condition for exiting the loop 
    # Multiply by N(N-1)/2 to account for the fact that only the upper triangle, excluding diagonal, 
    # affects the losses
    N = target_dists.shape[-1]
    N = N*(N-1)/2
    exit_loss_condition = min_loss_change_per_sample * N #target_dists[0,...].numel() / N
    with tqdm(initial = 0, total = max_iterations, leave=None) as pbar:
        i = 0
        while i < max_iterations:

            if not running_mask.any():
                print(f'Change in loss is less than tolerance ({min_loss_change_per_sample}) '+\
                  '\n'+f'for all samples. Final per-sample loss: {torch.nanmean(prior_per_sample_loss)}')
                break
            
            sto = torch.where(running_mask)[0] # Samples to optimize
            optimizer.zero_grad()
            per_sample_loss = loss_fcn(coords[sto,...],target_dists[sto,...],r_c=r_c,long_scale=long_scale)
            loss = torch.nansum(per_sample_loss)
            loss.backward()
            optimizer.step()


            pbar.set_description(f'Optimizing 3D conformations. Average loss per distance value: {torch.nanmean(prior_per_sample_loss)/N:.4f}')
            pbar.update(1)
            
            if i > 0:
                per_sample_loss = per_sample_loss.detach()
                running_mask[sto] = (per_sample_loss - prior_per_sample_loss[sto]).abs() >= exit_loss_condition
                prior_per_sample_loss[sto] = per_sample_loss

            i+=1
            #if (i+1)%100 == 0 or i==0:
            #    pbar.set_description(f'Correcting Distance Maps. loss per distance value: {loss/target_dists.numel():.4f}')
            #    if i > 0:
            #        pbar.update(100)
    
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

########################################################
# Visualization
from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_dist_map(
    dists,
    fig=None,
    ax=None,
    cmap='RdBu',
    xticks=[],
    xticklabels=None,
    yticks=[],
    yticklabels=None,
    xlabel='Genomic index',
    ylabel='Genomic index',
    cbar_label='Distance',
    cbar_orientation=None,
    cbar_ticks=None,
    **kwargs
):
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    im = ax.matshow(dists.cpu().numpy(),cmap=cmap,**kwargs)

    # Ensure colorbar is the same size
    divider = make_axes_locatable(ax)
    if cbar_orientation == 'horizontal':
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
    else:
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.set_xlabel(xlabel)
        
    if cbar_orientation is not None:
        cbar = fig.colorbar(im,cax=cax,label=cbar_label,orientation=cbar_orientation,ticks=cbar_ticks)
    else:
        cbar = fig.colorbar(im,cax=cax,label=cbar_label,ticks=cbar_ticks)
    #cbar = fig.colorbar(im,label=cbar_label)
    
    ax.set_xticks(xticks,labels=xticklabels)
    ax.set_yticks(yticks,labels=yticklabels)

    ax.set_ylabel(ylabel)

    return fig, ax, im, cbar
    
    

########################################################
# Main class
class Distances(ConformationsABC):

    def __init__(
        self,
        input,
        description = 'Distance'
    ):
        self.__dist_maps = format_dists(input)
        self.__description = description
        self.__origami_transform = OrigamiTransform()
        self.__is_folded = input.ndim > 3 and input.shape[-3] == 2

    ########################################################
    # Needed for much of the functionality in Sample superclass
    @property
    def _values(self):
        return self.__dist_maps

    @_values.setter
    def _values(self,c):
        self.__dist_maps = c

    ########################################################
    # Basic data manipulation
    def flatten(self):
        return Distances(self.values.flatten(0,-3),self.__description)
    
    ########################################################
    # Distance Statistics
    @property
    def mean(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return Distances(self.flatten().values.to(device).mean(0).to(self.device),'Mean Distance')

    @property
    def median(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return Distances(self.flatten().values.to(device).median(0)[0].to(self.device),'Median Distance')

    @property
    def mean_by_bond_separation(self):
        out = self.mean
        n = out.num_beads
        for i in range(1,n):
            out.values[...,range(n-i),range(i,n)] = out.values[...,range(n-i),range(i,n)].mean()
            out.values[...,range(i,n),range(n-i)] = out.values[...,range(n-i),range(i,n)]
        return out

    @property
    def std(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return Distances(self.flatten().values.to(device).std(0).to(self.device),'Standard Deviation')

    @property
    def var(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return Distances(self.flatten().values.to(device).var(0).to(self.device),'Variance')
    
    
    ########################################################
    # Convert to/from the normalized form used with the U-Net
    def normalize_(self,normalizer=None,*args,**kwargs):
        normalizer = Normalizer(*args,**kwargs) if normalizer is None else normalizer
        self.__dist_maps = normalizer.normalize(self.__dist_maps)
        return self

    def normalize(self,*args,**kwargs):
        return self.clone().normalize_(*args,**kwargs)

    def unnormalize_(self,normalizer=None,*args,**kwargs):
        normalizer = Normalizer(*args,**kwargs) if normalizer is None else normalizer
        self.__dist_maps = normalizer.unnormalize(self.__dist_maps)
        return self

    def unnormalize(self,*args,**kwargs):
        return self.clone().unnormalize_(*args,**kwargs)

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
        #reconstructed_dists = coords_to_dists(dists_to_coords(self.values))
        self.uncorrected_coordinates.distances
        return self.__is_exact(reconstructed_dists)

    ########################################################
    # Optimize a set of coordinates to best match the given
    # distance maps
    def correct_coordinates(
        self,
        coords=None,
        *,
        min_loss_change_per_sample=5e-7,
        max_iterations=torch.inf,
        lr=.1,
        lr_decay=0,
        weight_decay=0,
        r_c=1.0,
        long_scale=1/8
    ):
        coords = self if coords is None else coords
        if issubclass(type(coords),ConformationsABC):
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
            min_loss_change_per_sample=min_loss_change_per_sample,
            max_iterations=max_iterations,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            r_c=r_c,
            long_scale=long_scale
        )

        return coords

    ########################################################
    # Visualization
    def plot(
        self,
        selection=0, # integer index, 'mean', 'mean_by_bond_separation', 'std', or 'var'
        *,
        fig=None,
        ax=None,
        cmap='RdBu',
        xticks=[],
        xticklabels=None,
        yticks=[],
        yticklabels=None,
        xlabel='Genomic index',
        ylabel='Genomic index',
        cbar_label=None,
        cbar_orientation=None,
        **kwargs
    ):
        
        if len(self) == 1:
            if cbar_label is None:
                cbar_label = self.__description
            return plot_dist_map(
                self.values[0,...].squeeze().cpu(),
                fig=fig,
                ax=ax,
                cmap=cmap,
                xticks=xticks,
                xticklabels=xticklabels,
                yticks=yticks,
                yticklabels=yticklabels,
                xlabel=xlabel,
                ylabel=ylabel,
                cbar_label=cbar_label,
                cbar_orientation=cbar_orientation,
                **kwargs
            )
            
        if type(selection) == int:
            to_plot = self[selection]

        elif selection == 'mean':
            to_plot = self.mean

        elif selection == 'median':
            to_plot = self.median

        elif selection == 'mean_by_bond_separation':
            to_plot = self.mean_by_bond_separation

        elif selection == 'std':
            to_plot = self.std

        elif selection == 'var':
            to_plot = self.var
        
        else:
            raise Exception(f"Selection should be an integer index, 'mean', 'mean_by_bond_separation', 'std', or 'var'")

        return to_plot.plot(
            fig=fig,
            ax=ax,
            cmap=cmap,
            xticks=xticks,
            xticklabels=xticklabels,
            yticks=yticks,
            yticklabels=yticklabels,
            xlabel=xlabel,
            ylabel=ylabel,
            cbar_label=cbar_label,
            **kwargs
        )

    def plot_with(
        self,
        other,
        *,
        fig=None,
        ax=None,
        cmap='RdBu',
        xticks=[],
        xticklabels=None,
        yticks=[],
        yticklabels=None,
        xlabel='Genomic Index',
        ylabel='Genomic Index',
        cbar_label=None,
        cbar_orientation=None,
        **kwargs
    ):
        assert type(other) == type(self), f'Expected {type(self)} object as input \'other\', but received {type(other)}'
        
        assert len(self) == 1 and len(other) == 1, 'Both Distances objects to be compared should have length 1, '+\
        f'but self has length {len(self)} and other has length {len(other)}'

        n = self.num_beads
        assert n == other.num_beads, 'Both distances objects have an equal number of beads, but '+\
        f'self has {n} while other has {other.num_beads}'

        to_plot = self.clone().flatten()
        i,j = torch.triu_indices(n,n,1)
        to_plot.values[0,j,i] = other.flatten().values[0,i,j]

        return to_plot.plot(
            fig=fig,
            ax=ax,
            cmap=cmap,
            xticks=xticks,
            xticklabels=xticklabels,
            yticks=yticks,
            yticklabels=yticklabels,
            xlabel=xlabel,
            ylabel=ylabel,
            cbar_label=cbar_label,
            cbar_orientation=cbar_orientation,
            **kwargs
        )

    def plot_dist_vs_separation(
        self,
        *,
        ax=None,
        xlabel='Bond Separation',
        ylabel='Distance',
        **kwargs
    ):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = None

        to_visualize = self.mean_by_bond_separation
        n = to_visualize.num_beads
        ax.plot(range(1,n),to_visualize.values[0,1:],**kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax

    ########################################################
    # Fold/unfold 
    def fold_(self):
        if self.__is_folded:
            return self
        self.__dist_maps = self.__origami_transform(self.values.unsqueeze(-3))
        self.__is_folded = True
        return self
    
    def fold(self):
        if self.__is_folded:
            return self
        return self.clone().fold_()

    def unfold_(self):
        if not self.__is_folded:
            return self
        self.__dist_maps = self.__origami_transform.inverse(self.values,2*self.values.shape[-1]).squeeze(-3)
        self.__is_folded = False
        return self

    def unfold(self):
        if not self.__is_folded:
            return self
        return self.clone().unfold_()
        

    ########################################################
    # Converting between sample subclasses
    
    # Always be able to return coordinates object
    @property
    def uncorrected_coordinates(self):
        return Coordinates(dists_to_coords(self.values),drop_invalid_conformations=False)
        
    @property
    def coordinates(self,drop_invalid_conformations=False):
        '''
        Optimize coordinates with default optimization values if necessary
        '''
        u_coords = self.uncorrected_coordinates
        if u_coords == self: # Essentially, checks is_exact
            return u_coords
        coords = self.correct_coordinates(u_coords) # This isn't working for some reason
        #coords = self.correct_coordinates()
        return Coordinates(coords,drop_invalid_conformations=drop_invalid_conformations)

    # Always be able to return trajectory object
    @property
    def trajectory(self): 
        return self.coordinates.trajectory

    # Always be able to return distance maps 
    @property
    def distances(self):
        return self



from Coordinates import Coordinates
