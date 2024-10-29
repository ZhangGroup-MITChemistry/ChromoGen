import torch
from tqdm.auto import tqdm
import numpy as np
import mdtraj as md
import os
import sys
sys.path.insert(1,'./')
from HiCMap import HiCMap 
from tensor_support import add_diagonal, remove_diagonal
from OrigamiTransform import OrigamiTransform
origami_transform = OrigamiTransform()

from Sample import Sample ## TEMP

############################################
# Functions to convert distance maps into coordinates
############################################

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

############################################
# Functions to convert coordinates into distances in a consistent manner with high precision
############################################

def to_cuda(tensor,device=None):
    if not torch.cuda.is_available():
        return tensor
    
    if device is not None: 
        # Device passed || device type passed || device index passed
        if (device == tensor.device)\
        or (device == tensor.device.type)\
        or (device == tensor.device.index and tensor.is_cuda):
            return tensor
        if type(device) == int: # Device index passed
            return tensor.cuda(device)
        return tensor.to(device)
        
    if tensor.is_cuda:
        return tensor
    return tensor.cuda()

def coords_to_dists(coords,use_cuda=True,dtype=None):
    return_device = coords.device
    return_dtype = coords.dtype
    if use_cuda:
        coords = to_cuda(coords)
    if dtype is None: 
        dtype = torch.double
    coords = coords.to(dtype)
    
    dists = torch.cdist(coords,coords)
    i = range(dists.shape[-1])
    dists[...,i,i] = 0 # cdist sometimes provides errantly small-but-nonzero values on the diagonal

    return dists.to(dtype=return_dtype,device=return_device)
    

############################################
# Functions to make coordinates better match generated distance maps
############################################
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

''' Original
def loss_fcn(coords,target_dists,r_c=1.0,long_scale=1/8):
    #dists = coords_to_dists(coords,increase_precision=False)
    dists = torch.cdist(coords,coords)
    i,j = torch.triu_indices(dists.shape[-1],dists.shape[-1],1)
    return smooth_transition_loss(dists[...,i,j],target_dists[...,i,j])
'''
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

############################################
# Functions to create dcd/psf topology files from coordinate data
############################################
def homopolymer_psf(num_beads,filepath):
    # Adapted from code provided by Xinqiang Ding
    lines = [
        'PSF CMAP CHEQ XPLOR',
        '',
        '{:>8d} !NTITLE'.format(1),
        '* HOMOPOLYMER PSF FILE',
        '',
        '',
        '{:>8d} !NATOM'.format(num_beads)
    ]

    # Atoms
    lines.extend([
        '{:>8d} {:<4s} {:<4d} {:<4s} {:<4s} {:<4s} {:<14.6}{:<14.6}{:>8d}{:14.6}'.format(
         i,     'POL', 1,     'POL', 'C',   'C',   0.0,    0.0,    0,    0.0
        ) for i in range(1,num_beads+1)
    ])

    # Bonds
    lines.extend([
        '',
        '{:>8d} !NBOND: bonds'.format(num_beads-1)
    ])
    for i in range(1,num_beads+1,4):
        line = ''
        for j in range(4):
            if i + j == num_beads:
                break
            line+= '{:>8d}{:>8d}'.format(i+j,i+j+1)
        lines.append(line)

    # Angles, dihedrals, & impropers
    lines.extend([
        '',
        '{:>8d} !NTHETA: angles'.format(0),
        '',
        '',
        '{:>8d} !NPHI: dihedrals'.format(0),
        '',
        '',
        '{:>8d} !NIMPHI: impropers'.format(0),
        ''
    ])

    with open(filepath,'w') as f:
        f.write('\n'.join(lines))

def coord_to_xyz(coords,filepath):
    coords = coords.squeeze()
    assert coords.ndim < 3
    assert coords.shape[-1] == 3
    while coords.ndim < 2:
        coords.unsqueeze(0)

    n_atoms = coords.shape[0]
    lines = [
        f'{n_atoms}',
        ''
    ]
    lines.extend([
        '\t'.join([
            'C',
            f'{coords[i,0]}',
            f'{coords[i,1]}',
            f'{coords[i,2]}'
        ]) for i in range(n_atoms)
    ])
    with open(filepath,'w') as f:
        f.write('\n'.join(lines))

def coord_to_dcd(coords,dest_dir,sample_name):

    # Bring the coordinates to a standardized shape.
    assert coords.shape[-1] == 3
    if coords.ndim == 1:
        coords.unsqueeze(0)
    if coords.ndim == 2:
        coords.unsqueeze(0)
    coords = coords.flatten(-coords.ndim,-3) # Combine batch dimensions
    n_molecules = coords.shape[0]
    n_atoms = coords.shape[1]

    # Create destination directory, if necessary
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Create the psf file
    psf_filepath = dest_dir + f'/{sample_name}.psf'
    homopolymer_psf(coords.shape[-2],psf_filepath)

    # Find un-reserved temp dir for the intermediate xyz files
    temp_xyz_dir = dest_dir + '/temp/'
    i=0
    while os.path.exists(temp_xyz_dir):
        temp_xyz_dir = dest_dir + f'/temp{i}/'
        i+=1
    os.makedirs(temp_xyz_dir)

    # Create trajectories & consolidate into a single trajectory, in order
    xyz_files = [temp_xyz_dir + f'mol_{mol}.xyz' for mol in range(n_molecules)]
    for mol,fp in enumerate(xyz_files):
        coord_to_xyz(coords[mol,...],fp)

    t = md.load(xyz_files,top=psf_filepath)

    # Remove the now-unneeded xyz files
    for fp in xyz_files:
        os.remove(fp)
    os.rmdir(temp_xyz_dir)

    # Create the dcd file
    dcd_filepath = dest_dir + f'/{sample_name}.dcd'
    t.save_dcd(dcd_filepath)

    return t

############################################
# Load samples more consistently
############################################
def load_dist_maps(filepath,map_location='cuda' if torch.cuda.is_available() else 'cpu'):

    try:
        data = torch.load(filepath,map_location=map_location)
    except:
        data = pd.read_pickle(filepath).unflatten()

        if type(data) != torch.Tensor:
            data = torch.tensor(data)
            
        data = data.to(map_location)
    
    if len(data.shape) == 2:
        data = data.unsqueeze(0)
    if len(data.shape) == 3:
        if data.shape[0] > 2: # Batch size is in the second channel
            data = data.unsqueeze(1)
        else: # MOST LIKELY, batch size is 1 & this dimension corresponds to channels
            data = data.unsqueeze(0)
    assert data.shape[-2] == data.shape[-1], f"Expected square matrix in final two dimensions, but received shape {data.shape}"

    if data.shape[-3] == 2:
        data = origami_transform.inverse(data)
        data = remove_diagonal(data)
    
    sample = Sample(data = data) 
    sample.unnormalize_()
    
    return add_diagonal(sample.unflatten())
    

############################################
# Sample class -- Still need to work on this
############################################
'''
class Sample: 
    def __init__(
        self,
        mean_dist_fp='../../data/mean_dists.pt',
        mean_sq_dist_fp='../../data/squares.pt',
        dtype=torch.double,
        seg_len = None, # Number of beads
        device = None,
        preserve_asymmetries=True,
        data = None,
        data_is_flat = None,
        preserve_data_dtype = True
    ): 

        # Assume GPU is desired unless specified otherwise
        if device is None: 
            try:
                device = torch.empty(1).cuda().device
            except:
                device = torch.empty(1).device

        mean_dist = torch.load(mean_dist_fp,map_location=device).flatten().to(dtype)
        mean_square_dist = torch.load(mean_sq_dist_fp,map_location=device).flatten().to(dtype)
        if seg_len is not None: 
            mean_dist = mean_dist[:seg_len]
            mean_square_dist = mean_square_dist[:seg_len]
        
        self.dist_std = (mean_square_dist - mean_dist**2).sqrt()
        self.inv_beta = torch.sqrt( 2*mean_square_dist/3 )
        self.inv_beta_sigmoid = torch.sigmoid( -self.inv_beta/self.dist_std )
        self.complement_inv_beta_sigmoid = 1 - self.inv_beta_sigmoid

        self.preserve_asymmetries = preserve_asymmetries
        if data is not None: 
            self.set_data(data,data_is_flat,preserve_data_dtype)
        else: 
            self.batch = None

        self.__coords = None

    @property
    def dtype(self):
        return self.dist_std.dtype

    @property
    def device(self):
        return self.dist_std.device

    @property
    def seg_len(self): 
        return len(self.dist_std)

    @property
    def coords(self): # use separate property so coords can't be set directly by outside users
        return self.__coords

    def __len__(self): 
        if self.batch is None: 
            return 0
        elif len(self.batch.shape) == 1:
            return 1
        elif (len(self.batch.shape) == 2) and (not self.is_flat):
            return 1
        else:
            return self.batch.shape[0]
        
    def to(self,*args):
        for attr in dir(self): 
            if type(getattr(self, attr)) == torch.Tensor:
                for arg in args:
                    setattr(self, attr, getattr(self, attr).clone().to(arg))

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda') 

    #####
    # Processing data
    def _infer_seg_len_(self): 
        # sample_len is an integer with length N(N-1)//2 
        if self.is_flat is None: 
            # Must infer matrix vs flattened form 
            if (len(self.batch.shape) == 1) or (self.batch.shape[-1] != self.batch.shape[-2]):
                self.is_flat = True
            else: 
                # Assumes batch.shape[-1] == batch.shape[-2] only occurs if in matrix form, 
                # which is overwhelmingly likely. 
                self.is_flat = False 
        
        if self.is_flat:
            M = self.batch.shape[0] 
            NN = (1+np.sqrt(1+8*M))/2 # Quadratic formula
            N = int(NN)
            assert N == NN, f'Invalid batch size. Cannot infer number of beads in the segment!'
        else: 
            assert len(self.batch.shape) > 1, \
            'User indicated that the provided batch was not flattened, but one-dimensional data was provided!'
            N = self.batch.shape[-1]

        # Finally, compute the number of beads in the segment. 
        self.batch_seg_len = N 
        self.triu_indices = torch.triu_indices(N,N,0)
        self.sep = self.triu_indices[1] - self.triu_indices[0] 

    def flatten(self): 
        
        if self.is_flat:
            return self.batch.clone()

        i,j = self.triu_indices
        return self.batch[...,i,j].clone()
        
    
    def flatten_(self,force=False):

        if self.is_flat:
            return 

        b = self.batch
        if not force and self.preserve_asymmetries and (b != b.transpose(-2,-1)).any():
            return # Flattening would cause us to lose the asymmetry in the matrix

        i,j = self.triu_indices

        self.batch = b[...,i,j]
        self.is_flat = True

    def unflatten(self):

        if not self.is_flat:
            return self.batch.clone() # Already in matrix form

        batch = torch.empty(*self.batch.shape[:-1],self.batch_seg_len,self.batch_seg_len,dtype=self.dtype,device=self.device)
        i,j = self.triu_indices

        batch[...,i,j] = self.batch 
        batch[...,j,i] = self.batch 
        
        return batch 
    
    def unflatten_(self):

        self.batch = self.unflatten()
        self.is_flat = False 
    
    
    def set_data(self,batch,is_flat=None,return_original_dtype=True):
        # is_flat == True: Last dimension contains the upper triangle of the distance matrix. 
        # is_flat == False: Data is still in matrix form, in the final 2 dimensions. 
        # is_flat is None: Must infer whether data is in the matrix or flattened form.

        # Convert numpy objects to torch tensors
        if type(batch) == np.ndarray: 
            batch = torch.from_numpy(batch) 

        # Validate input. 
        assert type(batch) == torch.Tensor, \
        f'The batch argument must be a torch.Tensor. Received {type(batch)}'
        
        assert type(is_flat)==bool or is_flat is None, \
        f'The is_flat argument must be one of True, False, or None. Received {type(is_flat)}'
        
        assert type(return_original_dtype) == bool, \
        f'The return_original_dtype argument must be either True or False. Received {type(return_original_dtype)}.'

        # Save the data to this DataProcessor object
        self.batch_dtype = batch.dtype if return_original_dtype else self.dtype
        self.batch = batch.clone().to(self.device,self.dtype)
        self.is_flat = is_flat
        self._infer_seg_len_() # Determines the number of genomic loci in the map
        self.flatten_() # Reduces memory usage & computational requirements. 

        # Infer whether this is in the normalized or distance form 
        self.normalized = (self.batch <= 1).all()

        self.coords = None 

    ''#' # Must still add this functionality 
    def normalize_dists(self,dists):
        if not self.norm_dists:
            return dists
        sep = self.sep_idx
        i,j = self.triu_indices
        bs = dists.shape[0] #self.batch_size
        j = j-1
        dists-= self.inv_beta[sep].repeat(bs,1) # Should eventually replace with expand to save memory 
        dists/= self.dist_std[sep].repeat(bs,1)
        dists.sigmoid_()
        dists-= self.inv_beta_sigmoid[sep].repeat(bs,1)
        dists/= self.complement_inv_beta_sigmoid[sep].repeat(bs,1)
        return dists 
    ''#'

    def _unnormalize_(self): 
        # Dists must be provided in flattened form
        #if self.normalize 
        
        sep,dists = self.sep, self.batch.clone()
        
        dists*= self.complement_inv_beta_sigmoid[sep].expand(*dists.shape[:-1],-1)
        dists+= self.inv_beta_sigmoid[sep].expand(*dists.shape[:-1],-1)
        dists.logit_()
        dists*= self.dist_std[sep].expand(*dists.shape[:-1],-1)
        dists+= self.inv_beta[sep].expand(*dists.shape[:-1],-1)
        
        ''#'
        dists*= self.complement_inv_beta_sigmoid[sep].repeat(*dists.shape[:-1],1)
        dists+= self.inv_beta_sigmoid[sep].repeat(*dists.shape[:-1],1)
        dists.logit_()
        dists*= self.dist_std[sep].repeat(*dists.shape[:-1],1)
        dists+= self.inv_beta[sep].repeat(*dists.shape[:-1],1)
        ''#'
        self.normalized = False
        self.batch = dists
    
    def unnormalize_(self):#,batch=None,is_flat=None,return_original_dtype=True):

        #if batch is not None:
        #    self.set_data(batch,is_flat,return_original_dtype)

        #assert self.batch_seg_len <= self.seg_len, \
        #f'mean/variance data insufficient for data with {self.batch_seg_len} genomic bins.'
        
        if self.normalized: # Only perform these operations if the data is normalized
            if self.is_flat: 
                self._unnormalize_()#self.batch)
            else:
                # We must contend with the asymmetric data
                batch = torch.empty_like(self.batch)
                
                i,j = torch.triu_indices(self.batch_seg_len,self.batch_seg_len,0)
                #i,j = self.triu_indices
                b = self.batch
                for ii,jj in [(i,j),(j,i)]: 
                        
                    self.batch = b[...,ii,jj]
                    self._unnormalize_()
                    batch[...,ii,jj] = self.batch 
                self.batch = batch 

            self.normalized = False 
            
        return self.unflatten()

    ############################
    # Getting 3D structures/repairing structures
    def compute_coordinates(self):

        if self.__coords is not None:
            # Already computed
            return
        
        assert self.batch is not None, "No data loaded into Sample object!"

        if self.__coords
        
        
        

    def get_scHiC(self,threshold=2):

        # Convert data to distances, unflatten, and compare to the threshold
        return (self.unnormalize_() < threshold).to(self.batch_dtype)

    def _hic_via_tanh_(self,r_c,sigma):

        self.unnormalize_()
    
        #r = self.batch.clone() # Distances
        if self.coords is None: 
            self.get_coords()

        r = torch.cdist(self.coords,self.coords)
        
        
        mask = r < r_c 
        r[mask] = .5*( 1 + torch.tanh( sigma*( r_c - r[mask] ) ) )
    
        mask^= True 
        r[mask] = .5 * ( r_c / r[mask] )**4
    
        return r.mean(0).squeeze().to(self.batch_dtype) 
    
    def contact_probabilities(self,approach='tanh',threshold=2,r_c=2,sigma=3): # to get Hi-C-like maps 

        if approach == 'tanh': 
            mat = self._hic_via_tanh_(r_c,sigma)
        elif approach == 'threshold':
            mat = self.get_scHiC(threshold).mean(0).squeeze()
        else: 
            raise Exception(f"approach {approach} is not recognized. Use 'tanh' or 'threshold'.")

        # Todo: Include genomic regions, etc., in the class so that they can be passed to the HiCMap object. 
        return HiCMap(mat) 

'''

