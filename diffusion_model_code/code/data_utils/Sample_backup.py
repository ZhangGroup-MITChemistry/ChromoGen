'''
This simplifies the process of manipulating data in its distance map, normalized distance map, etc., forms. 

Can currently output scHi-C maps. Will add a function to return 3D structures eventually too. 

The get_HiC function at the end is how I currently create Hi-C maps/plots. This probably makes the most sense 
to put into the GaussianDiffusion class, though it might also be best in the Sample class. 
'''

import torch
import numpy as np

# Used when inferring 3D structures. 
class DistLoss(torch.nn.Module): 

    def __init__(self,dists,self_interaction_included=True):
        super().__init__()

        # ignore diagonal if it's self-interaction
        i,j = torch.triu_indices(dists.shape[-2],dists.shape[-1],
                                               int(self_interaction_included)) 
        self.dists = dists[...,i,j].squeeze()

        self.triu_indices = torch.triu_indices(
            dists.shape[-2]+1-int(self_interaction_included),
            dists.shape[-2]+1-int(self_interaction_included),
            1
        )

    def get_dists(self,coords):
        return torch.nn.functional.pdist(coords)
    
    def bond_strength(self):
        '''
        could add molecular interactions as well
        '''

        

    def forward(self,coords):
        i,j = self.triu_indices
        return (self.dists - torch.cdist(coords,coords)[...,i,j] ).abs().sum()



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

    @property
    def dtype(self):
        return self.dist_std.dtype

    @property
    def device(self):
        return self.dist_std.device

    @property
    def seg_len(self): 
        return len(self.dist_std)

    #@property
    #def data(self): 
    #    return self.batch 

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

    ''' # Must still add this functionality 
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
    '''

    def _unnormalize_(self): 
        # Dists must be provided in flattened form
        sep,dists = self.sep, self.batch.clone()
        
        dists*= self.complement_inv_beta_sigmoid[sep].expand(*dists.shape[:-1],-1)
        dists+= self.inv_beta_sigmoid[sep].expand(*dists.shape[:-1],-1)
        dists.logit_()
        dists*= self.dist_std[sep].expand(*dists.shape[:-1],-1)
        dists+= self.inv_beta[sep].expand(*dists.shape[:-1],-1)
        
        '''
        dists*= self.complement_inv_beta_sigmoid[sep].repeat(*dists.shape[:-1],1)
        dists+= self.inv_beta_sigmoid[sep].repeat(*dists.shape[:-1],1)
        dists.logit_()
        dists*= self.dist_std[sep].repeat(*dists.shape[:-1],1)
        dists+= self.inv_beta[sep].repeat(*dists.shape[:-1],1)
        '''
        self.normalized = False
        self.batch = dists
    
    def unnormalize_(self):#,batch=None,is_flat=None,return_original_dtype=True):

        #if batch is not None:
        #    self.set_data(batch,is_flat,return_original_dtype)

        #assert self.batch_seg_len <= self.seg_len, \
        #f'mean/variance data insufficient for data with {self.batch_seg_len} genomic bins.'
        
        if self.normalized: # Only perform these operations if the data is normalized
            if self.is_flat: 
                self._unnormalize_(self.batch)
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
    # Getting 3D structures/removing bad structures
    def _dist_to_coord_(self,i):
        '''
        Assumes self interactions are excluded from the map 
        '''
        # Ensure we have distances in their unnormalized form
        self.unnormalize_()
    
        # For covenience...
        dists = self.batch[i,0,...]
        
        # Initialize the coordinates object 
        b,dt,dev = dists.shape[-1]+1, self.dtype, self.device
        coords = torch.empty(b,3,dtype=dt,device=dev)
        coords[:] = torch.nan 
        
        # Place bead 0 at the origin 
        coords[0,:] = 0 
        
        # Place bead 1 on the x axis 
        coords[1,0] = dists[0,0]
        coords[1,1:] = 0 
        
        # Get other x coordinates
        #coords[2:,0] = (dists[0,1]**2 + dists[0,2:]**2 - dists[1,2:]**2) / (2*dists[0,1]**2)
        coords[2:,0] = ( 1 + (dists[0,1:]**2 - dists[1,1:]**2)/dists[0,0]**2 ) / 2
        
        # Place bead 2 in the xy plane with positive y 
        coords[2,1] = ( dists[0,1]**2 - coords[2,0]**2 ).sqrt()
        coords[2,2] = 0 
        
        # Get other y coordinates
        coords[3:,1] = 1
        coords[3:,1]+= ( dists[0,2:]**2 - coords[3:,0]**2 - dists[2,2:]**2 + (coords[3:,0] - coords[2,0])**2 ) / (dists[0,1]**2 - coords[2,0]**2)
        coords[3:,1]/= 2
        
        # Give bead 3 a positive z value 
        coords[3,2] = ( dists[0,2]**2 - coords[3,:2].square().sum() ).sqrt() 
        
        # Get other z coordinates
        coords[4:,2] = dists[0,3:]**2 - dists[3,3:]**2
        coords[4:,2]+= (coords[4:,:2] - coords[3,:2]).square().sum(1) - coords[4:,:2].square().sum(1)
        coords[4:,2]/= coords[3,2]**2
        coords[4:,2]+= 1
        coords[4:,2]/= 2
        
        return coords
    
    def _adjust_coords_(
        self,
        coords,
        n_it=1000,
        lr=0.01
        
    ):
        '''
        Given an initial guess of coordinate locations and the
        distance map predicted by the diffusion model, adjust
        the coordinates to best match the predicted map. 
        '''
    
        dist_loss = DistLoss(self.batch,False)
    
        coords.requires_grad_(True)
    
        optimizer = torch.optim.Adam([coords],lr=lr) 
    
        for _ in range(n_it):
            optimizer.zero_grad()
            loss = dist_loss(coords)
            loss.backward()
            optimizer.step()
        
        coords.requires_grad_(False) 
        
        return coords

    def get_coords(self,n_it=1000,lr=0.01):
    
        # Get the coordinates for each sample
        coords = torch.empty(
            len(self),self.batch.shape[-1]+1,3, # Shape 
            dtype=self.dtype,
            device=self.device
        )

        # Get initial guess of coordinates based on subset of map 
        for i in range(len(self)): 
            coords[i,...] = self._dist_to_coord_(i)
        
        # Refine the coordinates to best match all distances 
        if n_it > 0:
            coords = self._adjust_coords_(coords,n_it,lr)

        # Remove NaN samples 
        coords = coords[~coords.isnan().any(1).any(1),...]
        self.coords = coords
        
        return coords.to(self.batch_dtype)

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
            return self._hic_via_tanh_(r_c,sigma)
    
        return self.get_scHiC(threshold).mean(0).squeeze() 

    
    
