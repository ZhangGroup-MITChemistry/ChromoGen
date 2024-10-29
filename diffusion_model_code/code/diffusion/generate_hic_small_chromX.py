#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=generate_hic_small_chromX
##SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
##SBATCH -t 0-3:00:00
##SBATCH -t 0-36:00:00
#SBATCH --output=./log_files/generate_hic_small_chromX.log

import sys
sys.path.insert(0,'/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/code/diffusion/')
sys.path.insert(1,'/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/code/data_utils/')

from classifier_free_guidance_greg import GaussianDiffusion, Unet
from accelerate import Accelerator
import torch 
import os 

chrom = 'X'
embedding_dimensions = (1,260,256)
milestone = 120#37#69
#fp = f'./results_small/model-{milestone}.pt'
fp = f'../../data/models/diffusion_small/model-{milestone}.pt'
save_folder = '../../data/samples/small_model/'

num_beads = 65
two_channels = False

c,image_size = 1+int(two_channels), num_beads-1
model = Unet(
    dim=64,
    #num_classes,
    cond_drop_prob = 0.5,
    init_dim = None,
    out_dim = None,
    dim_mults=(1, 2, 4, 8),
    channels = c,
    resnet_block_groups = 8,
    learned_variance = False,
    learned_sinusoidal_cond = False,
    random_fourier_features = False,
    learned_sinusoidal_dim = 16,
    attn_dim_head = 32,
    attn_heads = 4,
    embedding_dimensions=embedding_dimensions
)

diffusion = GaussianDiffusion(
    model,
    image_size=image_size,
    timesteps = 1000,
    sampling_timesteps = None,
    objective = 'pred_noise',
    beta_schedule = 'cosine',
    ddim_sampling_eta = 1.,
    offset_noise_strength = 0.,
    min_snr_loss_weight = False,
    min_snr_gamma = 5
)

try: 
    diffusion.to('cuda')
except:
    pass

diffusion.load(fp)

import pandas as pd
embeddings = pd.read_pickle(f'../../data/embeddings/chrom_{chrom}.tar.gz')

import torch
import numpy as np

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
            self.data = data

    @property
    def dtype(self):
        return self.dist_std.dtype

    @property
    def device(self):
        return self.dist_std.device

    @property
    def seg_len(self): 
        return len(self.dist_std)

    def __len__(self): 
        if self.batch is None: 
            return 0
        elif len(self.data.shape) == 1:
            return 1
        elif (len(self.data.shape) == 2) and (not self.is_flat):
            return 1
        else:
            return self.data.shape[0]
        
    def to(self,*args):
        for attr in dir(self): 
            if type(getattr(self, attr)) == torch.Tensor:
                for arg in args:
                    setattr(self, attr, getattr(self, attr).clone().to(arg))

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

        assert self.batch_seg_len <= self.seg_len, \
        f'mean/variance data insufficient for data with {self.batch_self_len} genomic bins.'
        
        if self.normalized: # Only perform these operations if the data is normalized
            if self.is_flat: 
                self._unnormalize_(self.batch)
            else:
                # We must contend with the asymmetric data
                batch = torch.empty_like(self.batch)
                
                i,j = torch.triu_indices(self.batch_seg_len,self.batch_seg_len,0)
                b = self.batch
                for ii,jj in [(i,j),(j,i)]: 
                        
                    self.batch = b[...,ii,jj]
                    self._unnormalize_()
                    batch[...,ii,jj] = self.batch 
                self.batch = batch 

            self.normalized = False 
            
        return self.unflatten()

    def get_scHiC(self,threshold=2):

        # Convert data to distances, unflatten, and compare to the threshold
        return (self.unnormalize_() < threshold).to(self.batch_dtype)


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def get_HiC(diffusion,embedding,n_maps=1000,cond_scale = 6., rescaled_phi = 0.7,plot=True):

    if embedding.shape not in [torch.Size([1,1,260,256]),torch.Size([1,260,256])]:
        raise Exception(f"Expected embedding shape (1,260,256) or (1,1,260,256), but received {embedding.shape}")

    diffusion.eval()
    sample = Sample(
        data = diffusion.sample(embedding.to(diffusion.device).expand(n_maps,-1,-1,-1),cond_scale=cond_scale,rescaled_phi=rescaled_phi)
    )

    hic_map = sample.get_scHiC().mean(0)[0,...]

    if plot: 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.matshow(hic_map.cpu().numpy(),norm=LogNorm(vmin=hic_map[hic_map>0].min(),vmax=1),cmap='RdYlBu_r')
        cbar = fig.colorbar(im)
    
    return sample, hic_map, fig

def fp(save_folder,region,cond_scale,rescaled_phi,milestone,chrom):
    if cond_scale == 1: 
        rescaled_phi = 0 # Rescaled phi doesn't do anything when cond_scale is 1
    return save_folder+f'sample_{region}_{int(cond_scale)}_{int(10*rescaled_phi)}_{milestone}_{chrom}.pkl'

import pickle
n_maps = 500#1000
diffusion.eval()
n=0 
cond_scales = [1.,2.,4.,6.,8.]
rescaled_phis = [0.,.25,.5,.75,1.]

for region in range(0,1000,50): 
    for cond_scale in cond_scales:#[1,2,4,6,8]:#[float(k) for k in range(1,11)]:
        for rescaled_phi in rescaled_phis:#[.5]:#[k/10 for k in range(1,11)]: 
            if os.path.exists(fp(save_folder,region,cond_scale,rescaled_phi,milestone,chrom)):
                continue
            emb = embeddings.iloc[region,0].to(diffusion.device).expand(n_maps,-1,-1,-1)
            sample = Sample(
                data = diffusion.sample(emb,cond_scale=cond_scale,rescaled_phi=rescaled_phi)
            )
            sample.to('cpu') 
            #pickle.dump(sample,open(f'./sampling_small/sample_{region}_{int(cond_scale)}_{int(10*rescaled_phi)}_{milestone}_{chrom}.pkl','wb'))
            pickle.dump(
                sample,
                open(fp(save_folder,region,cond_scale,rescaled_phi,milestone,chrom),'wb')
                #open(save_folder+'sample_{region}_{int(cond_scale)}_{int(10*rescaled_phi)}_{milestone}_{chrom}.pkl','wb')
            )

            n+=1
            print(f'{n/2000}% completed')
            
            

'''
n = 0
n_maps = 1000
cond_scale = 6.
rescaled_phi = 0.7
emb = embeddings.iloc[n].values[0]
sample,hic_map,fig = get_HiC(diffusion,emb,n_maps,cond_scale,rescaled_phi)

fig.savefig('./results_small/hic_map.png')

import pickle
pickle.dump({'sample':sample,'hic_map':hic_map,'fig':fig},open('./results_small/generated_hic_map.pkl','wb'))
'''
