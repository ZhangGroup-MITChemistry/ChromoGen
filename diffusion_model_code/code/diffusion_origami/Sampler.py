import os
import pandas as pd
import matplotlib.pyplot as plt
import torch

import sys
sys.path.insert(1,'../data_utils/')
from Sample import Sample 
from HiCDataset import HiCDataset
from EmbeddedRegions import EmbeddedRegions
from OrigamiTransform import OrigamiTransform
from Sample import Sample
sys.path.insert(2,'../diffusion_origami/')

import pickle
origami_transform = OrigamiTransform()

def is_square(mat):
    s = mat.shape
    assert len(s) > 1, f'Tensor must have at least two dimensions. Provided shape: {s}'
    return s[-1] == s[-2]

def copy_matrix_elements(from_mat,to_mat):
    assert is_square(from_mat) and is_square(to_mat), f'Both matrices must be square in the final two dimensions'

    n1,n2= from_mat.shape[-1],to_mat.shape[-1]
    n,N = min(n1,n2),max(n1,n2)

    i,j = torch.triu_indices(n,n,0)

    shift1,shift2 = max(0,n1-n2),max(0,n2-n1)
    to_mat[...,i,shift2+j] = from_mat[...,i,shift1+j]
    to_mat[...,shift2+j,i] = from_mat[...,shift1+j,i]
    

def add_diagonal(mat,diag_value=0):
    assert is_square(mat), f'Tensor must be square in the final two dimensions. Provided shape: {s}'
    
    N = mat.shape[-1] + 1
    new_mat = torch.empty(
        *mat.shape[:-2],N,N,
        dtype=mat.dtype,
        device=mat.device
    )
    
    copy_matrix_elements(mat,new_mat)
    i = torch.arange(N)
    new_mat[...,i,i] = diag_value

    return new_mat

def remove_diagonal(mat):
    assert is_square(mat), f'Tensor must be square in the final two dimensions. Provided shape: {s}'

    N = mat.shape[-1] - 1
    new_mat = torch.empty(
        *mat.shape[:-2],N,N,
        dtype=mat.dtype,
        device=mat.device
    )
    
    copy_matrix_elements(mat,new_mat)
    
    return new_mat

class Sampler: 

    def __init__(
        self,
        diffusion=None,#diffusion,
        embeddings=None,#embeddings,
        cond_scale=5.,
        rescaled_phi=0.5,
        nsamples=100
    ):

        self.diffusion = diffusion
        self.embeddings = embeddings
        self.cond_scale = cond_scale
        self.rescaled_phi = rescaled_phi
        self.nsamples = nsamples

    def __call__(
        self,
        region_index=None,
        *,
        embedding=None,
        nsamples = None,
        diffusion=None,
        embeddings=None,
        cond_scale=None,
        rescaled_phi=None
    ):
        assert (region_index is None) ^ (embedding is None), "Must pass EITHER region_index OR embedding"

        # Set the necessary options
        nsamples = self.nsamples if nsamples is None else nsamples
        diffusion = self.diffusion if diffusion is None else diffusion
        embeddings = self.embeddings if embeddings is None else embeddings
        cond_scale = self.cond_scale if cond_scale is None else cond_scale
        rescaled_phi = self.rescaled_phi if rescaled_phi is None else rescaled_phi

        # Get the desired embedding and expand it to the desired sample size
        emb = embedding if region_index is None else embeddings.iloc[region_index,0]
        emb = emb.to(diffusion.device).expand(nsamples,-1,-1,-1)

        # Sample with the diffusion model
        data = diffusion.sample(emb,cond_scale=cond_scale,rescaled_phi=rescaled_phi)

        # Reverse the origami transform
        data = origami_transform.inverse(data)

        # Remove the diagonal so that we can directly use the Sample class
        data = remove_diagonal(data) 

        # Make a Sample object
        sample = Sample(data=data)

        return sample

    def get_valid_attributes(self):
        valid_attributes = []
        for attr in dir(self): 
            if attr[0] != '_' and not callable(getattr(self, attr)): 
                valid_attributes.append(attr)
        return valid_attributes
    
    def set(self,**kwargs):
        '''
        Options: 
        diffusion
        embedded_regions
        cond_scale
        rescaled_phi
        nsamples
        '''
        assert len(kwargs) > 0, f'Expected at least one keyword argument.'
        attrs = self.get_valid_attributes()
        for attr,value in kwargs.items():
            assert attr in attrs, f'Error: Keyword argument {attr} not recognized. Valid attributes to set are: {attrs}.'
            assert (getattr(self,attr) is None) or type(value) == type(getattr(self,attr)), f'Expected type {type(getattr(self,attr))} for attribute {attr}, but received {type(value)}.'
        for attr, value in kwargs.items():
            setattr(self, attr, value)

def get_dist_env():
    if 'OMPI_COMM_WORLD_SIZE' in os.environ:
        world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
    else:
        world_size = int(os.getenv('SLURM_NTASKS'))

    if 'OMPI_COMM_WORLD_RANK' in os.environ:
        global_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
    else:
        global_rank = int(os.getenv('SLURM_PROCID'))
    return global_rank, world_size

if __name__ == "__main__":

    ############################################
    # Parallelization stuff
    ############################################
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    import socket
    
    global_rank, world_size = get_dist_env()
    hostname = socket.gethostname()
    # You have run dist.init_process_group to initialize the distributed environment
    # Always use NCCL as the backend. Gloo performance is pretty bad and MPI is currently
    # unsupported (for a number of reasons).
    dist.init_process_group(backend='nccl', rank=global_rank, world_size=world_size)

    # Set the cuda devices to avoid duplicate device in nccl
    torch.cuda.set_device(global_rank%2)
    ############################################

    #import sys
    #sys.path.insert('./')
    from classifier_free_guidance_greg import Unet, GaussianDiffusion

    unguided = False
    guided = True#False

    chroms = ['1','X']
    segment_length = 64
    c,image_size = 2,segment_length//2
    cond_scales = [.1,.2,.3,.4,.5,.6,.7,.8,.9]#[.25,.5,.75,1.,2.,3.,4.,5.]#[.5,1.,2.,3.,4.,5.,6.]
    rescaled_phis = [0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]
    nsamples_unguided = 10000
    nsamples = 500
    #nsamples = 1000
    #nsamples = 10000

    # All of these regions have 30 structures (EXCLUDING replicates) in chrom 1, while 
    # the chrom X regions have 32 structures for comparison (16 cells, mat/pat chromosomes) 
    regions_dict = {
        '1':[330,395],#[144,200],#,265,330,395,460,525,590,730,795,860,1260,1325],
        'X':[]#[100,236],#,381,445,553,610,675,810,900,965,1060,1125,1200]
    }

    #### 
    #model_folder = '../../data/models/diffusion_small_origami'#'./results_small'
    model_folder = '../../data/models/diffusion_small_origami/'#diffusion_small_origami_hg19_via_hg38_weights/'
    config_fp = '../../data/processed_data.hdf5'
    #embedding_dir = '../../data/embeddings/'
    #embedding_dir = '../../data/embeddings_hg19_via_hg38_params/'
    embedding_dir = '../../data/embeddings_65'
    mean_dist_fp = '../../data/mean_dists.pt'
    mean_sq_dist_fp='../../data/squares.pt'
    save_dir = '../../data/samples/origami_final_embeddings/'
    ###

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    def fp(region_idx,cond_scale,rescaled_phi,milestone,chrom,save_dir=save_dir):
        return save_dir + f'sample_{region_idx}_{cond_scale}_{rescaled_phi}_{milestone}_{chrom}.pkl'

    def get_existing_sample(filepath):
        if os.path.exists(filepath):
            sample = pd.read_pickle(filepath)
            sample = Sample(data = sample.unflatten())
            return sample
        else:
            return None

    def save_sample(sample,filepath):

        sample1 = get_existing_sample(filepath)
        if sample1 is not None: 
            data1 = sample1.unflatten().to(device=sample.device,dtype=sample.dtype)
            if len(data1.shape) < 3: 
                data1 = data1.unsqueeze(0)
            data2 = sample.unflatten()
            if len(data2.shape) < 3: 
                data2 = data2.unsqueeze(0)

            data = torch.cat((data1,data2),dim=0)
        else:
            data = sample.unflatten()
        sample = Sample(data=data.cpu(),device='cpu')
            
        pickle.dump(sample,open(filepath,'wb'))

    def get_nsamples(filepath,nsamples=nsamples):
        sample = get_existing_sample(filepath)
        if sample is None:
            return nsamples
        return max(0,nsamples-len(sample))

    def load_most_recent_milestone(diffusion,folder):

        if folder != '' and folder[-1] != '/': 
            folder = folder + '/'
    
        files = os.listdir(folder)
        milestone = 0 
        file = ''
        for f in files: 
            m = int('.'.join(f.split('.')[:-1]).split('-')[-1])
            if m > milestone: 
                milestone = m
                file = f
        
        diffusion.load(folder + file)
    
        return milestone
    
    # Initializing model
    print('Initializing model',flush=True)
    model = Unet(
        dim=64,
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
        embedding_dimensions=(1,260,256)#tuple(er.ifetch(0)[0].shape)
    ).to(f'cuda:{global_rank%2}')
    

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
    ).to(f'cuda:{global_rank%2}')

    # Loading most recent model parameters
    print('Loading model parameters',flush=True)
    milestone = load_most_recent_milestone(diffusion,model_folder) 
    
    ####### ALLOW THE NETWORK TO RUN ON GPUS IN PARALLEL ##############
    # % 2 because each node has 2 gpus locally referred to as 0 & 1
    # diffusion = DDP(diffusion, device_ids=[global_rank%2], output_device=None)
    ###################################################################

    # Loading embeddings
    print('Loading embeddings',flush=True)
    embeddings_dict = {}
    for chrom in chroms:
        embeddings_dict[chrom] = pd.read_pickle(f'{embedding_dir}/chrom_{chrom}.tar.gz')

    # Preparing Sampler
    print('Preparing Sampler',flush=True)
    sampler = Sampler(diffusion=diffusion)

    # If unguided is desired, generate that! 
    if unguided:
        print('Unguided Samples')
        f = save_dir + f'sample_unguided_{milestone}.pkl'
        nsamples_still_needed = get_nsamples(f,nsamples_unguided)
        while nsamples_still_needed > 0:
        #if nsamples_still_needed > 0:
            temp_nsamples = min(1000,nsamples_still_needed)
            sampler.set(embeddings=embeddings_dict[chroms[0]],cond_scale=0.,rescaled_phi=0.,nsamples=temp_nsamples)
            sample = sampler(0)
            save_sample(sample,f)
            nsamples_still_needed = get_nsamples(f,nsamples_unguided)

    if not guided: 
        sys.exit()
    # Sample loop
    print('Sampling Loop',flush=True)
    for chrom in chroms: 
        regions = regions_dict[chrom]
        embeddings = embeddings_dict[chrom]

        sampler.set(embeddings=embeddings)
        
        for cond_scale in cond_scales: 
            sampler.set(cond_scale=cond_scale)
            rescaled_phis1 = rescaled_phis if cond_scale != 1 else [0.]
            for rescaled_phi in rescaled_phis1: 
                sampler.set(rescaled_phi=rescaled_phi)
                for region_idx in regions:
                    # Get the relevant filepath
                    f = fp(region_idx,cond_scale,rescaled_phi,milestone,chrom)

                    # If some samples already exist, see how many more must be produced to obtain the desired
                    # number of total samples. 
                    nsamples_still_needed = get_nsamples(f,nsamples)
                    if nsamples_still_needed <= 0:
                        continue

                    # Generate those samples
                    sample = sampler(region_idx,nsamples=nsamples_still_needed)

                    # Save the sample 
                    save_sample(sample,f) 


