#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=train_small_origami_distributed
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:volta:2
#SBATCH -c 40
#SBATCH --nodes=1
#SBATCH --output=./log_files/train_small_distributed.log


###########################################################
# Import modules
###########################################################

# This should eventually be performed via relative imports
import os
import sys
sys.path.insert(0,'./') #/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/code/diffusion/')
from classifier_free_guidance_greg_distributed import GaussianDiffusion, Unet, Trainer
sys.path.insert(1,'/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/code/data_utils/')
from DataLoader import DataLoader
from ConfigDataset import ConfigDataset
from EmbeddedRegions import EmbeddedRegions

#####
# Specific to parallelization
#####
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import socket

def run(rank,world_size, hostname): # Only rank is used 
    ###########################################################
    # Set parameters 
    ###########################################################
    
    # Training data locations
    config_fp = '../../data/processed_data.hdf5'
    embedding_dir = '../../data/embeddings/'
    
    # Supporting data locations 
    mean_dist_fp = '../../data/mean_dists.pt'
    mean_sq_dist_fp='../../data/squares.pt'
    
    # Destination directory
    #save_folder = '../../data/models/diffusion_small_origami_distributed'#'./results_small'
    save_folder = './test_distributed'
    
    # Exclude chromosome X from training data so that it can be
    # used for network validation 
    training_chroms = ['X']#[f'{k}' for k in range(1,23)]
    
    # Training iteration details 
    segment_length = 64
    batch_size = 64
    shuffle_data = True
    
    
    #import pickle # Temporary 
    ###########################################################
    # Build the DataLoader with corresponding datasets. 
    ###########################################################
    
    print('Preparing Data',flush=True)
    print('Loading Configuration Dataset',flush=True)
    config_ds = ConfigDataset(
        config_fp,
        segment_length=segment_length,
        remove_diagonal=False,
        batch_size=0,
        normalize_distances=True,
        geos=None,
        organisms=None,
        cell_types=None,
        cell_numbers=None,
        chroms=training_chroms,
        replicates=None,
        shuffle=True,
        allow_overlap=True,
        two_channels=False,
        try_GPU=True,
        mean_dist_fp=mean_dist_fp,
        mean_sq_dist_fp=mean_sq_dist_fp
    )
    
    print('Loading Embeddings',flush=True)
    er = EmbeddedRegions(
        embedding_dir,
        chroms=training_chroms
    )
    
    print('Constructing DataLoader',flush=True)
    dl = DataLoader(
        config_ds,
        er,
        drop_unmatched_pairs=True, 
        shuffle = shuffle_data,
        batch_size=batch_size
    )
    print('Data Preparation Complete',flush=True)
    
    ####
    # Perform a quick check that I couldn't do with the memory restraints of the Jupyter Lab on SuperCloud
    chr = dl.index['Chromosome'].unique().tolist()
    assert len(chr) == len(training_chroms) 
    for k in training_chroms: 
        assert k in chr 
    
    #assert len(
    #assert 'X' not in dl.index['Chromosome'].unique().tolist()
    #pickle.dump(dl.index['Chromosome'].unique().tolist(),open('./results_small/chroms.pkl','wb'))
    ###########################################################
    # Build the diffusion model 
    ###########################################################
    
    #c,image_size,_ = tuple(config_ds.fetch((0,'mat'))[0].shape)
    c,image_size = 2,segment_length//2
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
        embedding_dimensions=tuple(er.ifetch(0)[0].shape)
    ).to(f'cuda:{rank%2}')
    
    #print(image_size,flush=True)
    # % 2 because each node has 2 gpus locally referred to as 0 & 1
    #model = DDP(model, device_ids=[rank%2], output_device=None)


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
    ).to(f'cuda:{rank%2}')#.cuda()

    # % 2 because each node has 2 gpus locally referred to as 0 & 1
    # diffusion = DDP(diffusion, device_ids=[rank%2], output_device=None)
    
    trainer = Trainer(
        diffusion,
        dl,
        train_batch_size = 128,
        gradient_accumulate_every = 1,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 600000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 5000,
        num_samples = 25,
        results_folder = save_folder,
        amp = False, # using false here turns off mixed precision 
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = False, #True,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False
    )

    if os.path.exists(save_folder): 
        milestone = 0 
        for k in os.listdir(save_folder):
            try: 
                n = int( k.split('-')[-1].split('.')[0] )
                if n > milestone: 
                    milestone = n
            except:
                pass
        if milestone > 0:
            trainer.load(milestone=milestone)
    
    # % 2 because each node has 2 gpus locally referred to as 0 & 1
    diffusion = DDP(diffusion, device_ids=[rank%2], output_device=None)

    trainer.train()

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
    # Dont change the following :
    global_rank, world_size = get_dist_env()
    
    hostname = socket.gethostname()

    # You have run dist.init_process_group to initialize the distributed environment
    # Always use NCCL as the backend. Gloo performance is pretty bad and MPI is currently
    # unsupported (for a number of reasons).
    dist.init_process_group(backend='nccl', rank=global_rank, world_size=world_size)

    # Set the cuda devices to avoid duplicate device in nccl 
    torch.cuda.set_device(global_rank%2)

    # now run your distributed training code
    run(global_rank, world_size, hostname)
    
    # following is just for demo purposes (and a sanity check)
    # you don't need this:
    if global_rank > 0 :
        print(f'printing NCCL variables on {global_rank} on {hostname}')
        for key in os.environ:
            if key.find('NCCL')>-1:
                print(f'{hostname} {key} : {os.getenv(key)}')





