#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=train_origami_64_no_embed_reduction #train_origami_64_resnet_like
##SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
##SBATCH -t 0-0:05
#SBATCH -t 0-65:00:00
#SBATCH --output=./log_files/train_origami_64_no_embed_reduction.log

###########################################################
# Set parameters 
###########################################################

# Training data locations
config_fp = '../../data/processed_data.hdf5'
embedding_dir = '../../data/embeddings_64_after_transformer/'

# Supporting data locations 
mean_dist_fp = '../../data/mean_dists.pt'
mean_sq_dist_fp='../../data/squares.pt'

# Destination directory
save_folder = '../../data/models/diffusion_origami_64_no_embed_reduction'

# Exclude chromosome X from training data so that it can be
# used for network validation 
training_chroms = [f'{k}' for k in range(1,23)]

# Training iteration details 
segment_length = 64
batch_size = 128#16#64
shuffle_data = True


###########################################################
# Import modules
###########################################################

# This should eventually be performed via relative imports
import os
import sys
#sys.path.insert(0,'./') #/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/code/diffusion/')
#from classifier_free_guidance_greg import GaussianDiffusion, Unet, Trainer
sys.path.insert(0,'./model_files/')
from GaussianDiffusion import GaussianDiffusion
from Trainer import Trainer
#sys.path.insert(1,'../Sampler/models/')
from Unet_no_embed_reduction import Unet
from Embedders import Flatten
sys.path.insert(2,'/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/code/data_utils/')
from DataLoader import DataLoader
from ConfigDataset import ConfigDataset
from EmbeddedRegions import EmbeddedRegions

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

##########################################
# models

c,image_size = 2,segment_length//2
unet = Unet(
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
)

#torch.save(model,'test_size.pt') # 2.2 GB

diffusion = GaussianDiffusion(
    unet,
    #embedder=Flatten(),#nn.Identity(),#embedder,
    image_size=image_size,
    timesteps = 1000,
    sampling_timesteps = None,
    objective = 'pred_noise',
    beta_schedule = 'cosine',
    ddim_sampling_eta = 1.,
    offset_noise_strength = 0.,
    min_snr_loss_weight = False,
    min_snr_gamma = 5
).to('cuda')

trainer = Trainer(
    diffusion,
    dl,
    train_batch_size = batch_size,
    gradient_accumulate_every = 1,
    augment_horizontal_flip = True,
    train_lr = 1e-4,
    train_num_steps = 1_200_000, #600_000,
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

trainer.train()

