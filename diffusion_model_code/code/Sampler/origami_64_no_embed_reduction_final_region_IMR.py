#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=origami_64_no_embed_reduction_full_scan_IMR
##SBATCH --partition=debug-gpu
#SBATCH --ntasks=1 # Per NODE
#SBATCH --gres=gpu:volta:1  # Per NODE
#SBATCH --cpus-per-task=20  # Per TASK
#SBATCH --array=0-7
#SBATCH --output=./log_files/origami_64_no_embed_reduction_full_scan_IMR_%a.log

import torch
import os
import pandas as pd 

import sys
sys.path.insert(0,'./')
from Sampler import Sampler
sys.path.insert(1,'../data_utils/SampleClass/')
from Distances import Distances

# Choose directories to save data to 
save_dir_2D = '../../data/samples/origami_64_no_embed_reduction/full_scan_IMR/'
save_dir_3D = '../../data/samples/origami_64_no_embed_reduction/full_scan_IMR/corrected/'

for d in [save_dir_2D,save_dir_3D]:
    if not os.path.exists(d):
        os.makedirs(d)

# Embedding directory
embedding_dir = '../../data/embeddings_64_IMR/'

# Number of samples to produce & batch sizes to use while doing so
num_samples = 1_000
batch_size = 1000

# Guidance weights and phi values to scan over
cond_scales = [5.]
rescaled_phis = [8.]

# Sampler instance, to perform the sampling procedure with the diffusion model
sampler = Sampler(
    model_dir='../../data/models/diffusion_origami_64_no_embed_reduction/',
    save_file=None,
    samples_device=torch.device('cpu'),
    # U-Net options
    dim=64,
    cond_drop_prob = 0.,
    init_dim = None,
    out_dim = None,
    dim_mults=(1, 2, 4, 8),
    channels = 2,
    resnet_block_groups = 8,
    learned_variance = False,
    learned_sinusoidal_cond = False,
    random_fourier_features = False,
    learned_sinusoidal_dim = 16,
    attn_dim_head = 32,
    attn_heads = 4,
    embedding_dimensions=(1,256,256),
    # Diffusion model options
    image_size=32,
    timesteps = 1000,
    sampling_timesteps = None,
    objective = 'pred_noise',
    beta_schedule = 'cosine',
    ddim_sampling_eta = 1.,
    offset_noise_strength = 0.,
    min_snr_loss_weight = False,
    min_snr_gamma = 5,
    cond_scale=None,
    rescaled_phi=None
)

# This function will convert the 2D distance maps to coordinates, returning them as torch.Tensors
def dist_to_coord(distances,batch_size=batch_size):
    i = 0
    coords = []
    while i < len(distances):
        j = min(i+batch_size,len(distances))
        temp_dists = Distances(distances[i:j,...])
        coords.append(temp_dists.unfold().unnormalize().coordinates.values)
        i = j
    return torch.cat(coords,dim=0)

# Find all regions with no more than 250 kb overlap to scan over the whole genome
rosetta = pd.read_pickle(embedding_dir + 'rosetta_stone.pkl')
regions = {}
for chrom in rosetta:
    indices = []
    stop = -1
    for i,(_,_,start) in enumerate(rosetta[chrom]):
        if start >= stop:
            indices.append(i)
            stop = start + 1_030_000 

    regions[chrom] = indices

i = 0
array = int(os.environ['SLURM_ARRAY_TASK_ID'])
embedding_file = lambda chrom: embedding_dir + f'chrom_{chrom}.tar.gz'
save_file_2D = lambda region, cond_scale, rescaled_phi, chrom: save_dir_2D + f'sample_{region}_{float(cond_scale)}_{float(rescaled_phi)}_120_{chrom}.pt'
save_file_3D = lambda region, cond_scale, rescaled_phi, chrom: save_dir_3D + f'sample_{region}_{float(cond_scale)}_{float(rescaled_phi)}_120_{chrom}.pt'
for chrom in regions:
    embeddings = pd.read_pickle(embedding_file(chrom))
    for region in regions[chrom]:
        embedding = embeddings.iloc[region].values[0]
        for cond_scale in cond_scales:
            for rescaled_phi in rescaled_phis:
                
                i+=1
                if i%8 != array:
                    continue
        
                # Save file, for generated sample
                sf = save_file_2D(region, cond_scale, rescaled_phi, chrom)

                # Remove existing data from the sampler
                sampler.purge()
                
                # Set the save file in the Sampler, loading any pre-generated conformations
                sampler.save_location = sf

                # Sample and save
                sampler.sample(
                    embedding,
                    cond_scale=cond_scale,
                    rescaled_phi=rescaled_phi,
                    num_samples=num_samples,
                    batch_size=batch_size,
                    save_results=True,
                    save_every_step=True,
                    coordinates=False
                )

                # Get/save the coordinates
                coords = dist_to_coord(sampler.samples)
                torch.save(coords.float().cpu(),save_file_3D(region,cond_scale,rescaled_phi,chrom)) 

                

