#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=origami_64_no_embed_reduction_different_milestones
##SBATCH --partition=debug-gpu
#SBATCH --ntasks=1 # Per NODE
#SBATCH --gres=gpu:volta:1  # Per NODE
#SBATCH --cpus-per-task=20  # Per TASK
#SBATCH --array=0-7
#SBATCH --output=./log_files/origami_64_no_embed_reduction_different_milestones_%a.log

import torch
import os
import pandas as pd

import sys
sys.path.insert(0,'./')
from Sampler import Sampler
sys.path.insert(1,'../data_utils/SampleClass/')
from Distances import Distances, Normalizer

# Choose folders to save data to 
save_folder_2D = '../../data/samples/origami_64_no_embed_reduction/multiple_milestones/'
save_folder_3D = '../../data/samples/origami_64_no_embed_reduction/multiple_milestones/corrected/'

for d in [save_folder_2D,save_folder_3D]:
    if not os.path.exists(d):
        os.makedirs(d)


# Choose models to load
milestones = list(range(5,120,5))

# Choose genomic regions to generate data for
regions = {
    '1':[144,200,265,330,395,460,525,590,730,795,860,1260,1325]
}

# Decide number of samples to generate
num_samples_unguided = 100_000
num_samples_guided = 10_000
batch_size = 1000

# Guidance strength/rescaling settings, and necessary filepaths
embedding_dir = '../../data/embeddings_64_after_transformer/'
cond_scale = 5.0
rescaled_phi = 8.0 

# Set the unguided embedding. 
# The choice here is irrelevant since the saved random embedding used for unguided options is saved in 
# the model, and that is used by default whenever cond_scale==0.0
unguided_embedding = torch.zeros(1,1,256,256) 

# This function will convert the 2D distance maps to coordinates, returning them as torch.Tensors
mean_fp = '../../data/mean_dists.pt'
mean_sq_fp = '../../data/squares.pt'
normalizer = Normalizer(mean_fp,mean_sq_fp)
def dist_to_coord(distances,normalizer=normalizer,batch_size=batch_size):
    i = 0
    coords = []
    while i < len(distances):
        j = min(i+batch_size,len(distances))
        temp_dists = Distances(distances[i:j,...])
        coords.append(temp_dists.unfold().unnormalize(normalizer).coordinates.values)
        i = j
    return torch.cat(coords,dim=0)

# For distributing work across multiple GPUs
array = int(os.environ['SLURM_ARRAY_TASK_ID'])
i = 0 
for milestone in milestones:
    i+= 1
    if i%8 != array:
        continue

    # Load the model with the corresponding milestone
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
        rescaled_phi=None,
        # Specify which model to load
        milestone=milestone
    )

    # Set the save file in the Sampler, loading any pre-generated conformations
    sampler.save_location = save_folder_2D + f'unguided_{milestone}.pt'

    # Sample and save
    sampler.sample(
        unguided_embedding,
        cond_scale=0.,
        rescaled_phi=0.,
        num_samples=num_samples_unguided,
        batch_size=batch_size,
        save_results=True,
        save_every_step=True,
        coordinates=False
    )

    # Get/save the coordinates
    coords = dist_to_coord(sampler.samples)
    torch.save(coords.float().cpu(),save_folder_3D + f'unguided_{milestone}.pt')

    for chrom in regions:
        # Load the embeddings for this chromosome
        embeddings = pd.read_pickle(embedding_dir+f'chrom_{chrom}.tar.gz')
        for region_idx in regions[chrom]:

            # Remove existing data from the sampler
            sampler.purge()
            
            # Set the save file in the Sampler, loading any pre-generated conformations
            sampler.save_location = save_folder_2D + f'sample_{region_idx}_{cond_scale}_{rescaled_phi}_{milestone}_{chrom}.pt'

            # Fetch the relevant embedding
            embedding = embeddings.iloc[region_idx].values[0]
            
            # Sample and save
            sampler.sample(
                unguided_embedding,
                cond_scale=cond_scale,
                rescaled_phi=rescaled_phi,
                num_samples=num_samples_guided,
                batch_size=batch_size,
                save_results=True,
                save_every_step=True,
                coordinates=False
            )
        
            # Get/save the coordinates
            coords = dist_to_coord(sampler.samples)
            torch.save(coords.float().cpu(),save_folder_3D + f'sample_{region_idx}_{cond_scale}_{rescaled_phi}_{milestone}_{chrom}.pt')

