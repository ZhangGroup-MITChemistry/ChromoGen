#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=origami_64_no_embed_reduction_test_progression
#SBATCH --partition=debug-gpu
#SBATCH --ntasks=1 # Per NODE
#SBATCH --gres=gpu:volta:1  # Per NODE
#SBATCH --cpus-per-task=20  # Per TASK
#SBATCH --array=0 #-1
#SBATCH --output=./log_files/origami_64_no_embed_reduction_test_progression_%a.log

import torch
import os

import sys
sys.path.insert(0,'./')
from Sampler_test_progression import Sampler

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

num_samples = 1 #50_000
batch_size = 1 #1000

cond_scales = [0.]
rescaled_phis = [0.]

array = int(os.environ['SLURM_ARRAY_TASK_ID'])
save_file = '../../data/samples/origami_64_no_embed_reduction/eval_mode/test_progression.pt'

i = 0
embedding = torch.zeros(1,1,256,256)
for cond_scale in cond_scales:
    for rescaled_phi in rescaled_phis:
        
        #i+=1
        #if i%2 != array:
        #    continue

        # Save file, for sample
        sf = save_file

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

