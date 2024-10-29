#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=origami_64_no_embed_reduction_cross_validation
##SBATCH --partition=debug-gpu
#SBATCH --ntasks=1 # Per NODE
#SBATCH --gres=gpu:volta:1  # Per NODE
#SBATCH --cpus-per-task=20  # Per TASK
#SBATCH --array=0-2
#SBATCH --output=./log_files/origami_64_no_embed_reduction_cross_validation_%a.log

import torch
import os

import sys
sys.path.insert(0,'./')
from Sampler import Sampler

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

num_samples = 1000
batch_size = 3000

cond_scales = [1.5,2.,2.5,3.,3.5,4.]
rescaled_phis = [.1,.3,.5,.7,.9]

i = 0
array = int(os.environ['SLURM_ARRAY_TASK_ID'])
folders = [
    (
        '/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/data/cross_validation_embeddings/CTCF_flips/',
        '/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/data/samples/origami_64_no_embed_reduction/cross_validation/CTCF/'
    ),
    (
        '/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/data/cross_validation_embeddings/IMR/',
        '/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/data/samples/origami_64_no_embed_reduction/cross_validation/IMR/'
    )
]
for ef,sf in folders:
    if not os.path.exists(sf):
        os.makedirs(sf)
    embedding_files = [f for f in os.listdir(ef) if f[-3:] == '.pt']
    for f in embedding_files:
        lf = ef+f # Load file, with embedding
        embedding = torch.load(lf).float().cuda().detach().unsqueeze(0)
        for cond_scale in cond_scales:
            for rescaled_phi in rescaled_phis:
                
                i+=1
                if i%3 != array:
                    continue
        
                # Save file, for sample
                sff = sf+f.replace('.pt','')
                sff+= f'_{cond_scale}_{rescaled_phi}.pt'

                # Remove existing data from the sampler
                sampler.purge()
                
                # Set the save file in the Sampler, loading any pre-generated conformations
                sampler.save_location = sff

                # Sample and save
                sampler.sample(
                    embedding,
                    cond_scale=cond_scale,
                    rescaled_phi=rescaled_phi,
                    num_samples=num_samples,
                    batch_size=batch_size,
                    save_results=True,
                    save_every_step=True,
                )

