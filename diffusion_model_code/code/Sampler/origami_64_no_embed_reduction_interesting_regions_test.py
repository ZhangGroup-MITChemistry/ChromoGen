#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=origami_64_no_embed_reduction_interesting_regions_test
#SBATCH --partition=debug-gpu
#SBATCH --ntasks=1 # Per NODE
#SBATCH --gres=gpu:volta:1  # Per NODE
#SBATCH --cpus-per-task=20  # Per TASK
#SBATCH --array=0 #-5
#SBATCH --output=./log_files/origami_64_no_embed_reduction_interesting_regions_test_%a.log

import torch
import os
import pandas as pd 

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

num_samples = 1_000
batch_size = 1000

#cond_scales = [1.5,2.,2.5,3.,3.5,4.,4.5,5.,5.5,6.,6.5,7.,7.5,7.]
#rescaled_phis = [.1,.3,.5,.7,.9]
cond_scales = [1.]#[5.]
rescaled_phis = [0.] #[8.]

#regions = { # chrom:region_idxs
#  '1':[144,200,265,330,395,460,525,590,730,795,860,1260,1325], # 330,395
#  'X':[100,236,381,445,553,610,675,810,900,965,1060,1125,1200]
#}

# Additional regions to scan genome more broadly
regions = {
    '1':[395],
    #'3':[5,100],
    #'4':[77,218],
    #'11':[54],
    #'13':[166,302],
    #'14':[174,335,519],
    #'15':[930,1322],
    #'16':[12,549],
    #'17':[22,707],
    #'18':[13],
    #'20':[15,186],
    #'22':[464,675],
    #'X':[4756] # for high K27 content
}

i = 0
array = int(os.environ['SLURM_ARRAY_TASK_ID'])
embedding_file = lambda chrom: f'../../data/embeddings_64_after_transformer/chrom_{chrom}.tar.gz'
save_dir = '../../data/samples/origami_64_no_embed_reduction/eval_mode/unused/'
save_file = lambda region, cond_scale, rescaled_phi, chrom: save_dir + f'sample_{region}_{float(cond_scale)}_{float(rescaled_phi)}_120_{chrom}.pt'
for chrom in regions:
    embeddings = pd.read_pickle(embedding_file(chrom))
    for region in regions[chrom]:
        embedding = embeddings.iloc[region].values[0]
        for cond_scale in cond_scales:
            for rescaled_phi in rescaled_phis:
                
                #i+=1
                #if i%6 != array:
                #    continue
        
                # Save file, for sample
                sf = save_file(region, cond_scale, rescaled_phi, chrom)

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
                )

