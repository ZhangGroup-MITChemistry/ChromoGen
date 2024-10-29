#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=sample_origami_64
##SBATCH --partition=debug-gpu
#SBATCH --ntasks=1 # Per NODE
#SBATCH --gres=gpu:volta:1  # Per NODE
#SBATCH --cpus-per-task=20  # Per TASK
#SBATCH --output=./log_files/sample_origami_64.log

import torch
import pandas as pd 
import pickle
import os

import sys
#sys.path.insert(0,'./models')
sys.path.insert(0,'../diffusion_compartmental/model_files')
from Unet import Unet
from Embedders import Embed, EmbedWithReinsertion
from GaussianDiffusion import GaussianDiffusion
from load_params import load_params

########################
# Selectable parameters 
nbeads = 64
use_embedding_reinsertion = False  # The newer embedding handler 
origami_fold = True
unet_dim = 64
milestone = 240
model_filepath = f'../../data/models/diffusion_small_origami/model-{milestone}.pt'
save_dir = '../../data/samples/origami_final_embeddings/'
embedding_dir = '../../data/embeddings_65'
#chrom = '1'
#region_idx = 330
regions = { # chrom:region_idxs
  '1':[144,200,265,330,395,460,525,590,730,795,860,1260,1325],
  'X':[100,236,381,445,553,610,675,810,900,965,1060,1125,1200]
}
cond_scales = [.9]#[.4]   #[.3,.35,.4,.45,.5]
rescaled_phis = [.9]#[.4] #[.3,.35,.4,.45,.5]
nsamples = 1000  # Per region

########################
# Build objects 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Embedder
embedding_dimensions = (1,260,256) if nbeads in [64,65] else (1,512,256)
embed_class = EmbedWithReinsertion if use_embedding_reinsertion else Embed
embedder = embed_class(unet_dim,embedding_dimensions)

# Unet 
unet = Unet(
    dim = unet_dim,
    cond_drop_prob = 0.5,
    init_dim = None,
    out_dim = None,
    dim_mults=(1, 2, 4, 8),
    channels = 1 + int(origami_fold),
    resnet_block_groups = 8,
    learned_variance = False,
    learned_sinusoidal_cond = False,
    random_fourier_features = False,
    learned_sinusoidal_dim = 16,
    attn_dim_head = 32,
    attn_heads = 4,
    embedding_dimensions=embedding_dimensions
)

# Diffusion model
if origami_fold: 
    assert nbeads % 2 == 0
    image_size = nbeads // 2
else:
    image_size = nbeads + 1
diffusion = GaussianDiffusion(
    unet,
    embedder=embedder,
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
diffusion = diffusion.to('cuda' if torch.cuda.is_available() else None)
load_params(diffusion,model_filepath)

########################
# Generate samples
def fp(region_idx,cond_scale,rescaled_phi,milestone,chrom,save_dir=save_dir):
    return save_dir + f'sample_{region_idx}_{cond_scale}_{rescaled_phi}_{milestone}_{chrom}.pt'

def save_sample(sample,f):
    if os.path.exists(f):
        sample = torch.cat(
            (
                torch.load(f,map_location=sample.device),
                sample
            ),
            dim = 0 
        )
    torch.save(sample.cpu(),f)

def n_samples_remaining(f,nsamples):
    if os.path.exists(f):
        sample = torch.load(f,map_location='cpu')
        return nsamples - sample.shape[0]
    else:
      return nsamples

for chrom in regions: 
    embeddings = pd.read_pickle(f'{embedding_dir}/chrom_{chrom}.tar.gz')
    for cond_scale in cond_scales:
        for rescaled_phi in rescaled_phis: 
            for region_idx in regions[chrom]:

                f = fp(region_idx,cond_scale,rescaled_phi,milestone,chrom)
                embedding = embeddings.iloc[region_idx,0]
                
                sample = diffusion.sample(
                    embedding,
                    batch_size=n_samples_remaining(f,nsamples),
                    cond_scale=cond_scale,
                    rescaled_phi=rescaled_phi
                )
                save_sample(sample,f)
                #torch.save(sample.cpu(),f)
                #print(f'sample.shape: {sample.shape}',flush=True)
                #pickle.dump(
                #    sample.cpu(),
                #    open('./sample.pkl','wb')
                #)
