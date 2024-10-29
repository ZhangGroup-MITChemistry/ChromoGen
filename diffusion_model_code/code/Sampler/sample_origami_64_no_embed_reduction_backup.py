#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=sample_origami_64_no_embed_reduction
##SBATCH --partition=debug-gpu
#SBATCH --ntasks=1 # Per NODE
#SBATCH --gres=gpu:volta:1  # Per NODE
#SBATCH --cpus-per-task=20  # Per TASK
#SBATCH --array=0-1
#SBATCH --output=./log_files/sample_origami_64_no_embed_reduction_%a.log

import torch
import pandas as pd 
import pickle
import os

import sys
#sys.path.insert(0,'./models')
sys.path.insert(0,'../diffusion_compartmental/model_files')
from Unet_no_embed_reduction import Unet
#from Embedders import Embed, EmbedWithReinsertion
from GaussianDiffusion import GaussianDiffusion
#from load_params import load_params

########################
# Selectable parameters 
nbeads = 64
use_embedding_reinsertion = False  # The newer embedding handler 
origami_fold = True
unet_dim = 64
milestone = 36#240
model_filepath = f'../../data/models/diffusion_origami_64_no_embed_reduction/model-{milestone}.pt'
save_dir = '../../data/samples/origami_64_no_embed_reduction/'
embedding_dir = '../../data/embeddings_64_after_transformer'
#chrom = '1'
#region_idx = 330
regions = { # chrom:region_idxs
  '1':[144,200]#,265,330,395,460,525,590,730,795,860,1260,1325],
  #'X':[100,236,381,445,553,610,675,810,900,965,1060,1125,1200]
}
cond_scales = [.5,.9,1.5,2.,3.,4.,5.,6.,7.]#[.4]   #[.3,.35,.4,.45,.5]
rescaled_phis = [0.,.25,.5,.75,1.]#[.4] #[.3,.35,.4,.45,.5]
nsamples = 1000  # Per region

########################
# Build objects 
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Embedder
embedding_dimensions = (1,256,256)#(1,260,256) if nbeads in [64,65] else (1,512,256)
#embed_class = EmbedWithReinsertion if use_embedding_reinsertion else Embed
#embedder = embed_class(unet_dim,embedding_dimensions)

# Unet 
c,image_size = 2,nbeads//2
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
    embedding_dimensions=embedding_dimensions#tuple(er.ifetch(0)[0].shape)
)

# Diffusion model
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

diffusion = diffusion.to('cuda' if torch.cuda.is_available() else None)
diffusion.load_state_dict(torch.load(model_filepath,map_location=diffusion.device)['model'])
#load_params(diffusion,model_filepath)

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
        return max(nsamples - sample.shape[0],0)
    else:
      return nsamples

i = 0
array = int(os.environ['SLURM_ARRAY_TASK_ID'])
for chrom in regions: 
    embeddings = pd.read_pickle(f'{embedding_dir}/chrom_{chrom}.tar.gz')
    for cond_scale in cond_scales:
        for rescaled_phi in rescaled_phis: 
            for region_idx in regions[chrom]:
                i+=1
                if i%2 != array:
                    continue

                f = fp(region_idx,cond_scale,rescaled_phi,milestone,chrom)
                n_samples_to_do = n_samples_remaining(f,nsamples)
                if n_samples_to_do <= 0:
                    continue
                
                embedding = embeddings.iloc[region_idx,0].to(diffusion.device)
                
                sample = diffusion.sample(
                    embedding.repeat(n_samples_to_do,1,1,1),
                    #batch_size=n_samples_remaining(f,nsamples),
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
