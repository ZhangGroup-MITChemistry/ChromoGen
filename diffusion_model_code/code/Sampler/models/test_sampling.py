import torch
import pandas as pd 
import pickle

import sys
sys.path.insert(0,'./')
from Unet import Unet
from Embedders import Embed, EmbedWithReinsertion
from GaussianDiffusion import GaussianDiffusion
from load_params import load_params

########################
# Selectable parameters 
nbeads = 64
use_embedding_reinsertion = False
embedding_dimensions = (1,260,256)
origami_fold = True
unet_dim = 64
model_filepath = '../../../data/models/diffusion_small_origami/model-120.pt'

embedding_dir = '../../../data/embeddings_65'
chrom = '1'
region_idx = 330
cond_scale = .4
rescaled_phi = .4
nsamples = 500#1000

########################
# Build objects 

# Embedder
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


########################
# Ensure everything runs as desired, ignoring parameter loads, etc.
'''
embeddings = torch.rand(1,*embedding_dimensions,device=diffusion.device)

sample = diffusion.sample(embeddings,batch_size=2)
print(f'sample.shape: {sample.shape}',flush=True) 
'''

########################
# Load parameters to ensure we obtain reasonable output with trained parameters
load_params(diffusion,model_filepath)

embeddings = pd.read_pickle(f'{embedding_dir}/chrom_{chrom}.tar.gz')
embedding = embeddings.iloc[region_idx,0]

sample = diffusion.sample(
    embedding,
    batch_size=nsamples,
    cond_scale=cond_scale,
    rescaled_phi=rescaled_phi
)
print(f'sample.shape: {sample.shape}',flush=True)
pickle.dump(
    sample.cpu(),
    open('./sample.pkl','wb')
)

