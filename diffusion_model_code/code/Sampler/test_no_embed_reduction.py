import torch
import sys 
sys.path.insert(0,'./models/')
from GaussianDiffusion import GaussianDiffusion
from Unet_no_embed_reduction import Unet
from Embedders import Flatten
from tqdm.auto import tqdm 

#c,image_size = 2,segment_length//2
unet = Unet(
    dim=64,
    cond_drop_prob = 0.5,
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
    embedding_dimensions=(1,260,256)#tuple(er.ifetch(0)[0].shape)
)

#torch.save(model,'test_size.pt') # 2.2 GB

diffusion = GaussianDiffusion(
    unet,
    embedder=Flatten(),#nn.Identity(),#embedder,
    image_size=32,#image_size,
    timesteps = 1000,
    sampling_timesteps = None,
    objective = 'pred_noise',
    beta_schedule = 'cosine',
    ddim_sampling_eta = 1.,
    offset_noise_strength = 0.,
    min_snr_loss_weight = False,
    min_snr_gamma = 5
).to('cuda')

#sample = torch.rand(1000,2,32,32,device=model.device)
embedding = torch.rand(1,1,260,256,device=diffusion.device)

diffusion.sample(embedding,batch_size=1000)


