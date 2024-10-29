import sys
sys.path.insert(0,'./')

import torch
from Unet import Unet
from Embedders import Embed, EmbedWithReinsertion

dim = 64
t = 500 # timestep
nsamples = 2 # batch size
for embedding_dimensions,imsize in [((1,260,256),(2,32,32))]:#,(1,512,256)]:
    print(f'embedding_dimensions: {embedding_dimensions}')

    unet = Unet(
        dim=64,
        cond_drop_prob=.5,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = imsize[0],
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        attn_dim_head = 32,
        attn_heads = 4,
        embedding_dimensions=embedding_dimensions
    )

    embedding = torch.rand(nsamples,*embedding_dimensions)
    batch = torch.rand(nsamples,*imsize)
    batched_times = torch.full((batch.shape[0],), t, dtype = torch.long)
    
    for embedder in [
        Embed(
            dim,
            embedding_dimensions
        ),
        EmbedWithReinsertion(
            dim,
            embedding_dimensions
        )
    ]:

        print(f'Embedder Type: {type(embedder)}')
    
        for device in [torch.device('cpu'),torch.device('cuda')]:
    
            print(f'Device: {device}')
            embedder = embedder.to(device)
            embedding = embedding.to(device)
            unet = unet.to(device) 
            batch = batch.to(device) 
            batched_times = batched_times.to(device) 
            
            out = embedder(embedding)
            print(f'Embedder out.shape: {out.shape}\tout.device: {out.device}')
            
            out = unet.forward_with_cond_scale(
                batch,
                out, # this is the embedding! 
                batched_times,
                cond_scale=.5,
                rescaled_phi=.5
            )
            print(f'Unet out.shape: {out.shape}\tout.device: {out.device}')
            
        print('')





