import sys
sys.path.insert(0,'./')

import torch
from Unet_resnet_like import Embed

dim = 64

for embedding_dimensions in [(1,260,256),(1,512,256)]:
    
    embedder = Embed(
        dim,
        embedding_dimensions
    )

    sample = torch.rand(2,*embedding_dimensions)

    out = embedder(sample)
    print(f'embedding_dimensions: {embedding_dimensions}')
    print(f'out.shape: {out.shape}')
    print('')