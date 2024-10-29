import torch


def remove_diagonal(x):
    s = x.shape[-1]
    i,j = torch.triu_indices(s,s,1)
    x2 = torch.empty(*x.shape[:-2],s-1,s-1,dtype=x.dtype,device=x.device)
    x2[...,i,j-1] = x[...,i,j]
    x2[...,j-1,i] = x[...,j,i]
    return x2

def add_diagonal(x,diag_value=0):
    s = x.shape[-1]
    i,j = torch.triu_indices(s,s,0)
    x2 = torch.empty(*x.shape[:-2],s+1,s+1,dtype=x.dtype,device=x.device)
    x2[...,i,j+1] = x[...,i,j]
    x2[...,j+1,i] = x[...,j,i] 
    i = torch.arange(s+1)
    x2[...,i,i] = diag_value
    return x2
    
def get_tan_sample(
    region_idx,
    chrom,
    embeddings#={chrom:embeddings_dict[chrom] for chrom in chroms}
):
    coords = get_tan_coords(region_idx,embeddings[chrom])
    dists = torch.cdist(coords,coords)
    return Sample(data=remove_diagonal(dists))

