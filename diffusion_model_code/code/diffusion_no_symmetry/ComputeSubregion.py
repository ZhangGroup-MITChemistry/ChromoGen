import torch
class ComputeSubregion: # This will select the relevant subregions!

    def __init__(self,nbeads,includes_self_interaction):

        self.return_size = nbeads // 2
        shift = int(includes_self_interaction)
        self.includes_self_interaction = includes_self_interaction
        self.start_size = nbeads - 1 + shift
        self.index_for_large_matrix = (
            torch.arange(self.return_size).repeat_interleave(self.return_size), # 0,0,0,...,1,1,1,...,N,N,N
            torch.cat([torch.arange(i,i+self.return_size) for i in range(shift,self.return_size+shift)]) # 1,2,3,...,N+1,2,3,..,N+2,...,N+small_size (if shift=1)
        )
    def __call__(self,full_maps):
        n,i,j = self.return_size, *self.index_for_large_matrix
        return full_maps[...,i,j].unflatten(-1,(n,n))

    def inverse(self,submap):
        N,i,j,optional_dims = self.start_size,*self.index_for_large_matrix,submap.shape[:-2]
        full_shape = (*optional_dims,N,N) 
        full_maps = torch.empty(
            full_shape,
            device=submap.device,
            dtype=submap.dtype
        ).fill_(torch.nan)
        
        full_maps[...,i,j] = full_maps[...,j,i] = submap.flatten(-2) 
        if self.includes_self_interaction: 
            i = torch.arange(N) 
            full_maps[...,i,i] = 0 
        return full_maps