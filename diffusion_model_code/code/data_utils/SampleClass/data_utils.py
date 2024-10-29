import torch

def remove_diagonal(x):
    s = x.shape[-1]
    i,j = torch.triu_indices(s,s,1)
    x2 = torch.empty(*x.shape[:-2],s-1,s-1,dtype=x.dtype,device=x.device)
    #x2[...,i,j-1] = x[...,i,j]
    x2[...,j-1,i] = x[...,j,i] #overwrites on the diagonal
    x2[...,i,j-1] = x[...,i,j]
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

def transpose_minor_axis(tensor):
    return tensor.flip(-1).transpose(-2,-1).flip(-1)

def ceil(val):
    f_val = float(val) # in case a string representing an int is passed or something like that
    i_val = int(val)
    return i_val if i_val==f_val else i_val+1

def is_odd(val):
    val = float(val)
    i_val = int(val)
    if val != i_val:
        # non-integers aren't counted as odd or even
        return False
    return (val%2) == 1
    
    
    
