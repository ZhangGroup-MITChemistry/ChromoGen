import os
import shutil

for d in os.listdir('./'):
    if not os.path.isdir(d):
        continue

    d1 = f'./{d}/'
    for f in os.listdir(d1):
        if 'chr_' not in f:
            continue
        
        _,chrom,region_idx,cond_scale,rescaled_phi = '.'.join( f.split('.')[:-1] ).split('_')

        f_new = f'sample_{region_idx}_{cond_scale}_{rescaled_phi}_120_{chrom}.pt'
        shutil.move(d1+f,d1+f_new)

    
