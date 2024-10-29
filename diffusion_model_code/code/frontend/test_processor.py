import os
import numpy as np
import torch
from scipy.sparse import load_npz
from get_inputs import get_hg_files

alignment = 'hg38' 

#######
# Download & process the data; get filepaths for my data 
fps = get_hg_files(alignment)

#######
# Ensure all of the data exactly matches the data from Zhuohan 

# Define chromosomes to check 
chroms = [f'chr{k}' for k in range(1,23)]
chroms.append('chrX')

# Get the filepaths corresponding to Zhuohan's data; remove those
# for which there is no comparison. 
zdir = '/home/gridsan/gschuette/binz_group_shared/zlao/data/hg38/'
fps_ = fps.copy()
fps_zhuohan = []
for fp in fps: 
    zfp = zdir + fp.split('/')[-1] 
    if not os.path.exists(fp):
        fps_.remove(fp) 
    elif os.path.exists(zfp):
        fps_zhuohan.append(zfp)
    else:
        fps_.remove(fp) 
fps = fps_ 

# Load both versions of both files and ensure they are equal. 
def pad_seq_matrix(matrix, pad_len=300):
    paddings = np.zeros((1, 4, pad_len)).astype('int8')
    dmatrix = np.concatenate((paddings, matrix[:, :, -pad_len:]), axis=0)[:-1, :, :]
    umatrix = np.concatenate((matrix[:, :, :pad_len], paddings), axis=0)[1:, :, :]
    return np.concatenate((dmatrix, matrix, umatrix), axis=2)
    
def load_ref_genome(fp):
    ref_gen_data = load_npz(fp).toarray().reshape(4, -1, 1000).swapaxes(0, 1)
    return torch.tensor(pad_seq_matrix(ref_gen_data))

for i,fp in enumerate(fps):

    # Zhuohan filepath 
    zfp = fps_zhuohan[i]
    
    # Ensure we are comparing the correct files
    chrom_file = fp.split('/')[-1]
    assert zfp.split('/')[-1] == chrom_file

    # Load the relevant data 
    my_data = load_ref_genome(fp)
    zh_data = load_ref_genome(zfp)

    # Ensure the data is equal everywhere
    if (my_data != zh_data).any():
        print(f'Failure at {chrom_file}',flush=True)
    else: 
        print(f'Success at {chrom_file}',flush=True)

    del my_data, zh_data
