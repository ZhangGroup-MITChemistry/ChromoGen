# Training data locations
config_fp = '../../data/processed_data.hdf5'
#embedding_dir = '../../data/embeddings_65/'
cooler_fp = '../../data/outside/GM12878_hg19.mcool'
resolution = 20_000

# Supporting data locations 
mean_dist_fp = '../../data/mean_dists.pt'
mean_sq_dist_fp='../../data/squares.pt'

# Destination directory
save_folder = '../../data/models/diffusion_origami_64_experimental_hic'

# Exclude chromosome X from training data so that it can be
# used for network validation 
training_chroms = [*[f'{k}' for k in range(1,23) if k!=7],'X']

# Training iteration details 
segment_length = 128#64
batch_size = 128#16#64
shuffle_data = True


import torch
import os
torch.set_num_threads(int(os.environ['SLURM_CPUS_PER_TASK']))
import sys
sys.path.insert(2,'/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/code/data_utils/')
from DataLoader_HiC1 import DataLoader
from ConfigDataset import ConfigDataset
from HiCDataset import HiCDataset

print('Preparing Data',flush=True)
print('Loading Configuration Dataset',flush=True)
config_ds = ConfigDataset(
    config_fp,
    segment_length=segment_length,
    remove_diagonal=False,
    batch_size=0,
    normalize_distances=True,
    geos=None,
    organisms=None,
    cell_types=None,
    cell_numbers=None,
    chroms=training_chroms,
    replicates=None,
    shuffle=False, #True, Shuffling should occur in dl object
    allow_overlap=True,
    two_channels=False,
    try_GPU=True,
    mean_dist_fp=mean_dist_fp,
    mean_sq_dist_fp=mean_sq_dist_fp
)

print('Preparing HiCDataset object',flush=True)
exp_hic = HiCDataset(
    cooler_fp=cooler_fp,
    resolution=resolution
)

print('Constructing DataLoader',flush=True)
dl = DataLoader(
    config_ds,
    exp_hic,
    #drop_unmatched_pairs=True,
    shuffle = shuffle_data,
    batch_size=batch_size,
    interp_hic_nans=False,
    processed_hic_folder='../../data/processed_hic_128/'
)
print('Data Preparation Complete',flush=True)



