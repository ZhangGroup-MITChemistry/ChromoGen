#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=processing_HiC
#SBATCH --array=7,19,21 #1-23
#SBATCH --cpus-per-task=4
#SBATCH --output=./log_files/processing_HiC_%a.log

import os
import sys
sys.path.insert(0,'./')
from DataLoader_HiC import DataLoader
from ConfigDataset import ConfigDataset
from HiCDataset import HiCDataset
from tqdm.auto import tqdm 

# Training data locations
config_fp = '../../data/processed_data.hdf5'
#embedding_dir = '../../data/embeddings_65/'
cooler_fp = '../../data/outside/GM12878_hg19.mcool'
resolution = 20_000

# Supporting data locations 
mean_dist_fp = '../../data/mean_dists.pt'
mean_sq_dist_fp='../../data/squares.pt'

#training_chroms = [f'{k}' for k in range(1,23)]
chrom = os.environ['SLURM_ARRAY_TASK_ID']
training_chroms = [chrom if chrom != '23' else 'X']


# Training iteration details
segment_length = 64
batch_size = 128#16#64
shuffle_data = True

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

exp_hic = HiCDataset(
    cooler_fp=cooler_fp,
    resolution=resolution
)

print('Constructing DataLoader',flush=True)
dl = DataLoader(
    config_ds,
    exp_hic,
    #drop_unmatched_pairs=True,
    shuffle = False,#shuffle_data,
    batch_size=batch_size,
    interp_hic_nans=False
)

N = len(dl) // batch_size + 1
_ = next(dl)
#while dl.internal_idx != 0:
for _ in tqdm(range(N),desc = 'Progress: ', total = N):
    try:
        _ = next(dl) 
    except: # occasionally receive out-of-bounds errors
        continue 
import pickle
pickle.dump(dl.hic_maps,open(f'hic_data_chrom_{training_chroms[0]}.pkl','wb'))





