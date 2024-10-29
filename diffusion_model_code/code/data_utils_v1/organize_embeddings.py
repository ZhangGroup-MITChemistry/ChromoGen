#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=organize_embeddings
#SBATCH -c 4
#SBATCH --array=1-23
#SBATCH --output=./log_files/organize_embeddings_%a.log

import pandas as pd
import torch 
import os 
import sys

# THIS IS TEMPORARY
import pickle

# Where are files located? 
#embeddings_main_dir = '../../data/raw_embeddings/'
#embeddings_main_dir = '/home/gridsan/gschuette/binz_group_shared/zlao/for_greg/hg19/'
#embeddings_main_dir = '/home/gridsan/gschuette/binz_group_shared/zlao/for_greg/hg19_with_new_hic_optim/node_embedding/'
#embeddings_main_dir = '/home/gridsan/gschuette/binz_group_shared/zlao/for_greg/hg19_128beads/node_embedding/'
#embeddings_main_dir = '/home/gridsan/gschuette/binz_group_shared/zlao/for_greg/after_transformation/size128/node_embedding/'
embeddings_main_dir = '/home/gridsan/gschuette/binz_group_shared/zlao/for_greg/IMR_size64/node_embedding/'

# Load Zhuohan's dictionary that relates file indices to genomic regions
info_dict = pd.read_pickle(embeddings_main_dir + 'my_dict.pickle')

# The following function parses the filenames to provide the index within 
# Zhuohan's dictionary, allowing us to extract relevant information. 
parse_filename = lambda f: int( f.split('_')[-1].split('.')[0] )

# We will store this data in the CPU RAM except when being actively used, so we'll map all the files to
# the CPU upon loading to save it there later. 
device = torch.device('cpu')

# Load all of the data in the usual ordering of chromosomes.
# Keep track of region length, chromosome number, & starting genomic position
# to aid with indexing in the dataloader. 
#chroms = [f'{k}' for k in range(1,23)]
#chroms.append('X')
chroms = os.environ['SLURM_ARRAY_TASK_ID']
chroms = [chroms if chroms != '23' else 'X']
data = []
region_size = [] 
chrom_idx = []   
start_idx = []
# TEMP
nn = 0 
for chrom in chroms: #['1']: 

    # Define some objects used repeatedly in the following loop.
    subdir = embeddings_main_dir + f'run_scripts_{chrom}/'
    filepath = lambda idx: subdir + f'chr_{chrom}_{idx}.pt'

    # Indices are easy to obtain 
    idxs = [parse_filename(fn) for fn in os.listdir(subdir)]
    idxs.sort()

    # They can be used in one liners to improve speed a bit.
    
    # No need to append to chrom_idx repeatedly in the inner loop
    chrom_idx.extend([chrom for _ in range(len(idxs))])

    # Remove the gradient information upon loading data since it serves no purpose downstream. 
    data.extend([ 
        torch.load(filepath(idx),map_location=device).requires_grad_(False) for idx in idxs
    ])
    
    region_size.extend([
        info_dict[chrom][idx,1]-info_dict[chrom][idx,0] for idx in idxs
    ])

    start_idx.extend([
        info_dict[chrom][idx,0] for idx in idxs
    ])

    # TEMP
    pickle.dump({'data':data[nn:],'idxs':idxs,'region_size':region_size[nn:],'start_idx':start_idx[nn:]},open(f'./Zhuohan_Data_IMR/data_files_{chrom}.pkl','wb'))
    nn = len(region_size) 
    
    print(f'Chromosome {chrom} completed!',flush=True)

sys.exit()

# Place values into a dataframe so that we can easily sort them. 
embed_df = pd.DataFrame()
embed_df['Region_Size'] = region_size
embed_df['Chromosome'] = chrom_idx
embed_df['Genomic_Index'] = start_idx
embed_df['Data'] = data
del region_size, chrom_idx, start_idx, data

# Organize the data by sorting
idx_cols = ['Region_Size','Chromosome','Genomic_Index']
embed_df.sort_values(by=idx_cols,axis='index',ignore_index=True,inplace=True)

# Create a MultiIndex for easier fetching of data inside of the DataLoader later on. 
embed_df.index = pd.MultiIndex.from_tuples(
    list(map(tuple,embed_df[idx_cols].values)),
    names=idx_cols
)

# Drop the now-redundant columns
embed_df.drop(columns=idx_cols,inplace=True)

# Save the processed data to the canonical HDF file
embed_df.to_hdf(
    data_filepath,
    key='Embeddings',
    complevel=9,
    
)
