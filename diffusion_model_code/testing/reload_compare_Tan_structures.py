#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=text_structure_shift
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --array=0-15
#SBATCH --output=./log_files/text_structure_shift_%a.log

import pandas as pd
import torch
import os
from tqdm.auto import tqdm 
import sys
sys.path.insert(0,'../code/data_utils/')
from ConfigDataset import ConfigDataset

all_config_datasets = {}
for cell_number in range(1,18):
    if cell_number == 8:
        continue
    all_config_datasets[cell_number] = {}
    for replicate in range(3):
        all_config_datasets[cell_number][replicate] = {}
        for chrom in [*[str(k) for k in range(1,23)],'X']:
            try:
                all_config_datasets[cell_number][replicate][chrom] = ConfigDataset(
                    '../data/processed_data.hdf5',
                    segment_length=64,
                    remove_diagonal=False,
                    batch_size=0,
                    normalize_distances=False,
                    geos=None,
                    organisms=None,
                    cell_types=None,
                    cell_numbers=[cell_number],
                    chroms=[chrom],
                    replicates=[replicate],
                    shuffle=True,
                    allow_overlap=True,
                    two_channels=False,
                    try_GPU=False,
                    mean_dist_fp=None,
                    mean_sq_dist_fp=None
                )
            except:
                continue

raw_tan_data_dir = '../data/tan_single-cell_2018/'


def get_cell_number(fp):
    return int( fp.split('/')[-1].split('_')[-1].split('-')[0] )

# Find the directories that actually contain 
all_dirs = [ raw_tan_data_dir + d for d in os.listdir( raw_tan_data_dir ) ]
valid_dirs = []
for d in all_dirs:

    files = os.listdir(d)
    contains_clean = False
    for f in files: 
        if 'impute3.round4.clean.3dg' in f:
            contains_clean = True
            break

    if contains_clean:
        valid_dirs.append(
            (
                get_cell_number(d),
                d
            )
        )
valid_dirs.sort()

def get_raw_coords(replicate_number,directory):

    if directory[-1] != '/':
        directory+= '/'

    clean_structure_files = [f for f in os.listdir(directory) if 'impute3.round4.clean.3dg' in f]

    for f in clean_structure_files:

        
        if '_rep1_' in f:
            if replicate_number == 1:
                filepath = directory + f
                break
        elif '_rep2_' in f:
            if replicate_number == 2:
                filepath = directory + f
                break
        elif replicate_number == 0:
            filepath = directory + f

    coord_df = pd.read_csv(
        filepath,
        sep='\t',
        header=None,
        names=['Chromosome','Genomic_Index','x','y','z']
    )
    return coord_df

def extract_specific_region(sub_coord_df,genomic_index,region_length=64,resolution=20_000):

    # Get the relevant region
    sub_coord_df = sub_coord_df[ 
        (sub_coord_df.Genomic_Index >= genomic_index) & 
        ( sub_coord_df.Genomic_Index < genomic_index + resolution * region_length )  
    ]
    sub_coord_df = sub_coord_df.sort_values('Genomic_Index',axis=0,ignore_index=True)

    # Fetch both maternal & paternal data
    coords = []
    for chrom in sub_coord_df.Chromosome.unique():
        vals = torch.from_numpy(sub_coord_df[ sub_coord_df.Chromosome == chrom ][['x','y','z']].values)

        if 'pat' in chrom:
            coords.append(vals)
        else:
            coords.insert(0,vals)

    return coords

def compare_to_raw(cell_number, raw_directory, all_config_datasets=all_config_datasets):

    unmatched_regions = [[],[],[]]
    n_regions = []
    
    for replicate in range(0,3):

        coord_df = get_raw_coords(replicate, raw_directory)

        N = 0
        genomic_regions = {}
        for chrom in all_config_datasets[cell_number][replicate]:
            genomic_regions[chrom] = all_config_datasets[cell_number][replicate][chrom].get_genomic_regions()
            N+= len(genomic_regions[chrom])
        n_regions.append(N)
        
        with tqdm( total=N, initial=0 ) as pbar:
        
            for chrom in genomic_regions:#genomic_regions.Chromosome.unique():

                processed_data = all_config_datasets[cell_number][replicate][chrom]
    
                chrom_regions = genomic_regions[ chrom ]
    
                sub_coord_df = coord_df[ (coord_df.Chromosome == f'{chrom}(mat)') | (coord_df.Chromosome == f'{chrom}(pat)')  ]
                
                for _,row in chrom_regions.iterrows():
                    genomic_index = row.Start
                    coord_info, processed_coords = processed_data.fetch_specific_coords(chrom,genomic_index)
                    raw_coords = extract_specific_region(sub_coord_df,genomic_index)
                    
                    if raw_coords is None or \
                    not torch.allclose( processed_coords[coord_info.Lineage=='mat'].squeeze(), raw_coords[0] ) or \
                    not torch.allclose( processed_coords[coord_info.Lineage=='pat'].squeeze(), raw_coords[1] ):
                        unmatched_regions[replicate].append( (chrom,genomic_index) )

                    pbar.update(1)

    return unmatched_regions, n_regions

import time
all_mismatched_regions = {}

valid_dirs = [
    valid_dirs[int(os.environ['SLURM_ARRAY_TASK_ID'])]
]

for cell_number, directory in valid_dirs:
    t = -time.time()
    unmatched_regions,n_regions = compare_to_raw(cell_number,directory)
    print(f'Cell {cell_number}: {time.time()+t:.4f}',flush=True)

    #for replicate_number in range(3):
    #    print( cell_number, replicate_number, n_regions[replicate_number], len(unmatched_regions[replicate_number]),flush=True)
    #print('',flush=True)
    all_mismatched_regions[cell_number] = {
        'unmatched_regions':unmatched_regions,
        'n_regions':n_regions
    }

    torch.save(all_mismatched_regions,f'./all_mismatched_regions_{cell_number}.pt')


