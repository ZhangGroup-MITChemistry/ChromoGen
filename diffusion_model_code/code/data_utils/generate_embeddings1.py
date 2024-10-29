#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=generate_embeddings1
##SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:volta:1
#SBATCH -c 40
##SBATCH -t 0-0:05
##SBATCH -t 0-36:00:00
#SBATCH --output=./log_files/generate_embeddings1.log

N = 1

import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import sys
sys.path.insert(1,'./')
from SequencesDataset import SequencesDataset
from ConfigDataset import ConfigDataset
sys.path.insert(2,'../frontend/')
from Tranmodel import Tranmodel 

####################################################################
# Select the regions to be processed 
####################################################################

# We want to process all chromosomes. 
if N == 0:
    chroms = [*range(1,23),'X']
else: 
    chroms = ['X',*reversed(range(1,23))]

# This is where the embeddings will be saved 
save_folder = '../../data/embeddings/'
dest_fp = lambda chrom: save_folder+f'chrom_{chrom}.tar.gz'#'../../data/processed_data.hdf5'
if not os.path.exists(save_folder): 
    os.mkdir(save_folder)

##################################
# Use the ConfigDataset class to identify
# the genomic regions whose embeddings are of interest to us. 
print('Determining which genomic regions must be embedded.\n',flush=True)
def get_genomic_regions(
    nbeads = 65,
    dataset_filepath = '../../data/processed_data.hdf5',
    batch_size = 64,
    two_channels = False,
    allow_overlap = True, 
    chroms=None
):
    
    cds = ConfigDataset(
        dataset_filepath,
        segment_length=nbeads,
        batch_size=batch_size,
        normalize_distances=True,
        geos=None,
        organisms=None,
        cell_types=None,
        cell_numbers=None,
        chroms=chroms,
        replicates=None,
        shuffle=True,
        allow_overlap=allow_overlap, 
        two_channels=two_channels,
        try_GPU=True,
        mean_dist_fp='../../data/mean_dists.pt',
        mean_sq_dist_fp='../../data/squares.pt'
    )
    
    return cds.get_genomic_regions()
    
regions = get_genomic_regions()

# Rename some columns to match what had previously been used downstream
# to avoid having to rewrite code. 
regions = regions.rename(
    columns={
        'Chromosome':'Chromosome',
        'Start':'Genomic_Index',
        'Stop':'Region_Length'
    }
)
regions['Region_Length']-= regions['Genomic_Index']
idx_cols = ['Region_Length','Chromosome','Genomic_Index'] # for later

##################################
# Load the pretrained frontend, which generates the embeddings 
frontend = Tranmodel.get_pretrained_model()
try: 
    frontend.cuda()
except: 
    pass 

for chrom in chroms: 
    print('/////////////////////////////////////////////////////////////////')
    if os.path.exists(dest_fp(chrom)):
        print(f'Chromosome {chrom} already processed!\n\n',flush=True)
        continue
    print(f'Chromosome {chrom}:\n',flush=True)
    
    # Load the raw sequencing data for this chromosome
    chr = f'chr{chrom}'
    ds = SequencesDataset(chroms=chr)
    print('\n')
    
    # Get the genomic regions relevant to this chromosome
    embed_df = regions[regions['Chromosome'] == str(chrom)].reset_index(drop=True)
    
    # Form the embeddings
    data = []
    idx_to_keep = np.ones(len(embed_df),dtype=bool)
    for i in tqdm(range(len(embed_df)), desc = f'Embedding Generation Progress (Chromosome {chrom})', total = len(embed_df)): # Show progress
        try: 
            data.append( frontend( # Generate embeddings of sequencing data
                ds.fetch([ (chr,embed_df.loc[i,'Genomic_Index'],embed_df.loc[i,'Region_Length']) ]).to(frontend.device) # Prepare sequencing data
            ).to('cpu') )
        except: 
            idx_to_keep[i] = False
    embed_df = embed_df[idx_to_keep]
    embed_df['Data'] = data
    del data 
    '''
    embed_df['Data'] = [
        frontend( # Generate embeddings of sequencing data
            ds.fetch([ (chr,embed_df.loc[i,'Genomic_Index'],embed_df.loc[i,'Region_Length']) ]).to(frontend.device) # Prepare sequencing data
        ).to('cpu') for i in tqdm(range(len(embed_df)), desc = f'Embedding Generation Progress (Chromosome {chrom})', total = len(embed_df)) # Show progress
    ]
    '''
    '''
    print(f'Generating Embeddings for Chromosome {chrom}',flush=True)
    embed_df['Data'] = [
        frontend( # Generate embeddings of sequencing data
            ds.fetch([ (chr,embed_df.loc[i,'Genomic_Index'],embed_df.loc[i,'Region_Length']) ]).to(frontend.device) # Prepare sequencing data
        ).to('cpu') for i in range(len(embed_df))
    ]
    '''
    '''
    for i in range(len(embed_df)): 
        raw = ds.fetch([ (chr,embed_df.loc[i,'Genomic_Index'],embed_df.loc[i,'Region_Length']) ]).to(frontend.device)
        embeddings.append( frontend(raw).to('cpu') )
    '''
    # Clear RAM 
    del ds, chr

    # Create a MultiIndex for easy fetching of specific embeddings in the DataLoader
    embed_df.index = pd.MultiIndex.from_tuples(
        list(map(tuple,embed_df[idx_cols].values)),
        names=idx_cols
    )

    # Drop the now-redundant columns
    embed_df = embed_df.drop(columns=idx_cols)

    # Save this to streamlined tar.gz file. 
    print(f'Saving Embeddings for Chromosome {chrom}',flush=True)
    embed_df.to_pickle(dest_fp(chrom))

    # Free RAM 
    del embed_df

    print('\n')


    

