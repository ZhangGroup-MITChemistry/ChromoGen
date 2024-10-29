#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=organize_embeddings
#SBATCH -c 48
#SBATCH --output=./log_files/organize_embeddings1.log

'''
Received an out of memory error when saving the file before, so 
process separately here... 
'''
import pandas as pd 
import numpy as np
import os

#dest_fp = lambda chrom: f'../../data/embeddings_128/chrom_{chrom}.tar.gz'#'../../data/processed_data.hdf5'
#dest_fp = lambda chrom: f'../../data/embeddings_64_after_transformer/chrom_{chrom}.tar.gz'
#dest_dir = '../../data/embeddings_128_after_transformer/'
dest_dir = '../../data/embeddings_64_IMR90/'
dest_fp = lambda chrom: dest_dir + f'/chrom_{chrom}.tar.gz'

#raw_dir = lambda chrom: f'../../data/raw_embeddings/run_scripts_{chrom}/'
#raw_dir = lambda chrom: f'/home/gridsan/gschuette/binz_group_shared/zlao/for_greg/hg19/run_scripts_{chrom}/'
#raw_dir = lambda chrom: f'/home/gridsan/gschuette/binz_group_shared/zlao/for_greg/hg19_with_new_hic_optim/node_embedding/run_scripts_{chrom}/'
#raw_dir = lambda chrom: f'/home/gridsan/gschuette/binz_group_shared/zlao/for_greg/after_transformation/size128/node_embedding/run_scripts_{chrom}/'
#raw_dir = lambda chrom: f'/home/gridsan/gschuette/binz_group_shared/zlao/for_greg/after_transformation/size64/node_embedding/run_scripts_{chrom}/'
raw_dir = lambda chrom: f'/home/gridsan/gschuette/binz_group_shared/zlao/for_greg/IMR_size64/node_embedding/run_scripts_{chrom}/'

chroms = [str(k) for k in range(1,23)]
chroms.append('X') 

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for chrom in chroms: 

    # Load the data from the temporary pickle file and place in a DataFrame
    idx_cols = ['Region_Length','Genomic_Index']
    embed_df = pd.DataFrame(
        pd.read_pickle(f'./Zhuohan_Data_IMR/data_files_{chrom}.pkl')
    ).rename( # More consistent with how I label the other data objects
        columns={'data':'Data','region_size':'Region_Length','start_idx':'Genomic_Index'}
    ).drop(   # This is actually unnecessary info
        columns='idxs'
    ).astype( 
        {'Region_Length':np.int64,'Genomic_Index':np.int64}
    ).sort_values( # This should also be unnecessary, but to be safe...
        by=idx_cols,
        axis='index',
        ignore_index=True
    )

    # Sanity checks 
    assert not embed_df.isna().any().any(), f'Chromosome {chrom} file contains NaN values!'
    assert len(embed_df) == len(os.listdir(raw_dir(chrom))), f'Incorrect number of entries for chromosome {chrom} file!'

    # Zhuohan saved the genomic index data in kb, so need to adjust related info to bp to match my other objects
    embed_df['Region_Length']*= 1000
    embed_df['Genomic_Index']*= 1000
    
    # Create a MultiIndex for easy fetching of specific embeddings in the DataLoader
    embed_df['Chromosome'] = chrom
    idx_cols = ['Region_Length','Chromosome','Genomic_Index']
    embed_df.index = pd.MultiIndex.from_tuples(
        list(map(tuple,embed_df[idx_cols].values)),
        names=idx_cols
    )

    # Drop the now-redundant columns
    embed_df = embed_df.drop(columns=idx_cols)

    # Save this to streamlined tar.gz file. 
    embed_df.to_pickle(dest_fp(chrom))

