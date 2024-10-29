#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=process_permitted_length
#SBATCH -c 4
#SBATCH --output=./log_files/process_permitted_length.log

filepath = '../../data/processed_data.hdf5'

import sys
sys.path.insert(1, './')

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import h5py
import os 
from process_raw_data import save_to_hdf5

def get_unique_chroms(filepath,key='Coordinates'): 
    '''
    Given an HDF5 file at filepath, return all unique chromosomes 
    with 3D structures. Unique identifiers are: Organism, cell type, 
    cell number, replicate number, chromosome, and lineage (mat/pat). 
    '''

    # Load the columns containing the identifier information. 
    chromid_df = pd.read_hdf(
        filepath, 
        key=key,
        columns=['Organism','Cell_Type','Cell','Replicate','Lineage','Chromosome']
    )

    # Drop_duplicates will reduce the chromid_df DataFrame to its unique rows. 
    return chromid_df.drop_duplicates(ignore_index=True)

def drop_preprocessed_chroms(chromid_df,filepath):
    '''
    Given a set of chromosomes, check which have already been processed
    for their valid start positions. Remove those from the DataFrame to 
    avoid reprocessing them. 
    '''
    
    # Check if the relevant key is in the file (i.e. if any lengths have been
    # processed at all) 
    with h5py.File(filepath, 'r') as f:
        if 'permitted_length' not in f.keys():
            # No scHi-C data is contained, so the file cannot contain the
            # inquired data. 
            return

    # Load the unique identifiers for permitted lengths already preprocessed. 
    perlen_df = get_unique_chroms(filepath,'permitted_length')
    
    # Compare the two DataFrames to obtain an index of rows to keep
    idx = np.where((chromid_df.merge(perlen_df,how='left',indicator=True)['_merge'] != 'left_only'))[0]
    
    # Keep the rows that appear in the inquired DataFrame; drop others. Reset the index, as well. 
    if len(idx) > 0:
        chromid_df.drop(idx,axis='index',inplace=True)
        chromid_df.reset_index(drop=True, inplace=True)

    # chromid_df was modified inplace, so no need to return anything. 

def load_genomic_indices(filepath,organism,cell_type,cell_number,chromosome,lineage,replicate_number):
    '''
    Load the genomic indices for monomers with known 3D positions in a specific chromsome 
    from a specific replicate from a specific experiment. 

    The function assumes that the smallest genomic separation between known monomer positions 
    corresponds to the the resolution and returns that value. 
    '''

    genomic_index = pd.read_hdf(
        filepath,
        key='Coordinates',
        where=[
            f"Organism == {organism}",
            f"Cell_Type == {cell_type}",
            f"Cell == {cell_number}",
            f"Chromosome == '{chromosome}'",
            f"Lineage == {lineage}",
            f"Replicate == {replicate_number}"
        ],
        columns=['Genomic_Index']
    )['Genomic_Index'].values

    #resolution = ( genomic_index[1:] - genomic_index[:-1] ).min()
    return genomic_index#, resolution 

def find_permitted_lengths(filepath,organism,cell_type,cell_number,chromosome,lineage,replicate_number):
    '''
    filepath: The hdf5 file containing all processed data. 
    resolution: Resolution of the 3D models (in bp) 
    required_length: How many uninterrupted beads are required? 

    This function assumes that the filepath is valid and already 
    contains the relevant 3D structural data.
    '''

    # Load the genomic indices corresponding to each monomer in the 3D structure of this chromosome
    genomic_index = load_genomic_indices(filepath,organism,cell_type,cell_number,chromosome,
                                                     lineage,replicate_number)

    # Find the genomic separation between neighboring monomers. 
    sep = genomic_index[1:] - genomic_index[:-1]

    # The minimum value should correspond to the resolution. 
    resolution = sep.min() 

    # We are only interested in whether or not there are skipped monomers between neighboring, 
    # known monomers, so make this boolean (True/1 == nearest neighbors, False/0 == bead was skipped). 
    # Make it a pd Series object to simplify manipulation. 
    sep = pd.Series(sep == resolution)

    # We are interested in the number of beads you can travel FORWARD from any selected monomer
    # before arriving at a monomer that was removed during the Dip-C cleaning process.
    # SO, reverse the index of sep. 
    sep = sep.reindex(index=sep.index[::-1])

    # Cumulative sum provides the number of nearest neighbor positions identified thus far. 
    perlen = sep.cumsum()

    # Subtract the values associated with positions where a monomer is missing to reset the 
    # count of 'number of monomers you can traverse from here' to 0. In effect, this 
    # restarts the count each time a monomer is skipped in the 3D data. 
    perlen-= sep.cumsum().where(~sep).ffill().fillna(0)

    # Set the index back to its original location, realigning the data with bead indices. 
    perlen = perlen.reindex(index=perlen.index[::-1]).astype(np.int64)

    # Place the series object in a DataFrame, since that's how we're saving data in this project.
    perlen_df = pd.DataFrame()
    perlen_df['permitted_length'] = perlen

    # Finally, insert the metadata needed to search the larger structure when loading from the HDF5 file. 
    perlen_df.insert(loc=0,column='Replicate',value=replicate_number)
    perlen_df.insert(loc=0,column='Lineage',value=lineage)
    perlen_df.insert(loc=0,column='Chromosome',value=chromosome)
    perlen_df.insert(loc=0,column='Cell',value=cell_number)
    perlen_df.insert(loc=0,column='Cell_Type',value=cell_type)
    perlen_df.insert(loc=0,column='Organism',value=organism)

    return perlen_df

def find_all_permitted_lengths(filepath):
    '''
    Given an HDF5 file containing 3D conformations, get all valid starting positions 
    for all 3D structures contained in that file. 
    '''

    # Get the set of unique chromosomes to be analyzed. 
    chromid_df = get_unique_chroms(filepath)

    # Remove any rows whose valid starts have already been processed
    drop_preprocessed_chroms(chromid_df,filepath) 
 
    # This will ensure that identifiers are sent to find_permitted_lenghts in the proper order
    identifiers = ['Organism','Cell_Type','Cell','Chromosome','Lineage','Replicate']
    
    # Address ALL of these unique rows.
    nrows = len(chromid_df)
    parlen_dfs = []
    for _,row in tqdm(chromid_df.iterrows(), desc = 'Processing Permitted Length', total = nrows):

        parlen_dfs.append( find_permitted_lengths(filepath,*row[identifiers].values) )
        
        # Get the permitted lengths, formatted as needed for saving. 
        perlen_df = find_permitted_lengths(filepath,*row[identifiers].values)
        
        # Save the permitted lengths DataFrame to file
        #save_to_hdf5(perlen_df,temp_filepath,compression_level=compression_level)
        
    
find_all_permitted_lengths(filepath)
    
    
