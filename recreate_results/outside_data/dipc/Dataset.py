from process_raw_data import load_coords, process_coord_data, load_contacts, clean_con_df
import numpy as np
import pandas as pd
import os 

def find_permitted_lengths(genomic_index):
    '''
    This function assumes that the filepath is valid and already 
    contains the relevant 3D structural data.
    '''

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

    # Pad the backend to account for the fact that len(sep) == len(genomic_index)-1
    perlen[len(perlen)] = 0 

    return perlen


def place_coord_lineages_sbs(coord_df):
    '''
    Rearrrange coord_df with columns 
    ['Lineage','Chromosome','Genomic_Index','x','y','z']
    to have columns
    ['Chromosome','mat_x','mat_y','may_z','pat_z','pat_y','pat_z'].
    Pad coordinates with np.nan when genomic positions  are present in 
    one copy but not the other. 
    '''

    combined_dfs = {}
    sub_cols = ['Genomic_Index','x','y','z']
    for chrom in coord_df['Chromosome'].unique().tolist():

        # Get the subset of the main DataFrame corresponding to the maternal copy of this chromosome
        mat = coord_df[ (coord_df['Chromosome']==chrom) & (coord_df['Lineage']=='mat') ][sub_cols]
        mat.rename(columns={'x':'mat_x','y':'mat_y','z':'mat_z'},inplace=True)

        # Perform the same steps for the paternal chromosome
        pat = coord_df[ (coord_df['Chromosome']==chrom) & (coord_df['Lineage']=='pat') ][sub_cols]
        pat.rename(columns={'x':'pat_x','y':'pat_y','z':'pat_z'},inplace=True)

        # Merge the two matrices
        c = pd.merge(mat,pat,on='Genomic_Index') 

        if len(c) == 0:
            continue

        # JUST IN CASE, resort the genomic index. Pretty sure this line doesn't do anything though. 
        c.sort_values('Genomic_Index',ignore_index=True,inplace=True)

        # Find the number of consecutive monomer following each known monomer (i.e. those not
        # removed in the cleaning process) 
        c['Permitted_Lengths'] = find_permitted_lengths(c['Genomic_Index'].values)

        # Place the combined DataFrame in the dictionary being sent back 
        combined_dfs[chrom] = c
    
    return combined_dfs

class Dataset:

    def __init__(self,filepath):

        # Set filepath attribute
        self.filepath = filepath

        # Get the DataFrame containing information 
        # that helps keep track of the data in each 
        # object associated with the datafile. 
        self._get_info_dfs()


    def _get_info_dfs(self):

        if os.path.exists(self.filepath): 
            # For now, just assume the file is 
            # properly formatted. 
            self._coord_info = pd.read_hdf(
                self.filepath,
                key='coord_info'
            )
        else: 
            # Need to initialize these objects.
            self._coord_info = pd.DataFrame(
                columns=[
                    'Accession',
                    'Organism',
                    'Cell_Type',
                    'Cell',
                    'Replicate',
                    'Chromosome',
                    'idx_min',
                    'idx_max'
                ]
            )

    def process_3dg_file(self,filepath,acc_number,organism,cell_type,
                         cell_number,replicate_number):
        
        # Check if this data has been processed before
        row = pd.DataFrame({
            'Accession':[acc_number],
            'Organism':[organism],
            'Cell_Type':[cell_type],
            'Cell':[cell_number],
            'Replicate':[replicate_number],
            'Chromosome':[''],
            'idx_min':[np.nan],
            'idx_max':[np.nan]
        })
        sub_cols = ['Accession','Organism','Cell_Type','Cell','Replicate']
        if len(self._coord_info) > 0:
            if ( self._coord_info[sub_cols].values == row[sub_cols].values ).all(1).any():
                # This file has already been processed
                return 

        # Load the file in its raw format. 
        coord_df = load_coords(filepath)

        # Process the data to obey the assumptions made in this class
        coord_df = process_coord_data(coord_df)

        # Place the maternal/paternal coordinates side-by-side in the same object 
        coord_dfs = place_coord_lineages_sbs(coord_df)

        # Update the info object. 
        rows = []
        end_idx = -1 if len(self._coord_info) == 0 else self._coord_info['idx_max'][len(self._coord_info)-1]
        for chrom,coord_df in coord_dfs.items():
            row['Chromosome'] = [chrom]
            row['idx_min'] = [end_idx + 1]
            end_idx+= len(coord_df) 
            row['idx_max'] = [end_idx]

            rows.append(row.copy())

        rows = pd.concat(rows,ignore_index=True)
        # Shift the index to the correct overall values prior to saving
        rows.index+= len(ds._coord_info) 

        # Save the info 
        indexable_columns = True
        rows.to_hdf(
            self.filepath,                  # The file where the data should be saved. 
            key='coord_info',               # The key corresponds to the type of data being saved so that tables with different
                                            # column structures can be saved to the same file. 
            mode='a',                       # Append to existing data so that we can process cells one at a time without
            append=True,                    # running into memory issues. 
            index=False,                    # We don't care about the index specific to the DataFrame, so don't save it
            format='table',                 # Slower than the fixed format, but allows the data to be indexed upon loading
            data_columns=indexable_columns, # Which columns do we want to be indexable upon loading data from the HDF5 file? 
            complevel=9                     # Maximum data compression is 9
        )

        # Add the each coord_df in coord_dfs to the save file
        coord_df = pd.concat(coord_dfs,ignore_index=True)
        # Shift the index to match the saved DataFrame
        coord_df.index+= rows['idx_min'].iloc[0] 
        
        # Save to file
        indexable_columns = ['Genomic_Index','index']
        coord_df.to_hdf(
            self.filepath,                  # The file where the data should be saved. 
            key='Coordinates',              # The key corresponds to the type of data being saved so that tables with different
                                            # column structures can be saved to the same file. 
            mode='a',                       # Append to existing data so that we can process cells one at a time without
            append=True,                    # running into memory issues. 
            index=True,                     # We don't care about the index specific to the DataFrame, so don't save it
            format='table',                 # Slower than the fixed format, but allows the data to be indexed upon loading
            data_columns=indexable_columns, # Which columns do we want to be indexable upon loading data from the HDF5 file? 
            complevel=9                     # Maximum data compression is 9
        )

        # Update the info object
        self._get_info_dfs()

    
    def process_scHiC_file(self,filepath,acc_number,organism,cell_type,
                         cell_number,replicate_number):

        # Check if this data has already been processed 
        row = pd.DataFrame({
            'Accession':[acc_number],
            'Organism':[organism],
            'Cell_Type':[cell_type],
            'Cell':[cell_number],
            'Replicate':[replicate_number],
            'Chromosome':[''],
            'idx_min':[np.nan],
            'idx_max':[np.nan]
        })
        sub_cols = ['Accession','Organism','Cell_Type','Cell','Replicate']
        if len(self._coord_info) > 0:
            if ( self._coord_info[sub_cols].values == row[sub_cols].values ).all(1).any():
                # This file has already been processed
                return 
        
        # Load the data
        con_df = load_contacts(filepath)

        # Clean the data (all modifications performed using in-place operations) 
        clean_con_df(con_df)

        # Clean the data 
        # Uhhhh, it seems I never finished this part. But it's fine since we don't use the
        # scHi-C contact data in the ChromoGen paper. 
        

            