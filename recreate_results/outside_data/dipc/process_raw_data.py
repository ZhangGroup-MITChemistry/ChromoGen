# A somewhat gross script, but it's what we did/it works
    
import pandas as pd
import numpy as np
import os

############################################################################################################
# Support functionality 

########
# Load data 

def load_coords(filepath):
    '''
    Use pandas to load the 3dg files containing the relevant coordinates. 
    
    Assumes Dip-C 3dg format
    '''
    coord_df = pd.read_csv(
        filepath,
        sep='\t',
        header=None,
        names=['Chromosome','Genomic_Index','x','y','z']
    )

    return coord_df

def load_contacts(filepath):
    '''
    Load scHi-C contact. Not relevant to the ChromoGen paper. 
    
    Assumes "two columns of 'chromosome,coordinate,haplotype' separated by a tab," 
    as described at https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM3271347
    '''

    # Load the data
    con_df = pd.read_csv(
        filepath,
        header=None,
        sep='\t',
        names=['Loc0','Loc1']
    )

    # Remove rows with NaN/missing values.
    con_df.dropna(
        axis='index',
        inplace=True,
    #    ignore_index=True
    )

    # Split the data separated by commas within each column.
    for i in range(2):
        con_df[[f'Chrom{i}',f'Genomic Index{i}',f'Haplotype{i}']] = con_df[f'Loc{i}'].str.split(',',expand=True)

    # Remove the unprocessed columns. 
    con_df.drop(columns=['Loc0','Loc1'],inplace=True)

    # Standardize the datatypes in each column.
    con_df = con_df.astype({'Chrom0':str,'Genomic Index0':np.int64,'Haplotype0':str,'Chrom1':str,'Genomic Index1':np.int64,'Haplotype1':str})

    # Remove duplicated lines, if present. 
    con_df.drop_duplicates(inplace=True,keep='first',ignore_index=True)

    return con_df

########
# Functions specific to parsing the data files

###
# 3DG files
def separate_lineage(coord_df):
    '''
    Redefine chromosomes from, chr<chrom>(<lineage>) convention to chr<chrom>
    convention with <lineage> placed in another column in the DataFrame.
    '''

    # Define lineages in their own column for easy access
    coord_df['Lineage'] = [k[-4:-1] for k in coord_df['Chromosome']]

    # Reduce the chromosome column to just the chromosome number 
    # (or X/Y, where relevant)
    coord_df['Chromosome'] = [ k[:-5] for k in coord_df['Chromosome']] 

    # Place lineage in the first position
    col_order = coord_df.columns.tolist()[:-1]
    col_order.insert(0,'Lineage') 
    coord_df = coord_df[col_order] 

    return coord_df 

def sort_coord_df(coord_df):
    '''
    Want all maternal to come before all paternal bead locations. 
    Within mat/pat chromosomes, want chromosome order 1,2,...,22,X,Y. 
    Within each chromosome, want bead location to be sequential. 
    '''
    
    # Must make single-digit numbers equal to f'0{number}' to get the
    # desired sorting of chromosome numbers. 
    vals = {str(k):'0'+str(k) for k in range(1,10)}
    coord_df.replace({'Chromosome':vals},inplace=True)

    # Sort values with the described order of importance. 
    coord_df.sort_values(
        by=['Lineage','Chromosome','Genomic_Index'],
        axis='index',
        inplace=True,
        ignore_index=True
    )

    # Remove the leading '0's where relevant. 
    vals = {'0'+str(k):str(k) for k in range(1,10)}
    coord_df.replace({'Chromosome':vals},inplace=True)


###########################
# Functions that processes coordinate data after loading from 3dg file (Dip-C format)
def process_coord_data(coord_df):
    '''
    After loading data from the 3dg file, process the DataFrame
    so that it agrees with the data structure used to save 3D 
    structures in a searchable HDF5 file containing 3D data 
    from multiple organisms, cell types, chromosomes. 
    '''

    # Remove all rows with nan values in any column, as
    # we have no use for these rows. 
    coord_df.dropna(
        axis='index',
        inplace=True,
        ignore_index=True
    )

    # When the starting file contains NaNs or missing values, the
    # datatypes may not be what is expected. This function ensures
    # that they obey the standard construction. 
    coord_df = coord_df.astype({
        'Chromosome':str,
        'Genomic_Index':np.int64, 
        'x':np.float64,
        'y':np.float64,
        'z':np.float64
    })

    # Certain data files downloaded from the GEO database contain 
    # duplicated data, so we must remove these duplicates. 
    coord_df.drop_duplicates(inplace=True,keep='first',ignore_index=True)
    
    # Convert Chromosome information from f"{chrom}({lineage})" to 
    # f"{chrom}" and create a new column to track the lineage. 
    coord_df = separate_lineage(coord_df) 

    # Want all maternal to come before all paternal bead locations. 
    # Within mat/pat copies, want chromosome order 1,2,...,22,X,Y. 
    # Within each chromosome, want bead location to be sequential. 
    sort_coord_df(coord_df)

    return coord_df

def find_permitted_lengths(genomic_index):
    '''
    This function assumes that the filepath is valid and already 
    contains the relevant 3D structural data.

    Find the maximum UNINTERRUPTED (i.e., no NaNs present, don't reach chromosome's end) 
    region that be drawn when starting from each monomer. 
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

###
# scHi-C files
def clean_con_df(con_df):
    '''
    For easier data analysis later, want the index
    at 'Genomic Index0' <= 'Genomic Index1' everywhere. 
    We also want these values to increase monotonically, 
    with higher weight on 'Genomic Index0'

    NOTE: This assumes that the index is 0,1,2,...,len(con_df)
    '''

    # Ensure index0 <= index1 in all rows
    idx = np.where(con_df['Genomic Index0'] > con_df['Genomic Index1'])[0]
    if len(idx) > 0:
        old = ['Chrom0','Genomic Index0','Haplotype0','Chrom1','Genomic Index1','Haplotype1']
        new = ['Chrom1','Genomic Index1','Haplotype1','Chrom0','Genomic Index0','Haplotype0']
        con_df.loc[idx,old] = con_df.loc[idx,new].values
        del old, new

        # In case duplicates were missed due to the originally swapped columns, remove duplicates here. 
        con_df.drop_duplicates(inplace=True,keep='first',ignore_index=True)
    
    # Sort the values 
    con_df.sort_values(
        by=['Genomic Index0','Genomic Index1'],
        axis='index',
        inplace=True,
        ignore_index=True
    )

###########################
# Function to save coordinate and scHi-C data to the same searchable HDF5 file 
def save_to_hdf5(coord_df,filepath,compression_level=9):
    '''
    Append the new information to the HDF5 file where all data is being saved. 

    key should correspond to the type of data, e.g. scHi-C or Coordinates. 
    '''

    # Rename "Cell Type" and column to "Cell_Type" to avoid issues with invalid datatype representations
    # Rename columns with spaces in their names to avoid issues with invalid datatype representations
    cols = {}
    for col in coord_df.columns.tolist():
        if ' ' in col: 
            cols[col] = '_'.join(col.split(' '))
    coord_df.rename(columns=cols,inplace=True)
    
    if coord_df.columns[-1] == 'z':
        key = 'Coordinates'
        indexable_columns = ['Organism','Cell_Type','Cell','Replicate','Lineage','Chromosome','Genomic_Index']
    elif coord_df.columns[-1] == 'Haplotype1':
        key = 'scHiC'
        indexable_columns = True # Want all columns to be indexable upon loading 
    elif coord_df.columns[-1] == 'permitted_length':
        key = 'permitted_length'
        indexable_columns = True # Want all columns to be indexable upon loading 
    else:
        raise Exception('The save_to_hdf5 function cannot determine the type of data in this DataFrame.')

    coord_df.to_hdf(
        filepath,                       # The file where the data should be saved. 
        key=key,                        # The key corresponds to the type of data being saved so that tables with different
                                        # column structures can be saved to the same file. 
        mode='a',                       # Append to existing data so that we can process cells one at a time without
        append=True,                    # running into memory issues. 
        index=False,                    # We don't care about the index from the DataFrame, so don't save it
        format='table',                 # Slower than the fixed format, but allows the data to be indexed upon loading
        data_columns=indexable_columns, # Which columns do we want to be indexable upon loading data from the HDF5 file? 
        complevel=compression_level     # Maximum data compression is 9; final file should use a value of 9 
    )

    # Return column names to their originals 
    if len(cols) > 0: 
        cols = {b:a for a,b in cols.items()}
        coord_df.rename(columns=cols,inplace=True)

class DatasetCreator:

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

if __name__ == '__main__':
    from pathlib import Path

    # This extracts cell type, cell number information from data directories.
    def parse_directory_science2018(directory):
    
        f = directory.split('_')
        cell_type = f[-3]
        cell_number = f[-1]
    
        if '-' in cell_number:
            cell_number = cell_number.split('-')[0]
    
        return cell_type, int(cell_number)
    
    # This extracts replicate information from filenames.
    def parse_fname_science2018(fname):
        '''
        returns (<desired file>,<replicate_number>)
        '''
    
        if 'impute3.round4.clean.3dg' not in file:
            # We don't care about this file
            return False, -1
    
        if 'rep1' in file:
            replicate_number = 1
        elif 'rep2' in file:
            replicate_number = 2
        else:
            replicate_number = 0
    
        return True, replicate_number

    # Prep some filepaths
    this_dir = Path(__file__).parent
    ds_fp = this_dir / 'processed_data.h5'
    raw_data_dir = this_dir / 'tan_single-cell_2018'

    # Initialize the DatasetCreator 
    ds = DatasetCreator(str(ds_fp))

    # We'll add some extra data to the HDF5 file in case
    # extra organisms/experiments are added eventually. 
    acc_number = 'GSE117876' # From the 2018 Longzhi Tan paper
    organism = 'Human'    

    # Get all the subdirectories inside the main supplementary info directory
    subdirectories = [str(f.relative_to(this_dir)) for f in raw_data_dir.glob('*') if f.is_dir()]
    print(subdirectories)
    subdirectories.sort() # Causes experiments to show up in order. 
    dsname = ds_fp.name
    for subdirectory in subdirectories:
        # Collect some metadata from the directory name
        cell_type, cell_number = parse_directory_science2018(subdirectory.split('/')[-1])
        print(f'Processing 3DG files for cell {cell_number} among {cell_type} cells.',flush=True)
    
        # Locate all files that we wish to place in the HDF5 file
        # to be used by the dataloader function.
        filenames = os.listdir(subdirectory)
        filenames.sort() # Causes replicas to show up in the correct order
        for file in filenames:
            # Locate the 'clean' 3dg file and collect the replicate number to store
            # as metadata in the HDF5 file. 
            relevant_file, replicate_number = parse_fname_science2018(file)
            if not relevant_file:
                # This isn't one of the 3d files we aim to process.
                continue
    
            raw_file = subdirectory + '/' + file
            ds.process_3dg_file(raw_file,acc_number,organism,cell_type,
                                cell_number,replicate_number)
            
            print('\t'f'Replica {replicate_number} processed and saved to the destination file, {dsname}.',flush=True)

