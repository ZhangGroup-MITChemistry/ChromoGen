import pandas as pd
import numpy as np

###########################
# Functions to load, mildly process data 
def load_coords(filepath):
    '''
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
        ignore_index=True
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


###########################
# Support functions to process data from Longzhi Tan 
def separate_lineage(coord_df):

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

def place_coord_lineages_sbs(coord_df):
    '''
    Rearrrange coord_df with columns 
    ['Lineage','Chromosome','Genomic_Index','x','y','z']
    to have columns
    ['Chromosome','mat_x','mat_y','may_z','pat_z','pat_y','pat_z'].
    Pad coordinates with np.nan when genomic positions  are present in 
    one copy but not the other. 
    '''

    for chrom in coord_df['Chromosome'].unique().tolist():
        

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

###########################
# Function to save coordinate and scHi-C data to the same searchable HDF5 file 
def save_to_hdf5(coord_df,filepath,indexable_columns,key,compression_level=9):
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

    
    



