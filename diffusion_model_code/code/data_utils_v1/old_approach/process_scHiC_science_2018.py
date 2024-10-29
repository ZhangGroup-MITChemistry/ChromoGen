from process_raw_data import load_contacts, clean_con_df, save_to_hdf5
import pandas as pd
import os
import h5py

data_dir = '../../data/tan_single-cell_2018/'
dest_file = '../../data/processed_data.hdf5'
datatype  = 'Contacts' # For now, this file only processes the scHi-C data. 
organism = 'Human' # All cells in this dataset are human. 

############
# Don't modify the following. 

def parse_folder_science2018(folder):

    f = folder.split('_')
    cell_type = f[-3]
    cell_number = f[-1]

    if '-' in cell_number: 
        cell_number = cell_number.split('-')[0]

    return cell_type, int(cell_number)

def insert_metadata(con_df):
    
    for col,val in [('Cell',cell_number),('Cell Type',cell_type),('Organism',organism)]:
        con_df.insert(
            loc=0,
            column = col,
            value  = val
        )

# This checks whether the existing HDF5 file already contains a particular experiment. 
def already_processed(filepath,organism,cell_type,cell_number):

    if not os.path.exists(filepath):
        # The file doesn't even exist, so it cannot contain the inquiried data.
        return False

    # Check if the relevant key is in the file. 
    with h5py.File(filepath, 'r') as f:
        if 'scHiC' not in f.keys():
            # No scHi-C data is contained, so the file cannot contain the
            # inquired data. 
            return False 
    
    # Load the data with the stated qualities if it's present. 
    temp = pd.read_hdf(
        filepath,
        key='scHiC',
        where=[
            f"Organism == {organism}",
            f"Cell_Type == {cell_type}",
            f"Cell == {cell_number}"
        ]
    )
    if len(temp) == 0:
        return False
    return True 

folders = os.listdir(data_dir)
folders.sort()
for folder in folders:

    # Collect some metadata from the folder name
    cell_type, cell_number = parse_folder_science2018(folder)
    
    # Locate all files that we wish to place in the HDF5 file
    # to be used by the dataloader function.
    subfolder = data_dir + folder
    filenames = os.listdir(subfolder)
    filenames.sort() # Causes replicas to show up in the correct order
    for file in filenames:
        
        if '.clean.con' not in file: 
            # This isn't one of the scHi-C files we aim to process. 
            continue 

        # Check whether the data file already contains this data
        if already_processed(dest_file,organism,cell_type,cell_number):
            # Don't want to duplicate data in the file! 
            continue 

        # Load the data and process it. 
        con_df = load_contacts(subfolder + '/' + file)
        clean_con_df(con_df) # These operations are performed inplace 

        # Insert metadata for organizational purposes within the hdf5 file. 
        insert_metadata(con_df)

        # Save the data to the destination folder. 
        save_to_hdf5(con_df,dest_file)

        print(f'Processed scHi-C data from cell {cell_number}')



