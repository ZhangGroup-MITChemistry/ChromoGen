from process_raw_data import load_coords, process_coord_data, save_to_hdf5
import pandas as pd
import os 
import h5py

data_dir = '../../data/tan_single-cell_2018/'
dest_file = '../../data/processed_data.hdf5'
datatype  = 'Coordinates' # For now, this file only processes the 3DG data. 
organism = 'Human' # All cells in this dataset are human. 

############
# Don't modify the following. 

# This extracts cell type, cell number information from data folders.
def parse_folder_science2018(folder):

    f = folder.split('_')
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

# This checks whether the existing HDF5 file already contains a particular experiment. 
def already_processed(filepath,organism,cell_type,cell_number,replicate_number):

    if not os.path.exists(filepath):
        # The file doesn't even exist, so it cannot contain the inquiried data.
        return False

    # Check if the relevant key is in the file. 
    with h5py.File(filepath, 'r') as f:
        if 'Coordinates' not in f.keys():
            # No scHi-C data is contained, so the file cannot contain the
            # inquired data. 
            return False 
    
    # Load the data with the stated qualities if it's present. 
    temp = pd.read_hdf(
        filepath,
        key='Coordinates',
        where=[
            f"Organism == {organism}",
            f"Cell_Type == {cell_type}",
            f"Cell == {cell_number}",
            f"Replicate == {replicate_number}"
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
        relevant_file, replicate_number = parse_fname_science2018(file)
        if not relevant_file: 
            # This isn't one of the 3d files we aim to process. 
            continue

        # Check whether the data file already contains this data
        if already_processed(dest_file,organism,cell_type,cell_number,replicate_number):
            # Don't want to duplicate data in the file! 
            continue 
        
        # Load the file in its raw format
        coord_df = load_coords(subfolder + '/' + file)

        # Process the data to match the structure used within the HDF5 file in this study. 
        coord_df = process_coord_data(coord_df,organism,cell_type,cell_number,replicate_number)

        # Save the data to the destination folder. 
        save_to_hdf5(coord_df,dest_file)
        print(f'Processed 3dg replica {replicate_number} from cell {cell_number}')

