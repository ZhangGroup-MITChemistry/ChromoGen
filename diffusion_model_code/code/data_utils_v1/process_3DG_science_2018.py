data_dir = '../../data/tan_single-cell_2018/'
filepath = '../../data/processed_data.hdf5'

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


acc_number = 'GSE117876'
organism = 'Human'
ds = Dataset(filepath)
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

        raw_file = subfolder + '/' + file
        ds.process_3dg_file(raw_file,acc_number,organism,cell_type,
                            cell_number,replicate_number)