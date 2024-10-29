#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=diffusion_dists_to_dcd
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --output=./log_files/diffusion_dists_to_dcd.log


import os
import sys
sys.path.insert(0,'/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/code/data_utils')
from NewSample import coord_to_dcd, load_dist_maps, correct_coords, dists_to_coords, coords_to_dists

def get_dir(filepath):
    return '/'.join([*filepath.split('/')[:-1],''])

def get_filename_root(filepath):
    return '.'.join(filepath.split('/')[-1].split('.')[:-1])

def pt_to_dcd(filepath):

    if not os.path.exists(filepath):
        return

    assert os.path.isfile(filepath), f"{filepath} is not a file!"

    if 'pt_files' in filepath:
        dest_dir = get_dir(filepath).replace('pt_files','dcd_files')
    else:
        dest_dir = get_dir(filepath) + 'dcd_files/'
    sample_name = get_filename_root(filepath)
    if os.path.exists(dest_dir + sample_name + '.dcd'):
        print(f'File {filepath} was already processed! Skipping.')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    ########################
    # Load data
    dists = load_dist_maps(filepath)

    # Remove samples with NaN/infinite-valued distances
    idx_to_keep = dists.isfinite().all(-1).all(-1).flatten()
    if not idx_to_keep.any():
        print(f'No finite conformations obtained from {filepath}')
        return
    dists = dists[idx_to_keep,...] 
    
    # Compute coordinates
    coords = dists_to_coords(dists)

    # Remove samples that will yield NaN/infinite values during optimization
    dists2 = coords_to_dists(coords)
    idx_to_keep = dists2.isfinite().all(-1).all(-1).flatten()
    if not idx_to_keep.any():
        print(f'No finite conformations obtained from {filepath}')
        return
    dists = dists[idx_to_keep,...]
    coords = coords[idx_to_keep,...]

    # Make the coordinates match the not-quite-physical distance maps better
    coords = correct_coords(coords,dists)

    # Save as dcd
    coord_to_dcd(coords,dest_dir,sample_name)

def convert_directory(directory):

    if directory == '':
        directory = './'
    if not directory[-1] == '/':
        directory+= '/'
    files = [directory + file for file in os.listdir(directory)]
    files.sort()
    
    for file in files:
        if not os.path.isfile(file):
            continue
        try:
            pt_to_dcd(file)
        except:
            continue

import pandas as pd
from ConfigDataset import ConfigDataset
def convert_Tan_data(
    configs_filepath,
    embedding_index_filepath,
    destination_directory,
    chrom,
    region_idxs,
    nbins=64
):

    if type(region_idxs) == int:
        region_idxs = [region_idxs]

    dest_dir = destination_directory
    if dest_dir == '':
        dest_dir = './'
    if dest_dir[-1] != '/':
        dest_dir = dest_dir + '/'
    
    # Load rosetta stone file to convert region_idx to genomic index
    index = pd.read_pickle(embedding_index_filepath)

    config_ds = ConfigDataset(
        configs_filepath,
        segment_length=nbins,
        remove_diagonal=False,
        batch_size=0,
        normalize_distances=False,
        geos=None,
        organisms=None,
        cell_types=None,
        cell_numbers=None,
        chroms=[chrom],
        replicates=None,
        shuffle=False,
        allow_overlap=True,
        two_channels=False,
        try_GPU=False,
        #mean_dist_fp=mean_dist_fp,
        #mean_sq_dist_fp=mean_sq_dist_fp
    )

    for region_idx in region_idxs:
        _,chrom_,start_idx = index[chrom][region_idx]
        assert chrom == chrom_, f'Seems to be an error with rosetta stone... Wanted chromosome {chrom}, received {chrom_}'
        info,coords = config_ds.fetch_specific_coords(chrom,start_idx)

        sample_name = f'chrom_{chrom}_region_{region_idx}_nbins_{nbins}'
        coord_to_dcd(coords,dest_dir,sample_name)
        info.to_pickle(dest_dir+sample_name+'_experiment_info.pkl')
        
if __name__ == '__main__':

    pt_to_dcd('../../data/samples/origami_64_no_embed_reduction/pt_files/sample_unguided_milestone_120_combined.pt')
