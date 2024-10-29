#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=RMSD_compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-95
#SBATCH --output=./log_files/RMSD_compute_chrom_22_no_replicates_with_homo_%a.log

import torch
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0,'../../code/data_utils/SampleClass/')
from Coordinates import Coordinates
from Trajectory import Trajectory
sys.path.insert(1,'../../code/data_utils/')
from ConfigDataset import ConfigDataset

save_folder = './rmsd_data/'

# Load the Tan configurations
config_ds = ConfigDataset(
    '../../data/processed_data.hdf5',
    segment_length=64,
    remove_diagonal=False,
    batch_size=0,
    normalize_distances=False,
    geos=None,
    organisms=None,
    cell_types=None,
    cell_numbers=None,
    chroms=['22'],
    replicates=None,
    shuffle=True,
    allow_overlap=True,
    two_channels=False,
    try_GPU=True,
    mean_dist_fp=None,
    mean_sq_dist_fp=None
)


def compute_rmsd(coords,reference):
    aligned_coords = coords.trajectory.clone().superpose(reference)
    return aligned_coords.rmsd(reference)

from tqdm.auto import tqdm
def minimum_mean_rmsds(coords,start_idx=None,stop_idx=None,max_comparisons=100_000):#,references):

    max_comparisons = min(max_comparisons,len(coords))

    if start_idx is not None:
        if stop_idx is None:
            stop_idx = min(start_idx + max_comparisons,len(coords))
    elif stop_idx is not None:
        start_idx = max(0,stop_idx - max_comparisons)
    else:
        start_idx = 0
        stop_idx = max_comparisons
    
    max_comparisons = stop_idx - start_idx 


    idx = torch.arange(len(coords))

    for i in tqdm(range(start_idx,stop_idx)):
        temp_rmsds = compute_rmsd(coords[idx!=i],coords[i])
        if i == start_idx:
            min_rmsds = temp_rmsds
            mean_rmsds = temp_rmsds / max_comparisons
        else:
            min_rmsds = torch.min( min_rmsds, temp_rmsds)
            mean_rmsds+= temp_rmsds / max_comparisons

    return min_rmsds, mean_rmsds

task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

#start = task_id * 1000
#stop = start + 1000

#gen_coords = Coordinates('../../data/samples/origami_64_no_embed_reduction/eval_mode/corrected/unguided.pt')[start:stop]
#tan_coords = Coordinates('/home/gridsan/gschuette/binz_group_shared/gkks/with_Zhuohan/conformations/Tan/unguided_equivalent.pt')

rosetta = pd.read_pickle('../../data/embeddings_64_after_transformer/rosetta_stone.pkl')
d = '/home/gridsan/gschuette/binz_group_shared/gkks/with_Zhuohan/produce_samples/GM/full_scan/corrected/'
gen_coords = []
tan_coords = []
for f in os.listdir(d):
    if f[-6:] != '_22.pt':
        continue
    gen_coords.append(Coordinates(d+f))
    
    # Avoid double-loading Tan coords 
    if '_5.0_' not in f:
        continue

    _,region_idx,_,_,_,chrom = f.split('_')
    chrom = chrom.split('.')[0]
    region_idx = int(region_idx)
    start = rosetta[chrom][region_idx][-1]

    temp_coord_info,temp_coords = config_ds.fetch_specific_coords(chrom,start)

    observed_chroms = []
    rows_to_use = []
    for i,row in temp_coord_info.iterrows():
        info = (row.Cell,row.Lineage)
        if info in observed_chroms:
            continue
        observed_chroms.append(info)
        rows_to_use.append(i)


    tan_coords.append(
        Coordinates(
            temp_coords[
                #np.where(temp_coord_info.Replicate == temp_coord_info.Replicate.Replicate[0])[0],
                torch.tensor(rows_to_use),
                ...
            ]
        )
    )

gen_coords = gen_coords[0].append(gen_coords[1:])
tan_coords = tan_coords[0].append(tan_coords[1:])
n = gen_coords.num_beads
i = (500-n)//2
j = i + n
homo_coords = Trajectory.from_dcd(
    '/home/gridsan/gschuette/binz_group_shared/gkks/with_Amogh/'+\
    '40_Bead_systems/fully_connected_40_bead_PLM_only/LAMMPS_Files/'+\
    'run_e0.35/DUMP_FILE.dcd',
    num_beads=500
)[:len(gen_coords),i:j,:]
# Just use the same number of coordinates as in the generated case to save computation time

#gen_coords = [ # Chromosome 22
#    Coordinates(d+f) for f in os.listdir(d) if f[-6:] == '_22.pt'
#]
#gen_coords = gen_coords[0].append(gen_coords[1:])

#d = '/home/gridsan/gschuette/binz_group_shared/gkks/with_Zhuohan/tan_full_scan/full_scan/'
#tan_coords = [ # Chromosome 22
#    Coordinates(d+f) for f in os.listdir(d) if f[-6:] == '_22.pt'
#]
#tan_coords = tan_coords[0].append(tan_coords[1:])

# Note: This caused numerical issues! 
#gen_coords._values*= 100
#tan_coords._values*= 100

#min_rmsds = minimum_mean_rmsds(gen_coords,start_idx=start,stop_idx=stop) # Try to show that Tan coords have similar generated coordinates

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

n = 0 
gen_idx = torch.arange(len(gen_coords))
tan_idx = torch.arange(len(tan_coords))
homo_idx = torch.arange(len(homo_coords))
#N = 3 * len(gen_idx) + 3 * len(tan_idx) + 3 * len(homo_idx)
N = len(gen_idx) + len(tan_idx) + 3 * len(homo_idx) # temp since we're skipping tan/gen-exclusives

with tqdm(initial=0,total=N) as pbar:
    for coords,label,coord_idx in [(gen_coords,'gen',gen_idx),(tan_coords,'tan',tan_idx),(homo_coords,'homo',homo_idx)]:
        for reference,label_r in [(gen_coords,'gen'),(tan_coords,'tan'),(homo_coords,'homo')]:

            # temp since we've already run this computation
            if label in ['gen','tan'] and label_r in ['gen','tan']:
                continue

            rmsd_values = []
            
            if label == label_r:
                for i,coord in enumerate(coords):
                    n+=1
                    pbar.update(1)
                    if n%96 != task_id:
                        continue
                    rmsd_values.append( compute_rmsd(coords[coord_idx!=i],coord) )
        
            else:
                for coord in reference:
                    n+=1
                    pbar.update(1)
                    if n%96 != task_id:
                        continue
                    rmsd_values.append( compute_rmsd(coords,coord) )


            torch.save( rmsd_values, save_folder + f'{label}_on_{label_r}_chrom_22_{task_id}_no_replicates.pt')

'''
with tqdm(initial=0,total=N) as pbar:
    for option in ['gen_on_gen','tan_on_tan','tan_on_gen','gen_on_tan']:

        rmsd_values = []

        if option == 'gen_on_tan':
            for coord in tan_coords:
                pbar.update(1)
                n+=1
                if n%96 != task_id:
                    continue
                rmsd_values.append( compute_rmsd(gen_coords,coord) )

        elif option == 'tan_on_gen':
            for coord in gen_coords:
                pbar.update(1)
                n+=1
                if n%96 != task_id:
                    continue
                rmsd_values.append( compute_rmsd(tan_coords,coord) )

        elif option == 'tan_on_tan':
            for i,coord in enumerate(tan_coords):
                n+=1
                pbar.update(1)
                if n%96 != task_id:
                    continue
                rmsd_values.append( compute_rmsd(tan_coords[tan_idx!=i],coord) )

        else:
            for i,coord in enumerate(gen_coords):
                pbar.update(1)
                n+=1
                if n%96 != task_id:
                    continue
                rmsd_values.append( compute_rmsd(gen_coords[gen_idx!=i],coord) )

        torch.save( rmsd_values, save_folder + f'{option}_chrom_22_{task_id}_no_replicates.pt' )
'''

