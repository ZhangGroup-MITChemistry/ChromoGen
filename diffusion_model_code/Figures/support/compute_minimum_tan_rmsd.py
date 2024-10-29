#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=RMSD_compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-99
#SBATCH --output=./log_files/RMSD_compute_%a.log

import torch
import os
import sys
sys.path.insert(0,'../../code/data_utils/SampleClass/')
from Coordinates import Coordinates

save_folder = './rmsd_data/'

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

start = task_id * 1000
stop = start + 1000

#gen_coords = Coordinates('../../data/samples/origami_64_no_embed_reduction/eval_mode/corrected/unguided.pt')[start:stop]
tan_coords = Coordinates('/home/gridsan/gschuette/binz_group_shared/gkks/with_Zhuohan/conformations/Tan/unguided_equivalent.pt')

#gen_coords._values*= 100
tan_coords._values*= 100

min_rmsds = minimum_mean_rmsds(tan_coords,start_idx=start,stop_idx=stop) # Try to show that Tan coords have similar generated coordinates

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

torch.save({'min_rmsds':min_rmsds,'mean_rmsds':mean_rmsds},save_folder + f'tan_on_tan_{task_id}.pt')


