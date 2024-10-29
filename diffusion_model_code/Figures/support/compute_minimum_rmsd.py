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
def minimum_rmsds(coords,references):

    min_rmsds = None
    for reference in tqdm(references):
        if min_rmsds is None:
            min_rmsds = compute_rmsd(coords,reference)

        else:
            min_rmsds = torch.min( min_rmsds, compute_rmsd(coords,reference) )

    return min_rmsds

task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

start = task_id * 1000
stop = start + 1000

gen_coords = Coordinates('../../data/samples/origami_64_no_embed_reduction/eval_mode/corrected/unguided.pt')[start:stop]
tan_coords = Coordinates('/home/gridsan/gschuette/binz_group_shared/gkks/with_Zhuohan/conformations/Tan/unguided_equivalent.pt')

# Note: This caused numerical issues, but I'm leaving it as-is since I didn't re-run after determining that
gen_coords._values*= 100
tan_coords._values*= 100

min_rmsds = minimum_rmsds(tan_coords,gen_coords) # Try to show that Tan coords have similar generated coordinates

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

torch.save(min_rmsds,save_folder + f'tan_on_gen_{task_id}.pt')


