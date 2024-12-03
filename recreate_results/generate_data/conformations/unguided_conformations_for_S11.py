#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=independent_regions
##SBATCH --partition=debug-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --array=0-7
#SBATCH --output=./log_files/independent_regions_%a.log

import ChromoGen
import json
import warnings
import torch
from pathlib import Path
import os 

# To help split up the work load across nodes
tid = tid if (tid:=os.environ.get('SLURM_ARRAY_TASK_ID')) is None else int(tid)

########################################################################################################
# Setup

###########
# Filepaths
try:
    paths = json.load(open('../../config.json','r'))
except:
    raise Exception('You must run generate_config.py before using other scripts.')

###
# ChromoGen model
cg_fp = Path('../../downloaded_data/models/chromogen.pt')
assert cg_fp.exists(), f"chromogen model's file, {cg_fp}, does not exist"

# Change to this if you want to use an independently trained model
# cg_fp = paths['filepaths']['data']['models']['recreated']['chromogen']

###
# Save directories
save_dir = './unguided_conformations/'

###########
# Generation details
batch_size = 1_000
n_batches = 1#00

########################################################################################################
# The actual script

save_dir = Path(save_dir)
save_dir.mkdir(exist_ok=True, parents=True)

############
# Prepare the model

# Initialize & load parameters
cg = ChromoGen.from_file(cg_fp)

try:
    cg.cuda()
except:
    warnings.warn('It seems no GPUs are available. This will take a loooong time without one.')

############
# Figure 4 a-b, IMR-90 conformations

# Attach the relevant files to ChromoGen
for k in range(n_batches):
    if tid is not None and k%tid != 0:
        continue

    f1 = save_dir / f'batch_{k}_raw_output.pt'
    if f1.exists():
        continue
        
    f2 = save_dir / f'batch_{k}_coords.pt'

    ####
    # Generate next batch

    ##
    # Actual generation. 
    # Fig. S11 requires the raw maps, so save those separately
    ''' Unfortunately, there seems to be a bug in how the forward function handles this case
    raw_dists = cg(
        batch_size,
        cond_scales= [0.], # not actually necessary when an integer is passed, but being transparent
        rescaled_phis = [0.],
        return_coords = False,
        correct_distmap = False
    )
    '''
    raw_dists = cg._diffuse(
        batch_size,
        cond_scale = 0., # not actually necessary when an integer is passed, but being transparent
        rescaled_phi = 0.,
        return_coords = False,
        correct_distmap = False
    )

    ##
    # Save the raw distance maps
    
    # Folding to reduce file size by ~50%. 
    # The default option, unlike when I did this in the paper, is to return unnormalized maps from ChromoGen. 
    # If you want to compare the unnormalized maps to our own, use the commented version. 
    # Note that the unnormalization must be performed on unfolded maps, so you'd have to run dists.unfold().unnormalize()
    # and NOT dists.unnormalize().unfold() if you want to analyze those structures. 

    #torch.save(raw_dists.normalize().fold().cpu().values, f)
    torch.save(raw_dists.fold().cpu().values, f1)

    ##
    # Compute optimized coordinates
    coords = raw_dists.coordinates

    ##
    # Save coordinates
    coords.save(f2)



