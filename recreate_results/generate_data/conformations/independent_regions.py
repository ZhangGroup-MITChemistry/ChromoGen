#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=independent_regions
##SBATCH --partition=debug-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --array=0-1
#SBATCH --output=./log_files/independent_regions_%a.log

import ChromoGen
import json
import warnings
from pathlib import Path
import os 

# To help split up the work load across nodes
tid = tid if (tid:=os.environ.get('SLURM_ARRAY_TASK_ID')) is None else int(tid)

########################################################################################################
# Get filepaths
try:
    paths = json.load(open('../../config.json','r'))
except:
    raise Exception('You must run generate_config.py before using other scripts.')

###
# ChromoGen
cg_fp = Path(paths['filepaths']['data']['models']['downloaded']['chromogen'])
assert cg_fp.exists(), f"chromogen model's file, {cg_fp}, does not exist"

# Change to this if you want to use an independently trained model
# cg_fp = paths['filepaths']['data']['models']['recreated']['chromogen']

###
# Inputs
bigWig_GM_fp = Path(paths['filepaths']['data']['outside']['inputs']['downloaded']['DNase-seq']['GM12878']['BigWig'])
bigWig_IMR_fp = Path(paths['filepaths']['data']['outside']['inputs']['downloaded']['DNase-seq']['IMR90']['BigWig'])
alignment_fp = Path(paths['filepaths']['data']['outside']['inputs']['downloaded']['alignment']['h5'])

# Change to the following to use re-created data, noting that the BigWig filepaths are the same either way
# since no processing was involved. 
#bigWig_GM_fp = Path(paths['filepaths']['data']['outside']['inputs']['downloaded']['DNase-seq']['GM12878']['BigWig'])
#bigWig_IMR_fp = Path(paths['filepaths']['data']['outside']['inputs']['downloaded']['DNase-seq']['IMR90']['BigWig'])
#alignment_fp = Path(paths['filepaths']['data']['outside']['inputs']['recreated']['alignment']['h5'])

assert bigWig_GM_fp.exists(), f"GM12878 BigWig file, {cg_fp}, does not exist"
assert bigWig_IMR_fp.exists(), f"IMR-90 BigWig file, {cg_fp}, does not exist"
assert alignment_fp.exists(), f"Processed alignment file, {alignment_fp}, does not exist"

###
# Save directories
save_dir = Path(paths['filepaths']['data']['conformations']['recreated']['independent_regions'])

gm_save = save_dir / 'GM12878'
imr_save = save_dir / 'IMR90'

gm_save.mkdir(exist_ok=True,parents=True)
imr_save.mkdir(exist_ok=True,parents=True)

########################################################################################################
# The actual script

############
# Prepare the model

# Initialize & load parameters
cg = ChromoGen.from_file(cg_fp)

try:
    cg.cuda()
except:
    warnings.warn('It seems no GPUs are available. This will take a loooong time without one.')

############
# GM12878 conformations from Fig. 3a-c, 4a-b
# Figure 3 a-c, all GM12878

# Attach the relevant files to ChromoGen
cg.attach_data(
    alignment_filepath=alignment_fp,
    bigWig_filepath=bigWig_GM_fp
)

# Can generate multiple regions simultaneously, but will generate samples separately here for greater transparency
n = 0
for chrom,start_idx in [
    # Fig 3
    (1,8_680_000),
    (1,29_020_000),
    (22,26_260_000),
    # Fig 4
    (6,6_000_000),
    (6,131_860_000),
    (21,28_500_000)
]:

    # By default, ChromoGen will generate samples at BOTH of the relevant
    # weights, but here we generate separately to more transparently obtain the data
    # Note that cond_scale = w + 1, where w was used in the paper. 
    n+=1
    if tid is not None and n%2 != tid:
        continue
    for cond_scale, rescaled_phi in [(1.0,0.),(5.,8.)]:
        coords = cg(chrom, start_idx, cond_scales= [cond_scale], rescaled_phis = [rescaled_phi], samples_per_region = 10_000)
        coords.save(gm_save/f'sample_{chrom}_{start_idx//1000}_{cond_scale}_{rescaled_phi}.pt')

############
# Figure 4 a-b, IMR-90 conformations

# Attach the relevant files to ChromoGen
cg.attach_data(
    alignment_filepath=alignment_fp,
    bigWig_filepath=bigWig_IMR_fp
)
for chrom,start_idx in [
    (6,6_000_000),
    (6,131_860_000),
    (21,28_500_000)
]:

    # By default, ChromoGen will generate samples at BOTH of the relevant
    # weights, but here we generate separately to more transparently obtain the data
    # Note that cond_scale = w + 1, where w was used in the paper. 
    n+=1
    if tid is not None and n%2 != tid:
        continue
    for cond_scale, rescaled_phi in [(1.0,0.),(5.,8.)]:
        coords = cg(chrom, start_idx, cond_scales= [cond_scale], rescaled_phis = [rescaled_phi], samples_per_region = 10_000)
        coords.save(imr_save/f'sample_{chrom}_{start_idx//1000}_{cond_scale}_{rescaled_phi}.pt')



