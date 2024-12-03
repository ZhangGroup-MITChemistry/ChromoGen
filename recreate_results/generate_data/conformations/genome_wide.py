#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=genome_wide
#SBATCH --ntasks=1
##SBATCH --cpus-per-task=40
##SBATCH --gres=gpu:volta:2
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --array=0-5
#SBATCH --output=./log_files/genome_wide_%a.log

import ChromoGen
import json
import warnings
from pathlib import Path
import pickle
import os 

# To help split up the work load across nodes
tid = 0 if (tid:=os.environ.get('SLURM_ARRAY_TASK_ID')) is None else int(tid)
#cell_type = 'GM12878' if tid < 2 else 'IMR90'
#remainder = tid%2
cell_type = 'GM12878' if tid < 3 else 'IMR90'
remainder = tid%3

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
bigWig_fp = Path(paths['filepaths']['data']['outside']['inputs']['downloaded']['DNase-seq'][cell_type]['BigWig'])
alignment_fp = Path(paths['filepaths']['data']['outside']['inputs']['downloaded']['alignment']['h5'])

# Change to the following to use re-created data, noting that the BigWig filepaths are the same either way
# since no processing was involved. 
#bigWig_GM_fp = Path(paths['filepaths']['data']['outside']['inputs']['downloaded']['DNase-seq']['GM12878']['BigWig'])
#bigWig_IMR_fp = Path(paths['filepaths']['data']['outside']['inputs']['downloaded']['DNase-seq']['IMR90']['BigWig'])
#alignment_fp = Path(paths['filepaths']['data']['outside']['inputs']['recreated']['alignment']['h5'])

assert bigWig_fp.exists(), f"{cell_type} BigWig file, {cg_fp}, does not exist"
assert alignment_fp.exists(), f"Processed alignment file, {alignment_fp}, does not exist"

###
# Save directories
save_dir = Path(paths['filepaths']['data']['conformations']['recreated']['genome_wide']) / cell_type
if tid in [0,3]:
    save_dir.mkdir(exist_ok=True,parents=True)
    for chrom in list(range(1,23)) + ['X']:
        (save_dir / f'chrom_{chrom}').mkdir(exist_ok=True)

########################################################################################################
# The actual script

############
# Load the genomic indices we used
start_indices = pickle.load(open('genome_wide_indices.pkl','rb'))

############
# Prepare the model

# Initialize & load parameters
cg = ChromoGen.from_file(cg_fp)

try:
    cg.cuda()
except:
    warnings.warn('It seems no GPUs are available. This will take a loooong time without one (or more).')

'''
(OK, for now, I removed the distribution across GPUs, so this isn't relevant. But... maybe I'll add it back in...)
NOTE: Sampling is distributed across all available GPUs by default. 
You can change this by using one of the following:
 - cg.max_gpus_to_use = <your preferred integer. Default: All available>
 - cg.gpus_to_use = [<torch.device, GPU index, or 'cuda:<gpu index>' 1>, ...]
    - NOTE: You can check the list of available GPUs using cg.available_gpus

NOTE: If you run into RAM issues, you can limit the number of samples generated on each GPU using
the following. (If cg.maximum_samples_per_gpu * len(cg.gpus_to_use) < samples_per_region, the task 
will automatically be split into batches.)
 - cg.maximum_samples_per_gpu = <your preferred integer. Default is 1000>

Similarly, while not relevant in this script, you can limit the number of sequence embeddings to 
generate at once per GPU, using:
 - cg.maximum_regions_embedded_per_GPU = <your preferred integer. Default is 6>
'''

############
# GM12878 conformations from Fig. 3a-c, 4a-b
# Figure 3 a-c, all GM12878

# Attach the relevant files to ChromoGen
cg.attach_data(
    alignment_filepath=alignment_fp,
    bigWig_filepath=bigWig_fp
)

############
# Generate samples
k = 0 
for chrom,indices in start_indices.items():
    # Can generate multiple regions simultaneously, but will generate samples separately here for greater transparency
    for _, genomic_index in indices:
        # By default, ChromoGen will generate samples at BOTH of the relevant
        # weights, but here we generate separately to more transparently obtain the data
        # Note that cond_scale = w + 1, where w was used in the paper. 
        k+=1
        #if k%2 != remainder:
        if k%3 != remainder:
            continue
        for cond_scale, rescaled_phi in [(1.0,0.),(5.,8.)]:
            fp = save_dir/f'chrom_{chrom}/start_idx_{genomic_index//1000}_cond_scale_{cond_scale}_rescaled_phi_{rescaled_phi}.pt'
            if not fp.exists():
                coords = cg(chrom, genomic_index, cond_scales= [cond_scale], rescaled_phis = [rescaled_phi], samples_per_region = 1_000)
                coords.save(fp)


