#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=create_dcd_files
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:volta:1
#SBATCH --output=./log_files/create_dcd_files.log

import os
import sys
sys.path.insert(0,'../../code/data_utils/')
from diffusion_dists_to_dcd import convert_directory

directories = [
    directory for directory in os.listdir('./') if os.path.isdir(directory)
]

for directory in directories:
    convert_directory(directory)


