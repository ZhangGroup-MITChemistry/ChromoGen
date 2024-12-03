#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=train_cgen_diffuser
#SBATCH --gres=gpu:volta:1
#SBATCH -c 20
#SBATCH -t 0-65:00:00
#SBATCH --output=./log_files/train_reimplemented.log

from ChromoGen.model import Diffuser
import torch
from pathlib import Path
import os
import subprocess

###########################################################
# Choose your filepaths. Using default parameters
# everywhere else rather than passing kwargs since we 
# set the values used in the paper as the default values 
# in the relevant methods/classes in the ChromoGen package. 
###########################################################

# Dip-C conformations using our HDF5-formatted scheme
config_fp = '../../downloaded_data/conformations/DipC/processed_data.h5'

# Directory containing pre-computed EPCOT embeddings. 
# EPCOT is far slower than one timestep of the diffusion model
# (which is all that's needed during training), so we computed 
# these ahead of time and placed them in .tar.gz tarballs.
embedding_dir = '../../downloaded_data/embeddings/GM12878/'

# Save directory
save_folder = './results/'

# Chromosomes to use while training
chroms = ['21']#[str(k) for k in range(1,23)]

###########################################################
# Initialize objects
###########################################################

# Initialize the model. This will automatically use the hyperparameters used in the paper. 
diffuser = Diffuser()

# Put the model on the GPU, if possible. 
if torch.cuda.is_available():
    diffuser.cuda()
    
# Get the trainer for this model. This will automatically select the training parameters used in the paper. 
# This assumes that the Dip-C conformations have been processed/embeddings have been generated using the 
# other relevant scripts. 
trainer = diffuser.get_trainer(
    embeddings_directory = embedding_dir,
    configuration_filepath = config_fp,
    results_folder = save_folder,
    chromosomes = chroms
)

# Load the last checkpoint (if it exists). 
trainer.load(
    milestone = -1,    # -1 indicates 'load whichever milestone is most recent and has both model & trainer files available'. A warning is output if no model-Trainer file pair exists, but the process will continue. 
    parallel_load=True # Whether to load the model/trainer files in parallel. Should set to False if these files are on the same drive. 
)

###########################################################
# Train
###########################################################

# Initiate actual model training. 
trainer.train()