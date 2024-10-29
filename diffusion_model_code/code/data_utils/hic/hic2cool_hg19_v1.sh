#!/bin/bash 

#SBATCH --job-name=hic2cool_hg19
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=./log_files/hic2cool_hg19.log

# This works as of hic2cool version 0.8.3

module purge
module load anaconda/Python-ML-2023b

# FASTQ>=30, primary+combined version
#hic2cool extract-norms -e ./GM12878_insitu_primary+replicate_combined_30.hic ../../../data/outside/GM12878_hg19.mcool
hic2cool convert ./GM12878_insitu_primary+replicate_combined_30.hic ../../../data/outside/GM12878_hg19.mcool --nproc=24


# FASTQ>= 30, primary version
#hic2cool extract-norms -e ./GM12878_insitu_primary_30.hic ../../../data/outside/GM12878_hg19.mcool

