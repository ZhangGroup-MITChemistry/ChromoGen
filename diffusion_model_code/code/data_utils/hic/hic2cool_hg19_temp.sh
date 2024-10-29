#!/bin/bash 

#SBATCH --job-name=hic2cool_hg19
#SBATCH --partition=debug-cpu 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=./log_files/hic2cool_hg19.log

<<'###'
We want a cooler file with contact information at 20kb resolution. 
However, the .hic file downloaded from GEO does not include contact information at this resolution, so we must (1) convert 
contact information at a higher resolution and (2) generate 20kb resolution data using cooler's zoomify function.

In the first step, we'll convert the highest resolution available in the source file (1kb). This will be slower than 
converting, e.g., contacts at 10kb resolution, but it will provide greater flexibility in the future if additional resolutions
are desired. 
###

resolution=1000
final_resolution=20000
hic_file=./GM12878_insitu_primary+replicate_combined_30.hic
#dest_file=../../../data/outside/GM12878_hg19.mcool
dest_file=../../../data/outside/GM12878_hg19_1000.cool

# Load the environment where current hic2cool & cooler modules are installed 
module purge
module load anaconda/Python-ML-2023b

# Generate cooler file from hic file. This works as of hic2cool version 0.8.3.  
#hic2cool convert $hic_file $dest_file --nproc=$SLURM_JOB_CPUS_PER_NODE -r $resolution

# The hic file is no longer needed, so delete it 
#rm $hic_file 

# Get the new resolution. This works as of cooler version 0.9.3.

hic2cool extract-norms $dest_file $hic_file

#cooler zoomify -r=$final_resolution $dest_file 


