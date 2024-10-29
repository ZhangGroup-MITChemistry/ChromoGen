#!/bin/bash 

#SBATCH --job-name=hic2cool_hg19
#SBATCH --partition=debug-cpu 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --output=./log_files/hic2cool_IMR90_hg19.log

<<'###'
We want a cooler file with contact information at 20kb resolution. 
However, the .hic file downloaded from GEO does not include contact information at this resolution, so we must (1) convert 
contact information at a higher resolution and (2) generate 20kb resolution data using cooler's zoomify function.

In the first step, we'll convert the highest resolution available in the source file (1kb). This will be slower than 
converting contacts at, e.g., 10kb resolution, but it will provide greater flexibility in the future if additional resolutions
are desired. 
###

# resolution 1000 not available in the IMR-90 data
resolution=5000 #1000
final_resolution=20000
#hic_file=./GM12878_insitu_primary+replicate_combined_30.hic
#dest_file=../../../data/outside/GM12878_hg19 # Don't include root here
hic_file=./IMR90_insitu_combined_30.hic
dest_file=../../../data/outside/IMR90_hg19


# Load the environment where current hic2cool & cooler modules are installed 
module purge
module load anaconda/Python-ML-2023b

# Generate cooler file from hic file. This works as of hic2cool version 0.8.3.
dest_file1=$dest_file\.cool
hic2cool convert $hic_file $dest_file1 --nproc=$SLURM_JOB_CPUS_PER_NODE -r $resolution

# Place the normalization vectors into the cool file, as well
#hic2cool extract-norms $dest_file1 $hic_file

######
# Alternative to the above. But... probably do it after zoomify (see the version of this command located there)
# cooler balance -p $SLURM_JOB_CPUS_PER_NODE $dest_file1

# The hic file is no longer needed, so delete it 
rm $hic_file 

# Get the new resolution. This works as of cooler version 0.9.3.
#dest_file1=$dest_file\.mcool
cooler zoomify -p $SLURM_JOB_CPUS_PER_NODE -r $final_resolution $dest_file1 

dest_file1=$dest_file\.mcool
# If normalization still needed... 
cooler balance -p $SLURM_JOB_CPUS_PER_NODE $dest_file1\::/resolutions/$final_resolution


