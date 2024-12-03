#!/bin/bash

#SBATCH --job-name=prep_HiC_data
#SBATCH --cpus-per-task=24
#SBATCH --array=0-1
#SBATCH --output=./log_files/prep_HiC_data_%a.log

#################################
# Environment setup/check

# Select the specific versions of hic2cool and cooler that we want to use (if needed)
#alias hic2cool="/home/gridsan/gschuette/.local/bin/hic2cool"
#alias cooler="/home/gridsan/gschuette/.local/bin/cooler"

# Check the cooler/hic2cool versions
cooler --version   # version 0.9.3 used in paper
hic2cool --version # version 0.8.3 used in paper

#################################
# Select the values to process the Hi-C data for a specific cell type 

# Want to download Hi-C data to the directory containing this script so other scripts run properly, so get this directory. 
# Also, select the number of threads to use in hic2cool & cooler operations
if [ -z "${SLURM_JOB_CPUS_PER_NODE}" ]
then 
    # This is NOT being run within a slurm job
    # Use all available cores
    NPROC=`nproc --all`
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
else
    NPROC=$SLURM_JOB_CPUS_PER_NODE
    SCRIPT_DIR=`scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}' | head -n 1`
    SCRIPT_DIR=$(dirname "${SCRIPT_DIR}") # remove filename/duplicated paths
fi

# Default to GM12878 if not running inside a slurm task array
if [ -z "${SLURM_ARRAY_TASK_ID}" ] || [ $SLURM_ARRAY_TASK_ID -eq 0 ]
then
    # GM12878
    resolution=1000  # Resolution to pull from .hic file
    resolutions="1000,5000,20000" # Resolutions to includes in final .mcool file
    download_url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FGM12878%5Finsitu%5Fprimary%2Breplicate%5Fcombined%5F30.hic" 
    # ^Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525 (Rao et al. 2014)
    dest_file=$SCRIPT_DIR\/GM12878_hg19 # Don't include root here
else
    # IMR-90
    resolution=5000  # Resolution to pull from .hic file
    resolutions="5000,20000" # Resolutions to includes in final .mcool file
    download_url="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl/GSE63525%5FIMR90%5Fcombined%5F30.hic" 
    # ^Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525 (Rao et al. 2014)
    dest_file=$SCRIPT_DIR\/IMR90_hg19 # Don't include root here
fi

#################################
# Run the (unfinished parts of the) pipeline 

###
# STEP 1: Download raw data
[ -f $dest_file\.hic ] || [ -f $dest_file\.cool ] || [ -f $dest_file\.mcool ] || curl $download_url -o $dest_file\.hic

<<'###'
We want a cooler file with contact information at 20kb resolution.
However, the .hic file downloaded from GEO does not include contact information at this resolution, so we must (1) convert
contact information at a higher resolution and (2) generate 20kb resolution data using cooler's zoomify function.

As such, we'll convert the highest resolution available in the source file (5kb). This will be slower than
converting contacts at, e.g., 10kb resolution, but it will provide greater flexibility in the future if additional resolutions
are desired.
###

###
# STEP 2: Generate cooler file from hic file.
[ -f $dest_file\.mcool ] || [ -f $dest_file\.cool ] || hic2cool convert $dest_file\.hic $dest_file\.cool --nproc $NPROC -r $resolution

# Delete the hic file since it's no longer needed
[ -f $dest_file\.hic ] && ([ -f $dest_file\.cool ] || [ -f $dest_file\.mcool ]) && rm $dest_file\.hic

###
# STEP 3: Zoomify 
# Get the new resolution. 
if [ -f $dest_file\.cool ] && [ ! -f $dest_file\.mcool ]
then 
    cooler zoomify -p $NPROC --resolutions $resolutions $final_resolution $dest_file\.cool
elif [ -f $dest_file\.mcool ]
then
    cooler zoomify -p $NPROC --resolutions $resolutions $final_resolution $dest_file\.mcool
fi

# Delete the .cool file since it's no longer needed
[ -f $dest_file\.mcool ] && [ -f $dest_file\.cool ] && rm $dest_file\.cool

###
# STEP 4: Balance
# NOTE: Steps 3 & 4 can be accomplished together by simply adding the --balance flag to the zoomify operation. However, we didn't do that in the paper, so we split the operations here for consistency. 
if [ -f $dest_file\.mcool ]; then
    echo $resolutions | sed -n 1'p' | tr ',' '\n' | while read intermediate_resolution; do  
        cooler balance -p $NPROC $dest_file\.mcool::/resolutions/$intermediate_resolution
    echo $word; done
fi
