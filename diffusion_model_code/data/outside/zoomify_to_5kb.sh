#!/bin/bash 

#SBATCH --job-name=zoomify_to_5kb
##SBATCH --partition=debug-cpu 
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --output=./log_files/zoomify_to_5kb_2.log

resolution=1000
final_resolution=5000
file=./GM12878_hg19.mcool
outfile=./GM12878_hg19_3.mcool

#cooler zoomify --resolutions=5000,20000 $file::/resolutions/1000 --balance --base-uri=$file::/resolutions/20000 --out=$file::/resolutions/5000
#cooler zoomify --resolutions=5000,20000 $file::/resolutions/1000 --balance --base-uri=$file::/resolutions/20000 --out=$outfile

cooler zoomify -p $SLURM_JOB_CPUS_PER_NODE --resolutions=5000,20000 $file::/resolutions/1000 --balance --out=$outfile #::/resolutions/1000

#cooler zoomify --resolutions=5000,20000 $file::/resolutions/1000 --balance --out=$outfile\.cool

#cooler zoomify --resolutions=5000 $file::/resolutions/1000 --balance --out=$file::/resolutions/5000

#cooler balance -p $SLURM_JOB_CPUS_PER_NODE $file\::/resolutions/$final_resolution


