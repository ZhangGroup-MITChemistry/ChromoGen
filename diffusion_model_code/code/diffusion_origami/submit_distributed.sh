#!/bin/bash

#SBATCH --job-name=origami_small_distributed
#SBATCH --partition=debug-gpu
#SBATCH --nodes=1           # Number of nodes
##SBATCH --ntasks-per-node=2 # Per NODE
#SBATCH --ntasks-per-node=1 # Per NODE
#SBATCH --gres=gpu:volta:2  # Per NODE
#SBATCH --cpus-per-task=40 #20  # Per TASK
#SBATCH --distribution=nopack
#SBATCH --output=./log_files/train_small_distributed_test.log

# Load modules
module load anaconda/Python-ML-2023b # Also loads compatible cuda & nccl versions
module load mpi/openmpi-4.1.5        # To submit parallel tasks

# These flags tell MPI how to set up communication
export MPI_FLAGS="--tag-output --bind-to socket -map-by core -x PSM2_GPUDIRECT=1 -x NCCL_NET_GDR_LEVEL=5 -x NCCL_P2P_LEVEL=5 -x NCCL_NET_GDR_READ=1"

# Set some environment variables needed by torch.distributed 
export MASTER_ADDR=$(hostname -s)
# Get unused port
export MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "MASTER_ADDR : ${MASTER_ADDR}"
echo "MASTER_PORT : ${MASTER_PORT}"

# Submit the job 
#mpirun ${MPI_FLAGS} python train_small_distributed.py
export CUDA_VISIBLE_DEVICES="0,1"
accelerate launch train_small_distributed.py

