#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -n 8           # number of cores
#SBATCH -t 3-0:00:00   # time in d-hh:mm:ss
#SBATCH -p general     # partition
#SBATCH -G a6000:1
#SBATCH -q grp_psulc     # QOS
#SBATCH --job-name="rmv_base"
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment
 
#Load required modules for job's environment
module load mamba/latest
module load cuda-11.7.0-2d
source activate oxdnapy12
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps-pipe_$SLURM_TASK_PID
export CUDA_MPS_LOG_DIRECTORY=/tmp/mps-log_$SLURM_TASK_PID
mkdir -p $CUDA_MPS_PIPE_DIRECTORY
mkdir -p $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d

#Anything below this will actually run 
# ~/oxDNA/build/bin/oxDNA ./input
python3 /home/ava/MetaBackbone_project/Metabackbone-scripts/Notebook/structure_editting_scripts/run_evolutionary_algorithm.py   