#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -n 1          # number of cores
#SBATCH -t 0-2:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q grp_psulc       # QOS
#SBATCH -G a6000:1
#SBATCH --job-name="test"
#SBATCH -o slurm.psulc_gpu_%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.psulc_gpu_%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load required modules for job's environment
module load mamba/latest
module load cuda-11.7.0-gcc-12.3.0
source activate oxdnapy12

./code tunnel