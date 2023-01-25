#!/bin/bash

#SBATCH -J openfold-mt           # Job name
#SBATCH -o openfold.o%j       # Name of stdout output file
#SBATCH -e openfold.e%j       # Name of stderr error file
#SBATCH -p gpu-a100          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes
#SBATCH -n 12              # Total # of mpi tasks
#SBATCH -t 01:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH -A Deep-Learning      # Project/Allocation name (req'd if you have more than 1)
# #SBATCH --mail-user=mjt2211@columbia.edu

srun ./run-tacc-inf.sh
