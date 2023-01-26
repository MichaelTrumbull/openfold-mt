#!/bin/bash

#SBATCH -J openfold-mt           # Job name
#SBATCH -o output_log       # Name of stdout output file #use %j for job number
#SBATCH -e error_log       # Name of stderr error file
#SBATCH -p gpu-a100          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes
#SBATCH -n 1              # Total # of mpi tasks
#SBATCH -t 00:30:00        # Run time (hh:mm:ss)

srun ./run-tacc-inf.sh
