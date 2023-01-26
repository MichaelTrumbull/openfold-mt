#!/bin/bash

#SBATCH -J openfold-mt           # Job name
#SBATCH -o log_output       # Name of stdout output file #use %j for job number
#SBATCH -e log_error       # Name of stderr error file
#SBATCH -p gpu-a100          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes
#SBATCH -n 1              # Total # of mpi tasks
#SBATCH -t 00:15:00        # Run time (hh:mm:ss)

###srun run-tacc-inf.sh

echo START OF run-tacc-inf
source ~/.bashrc
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
echo AFTER expor
source /work/09123/mjt2211/ls6/openfold-mt/lib/conda/bin/activate
echo AFTER src
conda activate openfold_venv
echo activated venv
which python

python3 run_pretrained_openfold.py \
    fasta_dir \
    scratch/00946/zzhang/data/openfold/cameo/mmcif_files \
    --output_dir ./ \
    --model_device "cuda:0" \
    --config_preset "model_1_ptm" \
    --openfold_checkpoint_path openfold/resources/openfold_params/finetuning_ptm_2.pt \
    --use_precomputed_alignments /scratch/00946/zzhang/data/openfold/cameo/alignments 
    
    
