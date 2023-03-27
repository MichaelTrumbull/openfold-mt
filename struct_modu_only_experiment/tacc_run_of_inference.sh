#!/bin/bash

#SBATCH -J openfold-mt           # Job name
#SBATCH -o o.log       # Name of stdout output file #use %j for job number
#SBATCH -e e.log       # Name of stderr error file
#SBATCH -p gpu-a100          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes
#SBATCH -n 1              # Total # of mpi tasks
#SBATCH -t 07:00:00        # Run time (hh:mm:ss)

source ~/.bashrc
source /work/09123/mjt2211/ls6/openfold-mt/lib/conda/bin/activate
conda activate openfold_venv

for r in 0 1 2 3
do
for i in -1 0 1 2 3 4 5 6 7
do
## this is the file that will be loaded in to determine where to zero 
python3 set_iteration_doe_file.py -r $r -i $i

CUDA_VISIBLE_DEVICES=0 python3 run_pretrained_openfold.py \
    /work/09123/mjt2211/ls6/openfold-mt/DOE_fasta_dir \
    /scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files \
    --output_dir ./ \
    --model_device "cuda:0" \
    --config_preset "model_4_ptm" \
    --openfold_checkpoint_path openfold/resources/openfold_params/finetuning_no_templ_ptm_1.pt \
    --use_precomputed_alignments /scratch/09120/sk844/validation_set_cameo/alignments \
    --subtract_plddt \
    --skip_relaxation
done
done

python3 set_iteration_doe_file.py -r -1 -i -1

CUDA_VISIBLE_DEVICES=0 python3 run_pretrained_openfold.py \
    /work/09123/mjt2211/ls6/openfold-mt/DOE_fasta_dir \
    /scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files \
    --output_dir ./ \
    --model_device "cuda:0" \
    --config_preset "model_4_ptm" \
    --openfold_checkpoint_path openfold/resources/openfold_params/finetuning_no_templ_ptm_1.pt \
    --use_precomputed_alignments /scratch/09120/sk844/validation_set_cameo/alignments \
    --subtract_plddt \
    --skip_relaxation

##
# note: this template free run requires
#   1. >model 3
#   2. finetuning with no template
#   3. removed of .hhr file from alignment data
##