#!/bin/bash

source ~/.bashrc
source /data/mjt2211/openfold-mt/lib/conda/bin/activate
conda activate openfold_venv

CUDA_VISIBLE_DEVICES=0 python3 run_pretrained_openfold.py \
    /data/mjt2211/openfold-mt/struct_modu_only_experiment/fasta_dir \
    /data/mjt2211/openfold-mt/struct_modu_only_experiment/mmcif_files \
    --output_dir ./ \
    --model_device "cuda:0" \
    --config_preset "model_4_ptm" \
    --openfold_checkpoint_path openfold/resources/openfold_params/finetuning_no_templ_ptm_1.pt \
    --use_precomputed_alignments /data/mjt2211/openfold-mt/struct_modu_only_experiment/alignments \
    --subtract_plddt \
    --skip_relaxation

##
# note: this template free run requires
#   1. >model 3
#   2. finetuning with no template
#   3. removed of .hhr file from alignment data
##