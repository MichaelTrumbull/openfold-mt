#!/bin/bash

source ~/.bashrc
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
conda activate /scratch/00946/zzhang/python-env/openfold-venv
which python

python train_openfold.py \
     /scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files \
     /scratch/00946/zzhang/data/openfold/ls6-tacc/alignment_openfold \
     /scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files \
     full_output \
     2021-10-10 \
     --val_data_dir /scratch/00946/zzhang/data/openfold/cameo/mmcif_files \
     --val_alignment_dir /scratch/00946/zzhang/data/openfold/cameo/alignments \
     --template_release_dates_cache_path=/scratch/00946/zzhang/data/openfold/ls6-tacc/mmcif_cache.json \
     --precision=bf16 \
     --train_epoch_len 128000 \
     --gpus=3 \
     --num_nodes=4 \
     --accumulate_grad_batches 11 \
     --replace_sampler_ddp=True \
     --seed=7152022 \
     --deepspeed_config_path=deepspeed_config.json \
     --checkpoint_every_epoch \
     --obsolete_pdbs_file_path=/scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/obsolete.dat \
     --train_chain_data_cache_path=/scratch/00946/zzhang/data/openfold/ls6-tacc/chain_data_cache.json \
     --wandb \
     --wandb_project openfold-4node \
     --wandb_entity zhaozhang \
     --experiment_name test-4nodes-ls6
