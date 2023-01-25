#!/bin/bash

source ~/.bashrc
#export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
#conda activate /scratch/00946/zzhang/python-env/openfold-venv
source /work/09123/mjt2211/ls6/openfold-mt/lib/conda/bin/activate
conda activate openfold_venv
which python

python3 run_pretrained_openfold.py \
    fasta_dir \
    #/scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files \ #data/pdb_mmcif/mmcif_files/ \
    #--uniref90_database_path data/uniref90/uniref90.fasta \
    #--mgnify_database_path data/mgnify/mgy_clusters_2018_12.fa \
    #--pdb70_database_path data/pdb70/pdb70 \
    #--uniclust30_database_path data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --output_dir ./ \
    #--bfd_database_path data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --model_device "cuda:0" \
    #--jackhmmer_binary_path lib/conda/envs/openfold_venv/bin/jackhmmer \
    #--hhblits_binary_path lib/conda/envs/openfold_venv/bin/hhblits \
    #--hhsearch_binary_path lib/conda/envs/openfold_venv/bin/hhsearch \
    #--kalign_binary_path lib/conda/envs/openfold_venv/bin/kalign \
    --config_preset "model_1_ptm" \
    --openfold_checkpoint_path openfold/resources/openfold_params/finetuning_ptm_2.pt \
    --use_precomputed_alignments /scratch/00946/zzhang/data/openfold/cameo/alignments # I think?
    

#python train_openfold.py \
#     /scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files \
#     /scratch/00946/zzhang/data/openfold/ls6-tacc/alignment_openfold \
#     /scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files \
#     full_output \
#     2021-10-10 \
#     --val_data_dir /scratch/00946/zzhang/data/openfold/cameo/mmcif_files \
#     --val_alignment_dir /scratch/00946/zzhang/data/openfold/cameo/alignments \
#     --template_release_dates_cache_path=/scratch/00946/zzhang/data/openfold/ls6-tacc/mmcif_cache.json \
#     --precision=bf16 \
#     --train_epoch_len 128000 \
#     --gpus=3 \
#     --num_nodes=4 \
#     --accumulate_grad_batches 11 \
#     --replace_sampler_ddp=True \
#     --seed=7152022 \
#     --deepspeed_config_path=deepspeed_config.json \
#     --checkpoint_every_epoch \
#     --obsolete_pdbs_file_path=/scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/obsolete.dat \
#     --train_chain_data_cache_path=/scratch/00946/zzhang/data/openfold/ls6-tacc/chain_data_cache.json \
#     --wandb \
#     --wandb_project openfold-4node \
#     --wandb_entity zhaozhang \
#     --experiment_name test-4nodes-ls6
