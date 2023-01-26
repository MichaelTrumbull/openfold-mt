#!/bin/bash
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
    
