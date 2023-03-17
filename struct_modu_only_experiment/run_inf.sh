#!/bin/bash


source ~/.bashrc
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH #### 
source /work/09123/mjt2211/ls6/openfold-mt/lib/conda/bin/activate ##### change path
conda activate openfold_venv
which python
### scratch/00946/zzhang/data/openfold/cameo/mmcif_files # wrong loc for templates. I don't know why.
python3 run_pretrained_openfold.py \
    /work/09123/mjt2211/ls6/openfold-mt/cameo_dir/fasta_dir \
    /scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files \
    --output_dir ./ \
    --model_device "cuda:0" \
    --config_preset "model_4_ptm" \ #### is this right for template-less
    --openfold_checkpoint_path openfold/resources/openfold_params/finetuning_ptm_2.pt \ ### need to change to something templateless?
    --use_precomputed_alignments /scratch/09120/sk844/validation_set_cameo/alignments \
    --subtract_plddt \
    --skip_relaxation

