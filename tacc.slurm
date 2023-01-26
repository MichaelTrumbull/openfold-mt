#!/bin/bash

#SBATCH -J openfold-mt           # Job name
#SBATCH -o o.log       # Name of stdout output file #use %j for job number
#SBATCH -e e.log       # Name of stderr error file
#SBATCH -p gpu-a100          # Queue (partition) name
#SBATCH -N 1               # Total # of nodes
#SBATCH -n 1              # Total # of mpi tasks
#SBATCH -t 06:00:00        # Run time (hh:mm:ss)

###srun run-tacc-inf.sh

source ~/.bashrc
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
source /work/09123/mjt2211/ls6/openfold-mt/lib/conda/bin/activate
conda activate openfold_venv
which python
### scratch/00946/zzhang/data/openfold/cameo/mmcif_files # wrong loc for templates. I don't know why.
python3 run_pretrained_openfold.py \
    fasta_dir \
    /scratch/00946/zzhang/data/openfold/ls6-tacc/pdb_mmcif/mmcif_files \
    --output_dir ./ \
    --model_device "cuda:0" \
    --config_preset "model_1_ptm" \
    --openfold_checkpoint_path openfold/resources/openfold_params/finetuning_ptm_2.pt \
    --use_precomputed_alignments /scratch/00946/zzhang/data/openfold/cameo/alignments \
    --subtract_plddt
    

### https://www.rcsb.org/downloads/fasta    
### 6tf4, 7b1w, 7bew, 7dmf, 7e0m, 7f0h, 7kos, 7lnu, 7mrq, 7o62, 7plb, 7rdt, 7siq, 7tbs, 6wmk, 7b26, 7bhy, 7dms, 7e2v, 7f17, 7kqv, 7ls0, 7mrs, 7obm, 7plq, 7re4, 7sir, 7tcb, 6wqc,  7b28, 7bny, 7dnm, 7e37, 7f6e, 7ksp, 7lvz, 7msj, 7ofn, 7poh, 7re6, 7spo, 7v1v, 6xqj, 7b29, 7cjs, 7dnu, 7e3z, 7f8a, 7kua, 7lxc, 7mu9, 7og0, 7ppp, 7rg7, 7spp, 7v5y, 6z01, 7b2a, 7da9, 7don, 7ee3, 7fbh, 7kuw, 7m0q, 7ndr, 7ool, 7pq7, 7rrm, 7swh, 7v6p, 7ab3, 7b2o, 7dah, 7dq9, 7eea, 7fbp, 7kzh, 7m4m, 7nf9, 7ouq, 7pqf, 7rwk, 7swk, 7vdy, 7amc, 7b3a, 7dck, 7dqg, 7ef6, 7fe3, 7l19, 7m4n, 7nl4, 7p1b, 7prd, 7s02, 7sy9, 7vmu, 7ar0, 7b4q, 7dcm, 7dru, 7eft, 7fh3, 7l6j, 7mfi, 7nmb, 7p1v, 7q03, 7s13, 7t24, 7vnb, 7atr, 7b7t, 7dfe, 7dsu, 7eqx, 7fip, 7l6y, 7mfw, 7nwz, 7p3b, 7q1b, 7s13, 7t5f, 7vpf, 7au7, 7bbz, 7djy, 7dtp, 7eqx, 7kdx, 7lbu, 7mpz, 7nx0, 7p3t, 7q47, 7s6g, 7t71, 7w83, 7awk, 7bcb, 7dk9, 7dut, 7esx, 7kgc, 7lc5, 7mqy, 7nx0, 7p82, 7r84, 7s94, 7t8o, 7wbr, 7b0d, 7bcj, 7dkk, 7duv, 7et8, 7kik, 7lew, 7mrk, 7ny6, 7pbk, 7rbw, 7sf6, 7t9w, 7b1k, 7bcz, 7dko, 7dvn, 7ezg, 7kiu, 7ljh, 7mro, 7o49, 7pkx, 7rds, 7sh3, 7tav
