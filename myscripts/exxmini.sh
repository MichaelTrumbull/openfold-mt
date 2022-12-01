source lib/conda/etc/profile.d/conda.sh
conda activate openfold_venv

CUDA_VISIBLE_DEVICES=0 python colabrun.py