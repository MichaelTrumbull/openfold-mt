source lib/conda/etc/profile.d/conda.sh
conda activate openfold_venv

# Need:
# pip install ipywidgets
# pip install py3Dmol
CUDA_VISIBLE_DEVICES=0 python3 colabrun.py --protien_name 6t1z --representation z