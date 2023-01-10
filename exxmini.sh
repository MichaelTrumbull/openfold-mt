source lib/conda/etc/profile.d/conda.sh
conda activate openfold_venv

# Need:
# pip install ipywidgets
# pip install py3Dmol

g=0
p="4JA4"

# target run
#CUDA_VISIBLE_DEVICES=$g python3 colabrun.py --protien_name $p --variation_mode none --value 1 --representation s
CUDA_VISIBLE_DEVICES=$g python3 colabrun.py --protien_name 4JA3 --variation_mode none --value 1 --representation s
CUDA_VISIBLE_DEVICES=$g python3 colabrun.py --protien_name 6N3I --variation_mode none --value 1 --representation s
r="z"
m="noise"

for v in 5 15 20 25 30 35 40 45 50
do
    CUDA_VISIBLE_DEVICES=$g python3 colabrun.py --protien_name $p --variation_mode $m --value $v --representation $r
done

echo DONE

# NOTES
# 4JA4: value >= 100 throws error "ValueError: The number of positions must match the number of atoms"