source lib/conda/etc/profile.d/conda.sh
conda activate openfold_venv

# Need:
# pip install ipywidgets
# pip install py3Dmol

g=1
p="6N3I"

# target run
#CUDA_VISIBLE_DEVICES=$g python3 colabrun.py --protien_name $p --variation_mode none --value 1 --representation s
r="z"
m="noise"

for v in 5 10 15 20 25
do
    CUDA_VISIBLE_DEVICES=$g python3 colabrun.py --protien_name $p --variation_mode $m --value $v --representation $r
done

echo DONE

# NOTES
# 4JA4: value >= 25 throws error "ValueError: The number of positions must match the number of atoms"