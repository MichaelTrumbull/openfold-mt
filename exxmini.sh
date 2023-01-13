source lib/conda/etc/profile.d/conda.sh
conda activate openfold_venv

# Need:
# pip install ipywidgets
# pip install py3Dmol

g=0
p="2DRI"

# target run
CUDA_VISIBLE_DEVICES=$g python3 colabrun.py --protien_name $p --variation_mode none --value 0 --representation z
r="z"
m="noise"

for v in 1 3 6 9 10 13 15 17 20 25
do
    CUDA_VISIBLE_DEVICES=$g python3 colabrun.py --protien_name $p --variation_mode $m --value $v --representation $r
done

echo DONE

# NOTES
# 4JA4: value >= 25 throws error "ValueError: The number of positions must match the number of atoms"