source lib/conda/etc/profile.d/conda.sh
conda activate openfold_venv

# Need:
# pip install ipywidgets
# pip install py3Dmol

g=0
p="4JA4"

# target run
CUDA_VISIBLE_DEVICES=$g python3 colabrun.py --protien_name $p --variation_mode none --value 1 --representation s

r="z"
m="noise"

for v in 10 100 1000 10000 100000 1000000 10000000
do
    CUDA_VISIBLE_DEVICES=$g python3 colabrun.py --protien_name $p --variation_mode $m --value $v --representation $r

done

echo DONE