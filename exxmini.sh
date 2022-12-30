source lib/conda/etc/profile.d/conda.sh
conda activate openfold_venv

# Need:
# pip install ipywidgets
# pip install py3Dmol
#CUDA_VISIBLE_DEVICES=0 python3 colabrun.py --protien_name 7MZZ --variation_mode none --value 1 --representation s

for v in 0.000001 0.00001 0.0001 0.001 0.01 0.1 10 100 1000 10000 100000
do
    for r in s z
    do
        CUDA_VISIBLE_DEVICES=0 python3 colabrun.py --protien_name 7MZZ --variation_mode mult --value $v --representation $r
    done
done

echo FINISHED FIRST FOR LOOP

CUDA_VISIBLE_DEVICES=0 python3 colabrun.py --protien_name 7MZZ --variation_mode zero --value 1 --representation s
CUDA_VISIBLE_DEVICES=0 python3 colabrun.py --protien_name 7MZZ --variation_mode zero --value 1 --representation z
CUDA_VISIBLE_DEVICES=0 python3 colabrun.py --protien_name 7MZZ --variation_mode zero --value 1 --representation m

for v in 0.000001 0.00001 0.0001 0.001 0.01 0.1 10 100 1000 10000 100000
do
    for r in s z m
    do
        CUDA_VISIBLE_DEVICES=0 python3 colabrun.py --protien_name 7MZZ --variation_mode noise --value $v --representation $r
    done
done
echo FULLY DONE