source lib/conda/etc/profile.d/conda.sh
conda activate openfold_venv

# Need:
# pip install ipywidgets
# pip install py3Dmol

g=0
p="1DGN"

# target run
CUDA_VISIBLE_DEVICES=$g python3 colabrun.py --protien_name $p --variation_mode none --value 1 --representation s

# s,z,m DOE
for v in 0.000001 0.00001 0.0001 0.001 0.01 0.1 10 100 1000 10000 100000
do
    for r in s z m
    do
        for m in mult noise
        do
            CUDA_VISIBLE_DEVICES=$g python3 colabrun.py --protien_name $p --variation_mode $m --value $v --representation $r
        done
    done
done

# zero out
CUDA_VISIBLE_DEVICES=$g python3 colabrun.py --protien_name $p --variation_mode zero --value 1 --representation s
CUDA_VISIBLE_DEVICES=$g python3 colabrun.py --protien_name $p --variation_mode zero --value 1 --representation z
CUDA_VISIBLE_DEVICES=$g python3 colabrun.py --protien_name $p --variation_mode zero --value 1 --representation m

# sz,sm,zm DOE
for v in 0.000001 0.00001 0.0001 0.001 0.01 0.1 10 100 1000 10000 100000
do
    for r in sz sm zm
    do
        for m in mult noise
        do
            CUDA_VISIBLE_DEVICES=$g python3 colabrun.py --protien_name $p --variation_mode $m --value $v --representation $r
        done
    done
done
