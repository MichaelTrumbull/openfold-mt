# Latent space perturbation study

### data and plots
**Note**: tmscore is the metric I use. It always denotes the tmscore between the protein generated when it is unperturbed vs when it is perturbed. result = tmscore(regular OF run, perturbed run). The one notable exception is when I try to 'kick it' into a new conformational state. To see if the generated protein was trending towards the new conformation I used xray data for the new conformation in the tmscore.
- Runs located in openfold-mt/data/taccruns
  - contains 35 GB of data
  - not saved to github. use exxmini.
  - As long as this data is sitting there the notebooks for plotting should work just fine
- IPython notebooks located in openfold-mt/personal-nbs
  - Numbered as I created new ones for new studies or plots. Keeps it organized
  - 1compare.ipynb
    - Ignore these plots. These were all remaid in a better way in the following notebooks
  - 2perturb_representations.ipynb
    - This is where I studied if I could perturb the model (with a guassian noise) while it generated a protein such that it would "kick it" into a new conformation.
    - This study was done on ~3 proteins (with each having another conformation or two)
    - It is clear that as perturbation increases the tmscore decreases for the main conformation (expected result) and also decreases for each alternative conformation (unfortunate result). 
    - An possible next step would be to run the perturbation multiple times but only pay attention to when its prediction confidence is highest. So if in this random perturbations OF has high confidence in one of its predictions then maybe you could take that prediction and repeat. This could slowly guide it to a different shape that is still plausible.
    - I think this idea just doesn't work though.
  - 3cameo_data_s0.ipynb
    - The entire cameo data set was used here. I am no longer trying to get new conformations. I am just seeing how perturbations effect the final prediction
    - Two things are studied here
    - How different OF models and different perturbation locations effect the final result. 
      - You can see that, for an unknown reason, different proteins (in the cameo data set) will perturb differently depending on where you perturb and which model you use. 
    - Can any protein features explain why certain proteins get perturbed and when?
      - No. 
      - As you can see at the bottom of this notebook I found the R^2 value to relate every protein feature to how much it was perturbed (tmscore). The highest value is too small to matter. No relationship found
        - protein features include things like its physical size when folded, the number of beta sheets present, MSA depth, residue length, Pi helicies, proportional fractions of all of these, ect. 
      - I could not find any relationship that could explain why certain proteins were perturbable and others were not.
  - 4track_latent_space.ipynb
    - This the first experiment of the rest of my work. 
    - for each plot I am not comparing perturbed or unperturbed any more. Each plot will be either perturbed OR unperturbed but not both.
    - A single plot corresponds to a single protein
    - During a prediction run of a protein I save the latent space representation of s (single) at each iteration. There are 8 iterations in the structure module (I label this 'i') and 4 recycles per prediction (I label this 'r')
    - Each saved latent space (2D tensor) I will flatten it and then compare it to all other latent spaces cosine similarity
    - As you can see, the first iteration of every recycle is wildely dissimilar from the rest, but the rest are roughly the same to each other.
    - More interesting plots follow in the next notebooks
  - 5latent_space_and_s0_data.ipynb
    - Here I do the same thing but I also compare how perturbable (by zeroeing out the s representation) the protein is for each model. 
    - I wanted to see if there was any relationship between the propogation of the latent space in the heatmap and how perturbable it was. I don't remember seeing a clear relationship.
  - 6DOE_track_latent_space.ipynb
    - Here i start perturbing (s=0) at individual recycles and iterations. So You can see how the perturbation propogates (and how the latent space recovers). 
    - The first set of plots look weird. They have a new pattern to them. This is because I also *sometimes* saved the s latent space at two different locations in the structure module. I label these saves as location A and location B. This is before and after IPA.
    - In the second half of the plots (where they look like the previous notebooks) you can see more clearly how the perturbation will not propogate at all unless you perturb the s directly after it comes from the evoformer block or if you perturb s at the initial (0) iteration in that recycle. 
    - These plots show how resiliant s is to perturbation. It can regenerate s from the pair representation. 
  - 7DOE_LSpace_Z_pert.ipynb
    - Here I perturb Z, the pair representation inside of the structure module.
    - In my initial tests that inspired this investigation I saw that OF was wildly resilient to s being perturbed, but could be successfully perturbed if z was zeroed. This is why these plots seem to have much greater an effect.
    - You can also see i severed the IPA connection to see what sort of effect that would have
    - Note: L9 stands for line 9. This means I am saving the latent space at line 9 seen in algorithm 20 in the AlphaFold supplimentary materials. In the code you can have the heatmap include line 6 but I found that to be largely not useful.
  - 8DOE_LSpace_S_pert_every_r.ipynb
    - Here I perturb s as it comes from the evoformer block (at line 0 of algorithm 20), but now I will do it at said iteration every recycle. 
    - It looks here that the dissimilarity between latent space is not representative of how perturbed the final prediction is as seen by the tmscore listed in the title.
  - 9DOE_tmscore_heatmap.ipynb
    - This heatmap is NOT similarity between latent space. I am perturbing the latent space as before, but now I am showing how perturbed the final predicted protein structure is. I show this through the tmscore between the predicted protein without perturbation and the predicted protein with perturbation. The heat map is not necessary though so the regular plot next to it describes it better.

### If you want to run these experiments

If you want to expand on these experiments then maybe you will want to reprogram it all yourself since its easier to code than it is to understand someones code. 
- In the various openfold/model files I have flags that will read a text file in the system. They will then decide when/if to set 's' or 'z' to zero and at what time. 
  - This text file is generated by set_iteration_doe_file.py (my running design-of-experiments scripts utilize this. These scripts are in the struct_modu_only_experiment/ directory)
- After going through a prediction I also modified some code at the end to name the files according to my naming scheme. 
- Again I think it would be 100x easier if you just redid this yourself because my version is so very messy. All the code I added, essentially, was to check where in the prediction we were, see if we want to zero, zero, save the file, rename the file according to my experiment, then repeat.
- In the end, you only need one line to get some data (s=s*0) and the rest is just for automation.


### Environments
When I plotted these I used python version 10 and then installed a standard library of elements such as pickle, numpy, pytorch, and matplotlib and let the system take care of the version. There is nothing fancy in this code for plotting. 


### proteins
Potentially interesting proteins:
- 7mqy_A, 7tav_B, 7t9w_B, 7bcz_A, 6xqj_A, 7nf9_A, 7e2v_A, 7s13_L, 7bcb_B
- These are all from the cameo dataset, so Sachin has the alignments
- They are interesting in their variety of results I got from them. Nothing crazy happened to these proteins but I did find them to be ‘different’



# start original OF readme


# README from the original OpenFold repository


![header ](imgs/of_banner.png)
_Figure: Comparison of OpenFold and AlphaFold2 predictions to the experimental structure of PDB 7KDX, chain B._


# OpenFold

A faithful but trainable PyTorch reproduction of DeepMind's 
[AlphaFold 2](https://github.com/deepmind/alphafold).

## Features

OpenFold carefully reproduces (almost) all of the features of the original open
source inference code (v2.0.1). The sole exception is model ensembling, which 
fared poorly in DeepMind's own ablation testing and is being phased out in future
DeepMind experiments. It is omitted here for the sake of reducing clutter. In 
cases where the *Nature* paper differs from the source, we always defer to the 
latter.

OpenFold is trainable in full precision, half precision, or `bfloat16` with or without DeepSpeed, 
and we've trained it from scratch, matching the performance of the original. 
We've publicly released model weights and our training data &mdash; some 400,000 
MSAs and PDB70 template hit files &mdash; under a permissive license. Model weights 
are available via scripts in this repository while the MSAs are hosted by the 
[Registry of Open Data on AWS (RODA)](https://registry.opendata.aws/openfold). 
Try out running inference for yourself with our [Colab notebook](https://colab.research.google.com/github/aqlaboratory/openfold/blob/main/notebooks/OpenFold.ipynb).

OpenFold also supports inference using AlphaFold's official parameters, and 
vice versa (see `scripts/convert_of_weights_to_jax.py`).

OpenFold has the following advantages over the reference implementation:

- **Faster inference** on GPU, sometimes by as much as 2x. The greatest speedups are achieved on (>= Ampere) GPUs.
- **Inference on extremely long chains**, made possible by our implementation of low-memory attention 
([Rabe & Staats 2021](https://arxiv.org/pdf/2112.05682.pdf)). OpenFold can predict the structures of
  sequences with more than 4000 residues on a single A100, and even longer ones with CPU offloading.
- **Custom CUDA attention kernels** modified from [FastFold](https://github.com/hpcaitech/FastFold)'s 
kernels support in-place attention during inference and training. They use 
4x and 5x less GPU memory than equivalent FastFold and stock PyTorch 
implementations, respectively.
- **Efficient alignment scripts** using the original AlphaFold HHblits/JackHMMER pipeline or [ColabFold](https://github.com/sokrypton/ColabFold)'s, which uses the faster MMseqs2 instead. We've used them to generate millions of alignments.
- **FlashAttention** support greatly speeds up MSA attention.

## Installation (Linux)

All Python dependencies are specified in `environment.yml`. For producing sequence 
alignments, you'll also need `kalign`, the [HH-suite](https://github.com/soedinglab/hh-suite), 
and one of {`jackhmmer`, [MMseqs2](https://github.com/soedinglab/mmseqs2) (nightly build)} 
installed on on your system. You'll need `git-lfs` to download OpenFold parameters. 
Finally, some download scripts require `aria2c` and `aws`.

For convenience, we provide a script that installs Miniconda locally, creates a 
`conda` virtual environment, installs all Python dependencies, and downloads
useful resources, including both sets of model parameters. Run:

```bash
scripts/install_third_party_dependencies.sh
```

To activate the environment, run:

```bash
source scripts/activate_conda_env.sh
```

To deactivate it, run:

```bash
source scripts/deactivate_conda_env.sh
```

With the environment active, compile OpenFold's CUDA kernels with

```bash
python3 setup.py install
```

To install the HH-suite to `/usr/bin`, run

```bash
# scripts/install_hh_suite.sh
```

## Usage

If you intend to generate your own alignments, e.g. for inference, you have two 
choices for downloading protein databases, depending on whether you want to use 
DeepMind's MSA generation pipeline (w/ HMMR & HHblits) or 
[ColabFold](https://github.com/sokrypton/ColabFold)'s, which uses the faster
MMseqs2 instead. For the former, run:

```bash
bash scripts/download_alphafold_dbs.sh data/
```

For the latter, run:

```bash
bash scripts/download_mmseqs_dbs.sh data/    # downloads .tar files
bash scripts/prep_mmseqs_dbs.sh data/        # unpacks and preps the databases
```

Make sure to run the latter command on the machine that will be used for MSA
generation (the script estimates how the precomputed database index used by
MMseqs2 should be split according to the memory available on the system).

If you're using your own precomputed MSAs or MSAs from the RODA repository, 
there's no need to download these alignment databases. Simply make sure that
the `alignment_dir` contains one directory per chain and that each of these
contains alignments (.sto, .a3m, and .hhr) corresponding to that chain. You
can use `scripts/flatten_roda.sh` to reformat RODA downloads in this way.
Note that the RODA alignments are NOT compatible with the recent .cif ground
truth files downloaded by `scripts/download_alphafold_dbs.sh`. To fetch .cif 
files that match the RODA MSAs, once the alignments are flattened, use 
`scripts/download_roda_pdbs.sh`. That script outputs a list of alignment dirs 
for which matching .cif files could not be found. These should be removed from 
the alignment directory.

Alternatively, you can use raw MSAs from 
[ProteinNet](https://github.com/aqlaboratory/proteinnet). After downloading
that database, use `scripts/prep_proteinnet_msas.py` to convert the data 
into a format recognized by the OpenFold parser. The resulting directory 
becomes the `alignment_dir` used in subsequent steps. Use 
`scripts/unpack_proteinnet.py` to extract `.core` files from ProteinNet text 
files.

For both inference and training, the model's hyperparameters can be tuned from
`openfold/config.py`. Of course, if you plan to perform inference using 
DeepMind's pretrained parameters, you will only be able to make changes that
do not affect the shapes of model parameters. For an example of initializing
the model, consult `run_pretrained_openfold.py`.

### Inference

To run inference on a sequence or multiple sequences using a set of DeepMind's 
pretrained parameters, run e.g.:

```bash
python3 run_pretrained_openfold.py \
    fasta_dir \
    data/pdb_mmcif/mmcif_files/ \
    --uniref90_database_path data/uniref90/uniref90.fasta \
    --mgnify_database_path data/mgnify/mgy_clusters_2018_12.fa \
    --pdb70_database_path data/pdb70/pdb70 \
    --uniclust30_database_path data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --output_dir ./ \
    --bfd_database_path data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --model_device "cuda:0" \
    --jackhmmer_binary_path lib/conda/envs/openfold_venv/bin/jackhmmer \
    --hhblits_binary_path lib/conda/envs/openfold_venv/bin/hhblits \
    --hhsearch_binary_path lib/conda/envs/openfold_venv/bin/hhsearch \
    --kalign_binary_path lib/conda/envs/openfold_venv/bin/kalign \
    --config_preset "model_1_ptm" \
    --openfold_checkpoint_path openfold/resources/openfold_params/finetuning_ptm_2.pt
```

where `data` is the same directory as in the previous step. If `jackhmmer`, 
`hhblits`, `hhsearch` and `kalign` are available at the default path of 
`/usr/bin`, their `binary_path` command-line arguments can be dropped.
If you've already computed alignments for the query, you have the option to 
skip the expensive alignment computation here with 
`--use_precomputed_alignments`.

`--openfold_checkpoint_path` or `--jax_param_path` accept comma-delineated lists
of .pt/DeepSpeed OpenFold checkpoints and AlphaFold's .npz JAX parameter files, 
respectively. For a breakdown of the differences between the different parameter 
files, see the README downloaded to `openfold/resources/openfold_params/`. Since 
OpenFold was trained under a newer training schedule than the one from which the 
`model_n` config presets are derived, there is no clean correspondence between 
`config_preset` settings and OpenFold checkpoints; the only restraints are that 
`*_ptm` checkpoints must be run with `*_ptm` config presets and that `_no_templ_`
checkpoints are only compatible with template-less presets (`model_3` and above).

Note that chunking (as defined in section 1.11.8 of the AlphaFold 2 supplement)
is enabled by default in inference mode. To disable it, set `globals.chunk_size`
to `None` in the config. If a value is specified, OpenFold will attempt to 
dynamically tune it, considering the chunk size specified in the config as a 
minimum. This tuning process automatically ensures consistently fast runtimes 
regardless of input sequence length, but it also introduces some runtime 
variability, which may be undesirable for certain users. It is also recommended
to disable this feature for very long chains (see below). To do so, set the 
`tune_chunk_size` option in the config to `False`.

For large-scale batch inference, we offer an optional tracing mode, which
massively improves runtimes at the cost of a lengthy model compilation process.
To enable it, add `--trace_model` to the inference command.

To get a speedup during inference, enable [FlashAttention](https://github.com/HazyResearch/flash-attention)
in the config. Note that it appears to work best for sequences with < 1000 residues.

Input FASTA files containing multiple sequences are treated as complexes. In
this case, the inference script runs AlphaFold-Gap, a hack proposed
[here](https://twitter.com/minkbaek/status/1417538291709071362?lang=en), using
the specified stock AlphaFold/OpenFold parameters (NOT AlphaFold-Multimer). To
run inference with AlphaFold-Multimer, use the (experimental) `multimer` branch 
instead.

To minimize memory usage during inference on long sequences, consider the
following changes:

- As noted in the AlphaFold-Multimer paper, the AlphaFold/OpenFold template
stack is a major memory bottleneck for inference on long sequences. OpenFold
supports two mutually exclusive inference modes to address this issue. One,
`average_templates` in the `template` section of the config, is similar to the
solution offered by AlphaFold-Multimer, which is simply to average individual
template representations. Our version is modified slightly to accommodate 
weights trained using the standard template algorithm. Using said weights, we
notice no significant difference in performance between our averaged template 
embeddings and the standard ones. The second, `offload_templates`, temporarily 
offloads individual template embeddings into CPU memory. The former is an 
approximation while the latter is slightly slower; both are memory-efficient 
and allow the model to utilize arbitrarily many templates across sequence 
lengths. Both are disabled by default, and it is up to the user to determine 
which best suits their needs, if either.
- Inference-time low-memory attention (LMA) can be enabled in the model config.
This setting trades off speed for vastly improved memory usage. By default,
LMA is run with query and key chunk sizes of 1024 and 4096, respectively.
These represent a favorable tradeoff in most memory-constrained cases.
Powerusers can choose to tweak these settings in 
`openfold/model/primitives.py`. For more information on the LMA algorithm,
see the aforementioned Staats & Rabe preprint.
- Disable `tune_chunk_size` for long sequences. Past a certain point, it only
wastes time.
- As a last resort, consider enabling `offload_inference`. This enables more
extensive CPU offloading at various bottlenecks throughout the model.
- Disable FlashAttention, which seems unstable on long sequences.

Using the most conservative settings, we were able to run inference on a 
4600-residue complex with a single A100. Compared to AlphaFold's own memory 
offloading mode, ours is considerably faster; the same complex takes the more 
efficent AlphaFold-Multimer more than double the time. Use the
`long_sequence_inference` config option to enable all of these interventions
at once. The `run_pretrained_openfold.py` script can enable this config option with the 
`--long_sequence_inference` command line option

### Training

To train the model, you will first need to precompute protein alignments. 

You have two options. You can use the same procedure DeepMind used by running
the following:

```bash
python3 scripts/precompute_alignments.py mmcif_dir/ alignment_dir/ \
    --uniref90_database_path data/uniref90/uniref90.fasta \
    --mgnify_database_path data/mgnify/mgy_clusters_2018_12.fa \
    --pdb70_database_path data/pdb70/pdb70 \
    --uniclust30_database_path data/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
    --bfd_database_path data/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
    --cpus_per_task 16 \
    --jackhmmer_binary_path lib/conda/envs/openfold_venv/bin/jackhmmer \
    --hhblits_binary_path lib/conda/envs/openfold_venv/bin/hhblits \
    --hhsearch_binary_path lib/conda/envs/openfold_venv/bin/hhsearch \
    --kalign_binary_path lib/conda/envs/openfold_venv/bin/kalign
```

As noted before, you can skip the `binary_path` arguments if these binaries are 
at `/usr/bin`. Expect this step to take a very long time, even for small 
numbers of proteins.

Alternatively, you can generate MSAs with the ColabFold pipeline (and templates
with HHsearch) with:

```bash
python3 scripts/precompute_alignments_mmseqs.py input.fasta \
    data/mmseqs_dbs \
    uniref30_2103_db \
    alignment_dir \
    ~/MMseqs2/build/bin/mmseqs \
    /usr/bin/hhsearch \
    --env_db colabfold_envdb_202108_db
    --pdb70 data/pdb70/pdb70
```

where `input.fasta` is a FASTA file containing one or more query sequences. To 
generate an input FASTA from a directory of mmCIF and/or ProteinNet .core 
files, we provide `scripts/data_dir_to_fasta.py`.

Next, generate a cache of certain datapoints in the template mmCIF files:

```bash
python3 scripts/generate_mmcif_cache.py \
    mmcif_dir/ \
    mmcif_cache.json \
    --no_workers 16
```

This cache is used to pre-filter templates. 

Next, generate a separate chain-level cache with data used for training-time 
data filtering:

```bash
python3 scripts/generate_chain_data_cache.py \
    mmcif_dir/ \
    chain_data_cache.json \
    --cluster_file clusters-by-entity-40.txt \
    --no_workers 16
```

where the `cluster_file` argument is a file of chain clusters, one cluster
per line (e.g. [PDB40](https://cdn.rcsb.org/resources/sequence/clusters/clusters-by-entity-40.txt)).

Optionally, download an AlphaFold-style validation set from 
[CAMEO](https://cameo3d.org) using `scripts/download_cameo.py`. Use the 
resulting FASTA files to generate validation alignments and then specify 
the validation set's location using the `--val_...` family of training script 
flags.

Finally, call the training script:

```bash
python3 train_openfold.py mmcif_dir/ alignment_dir/ template_mmcif_dir/ output_dir/ \
    2021-10-10 \ 
    --template_release_dates_cache_path mmcif_cache.json \ 
    --precision bf16 \
    --gpus 8 --replace_sampler_ddp=True \
    --seed 4242022 \ # in multi-gpu settings, the seed must be specified
    --deepspeed_config_path deepspeed_config.json \
    --checkpoint_every_epoch \
    --resume_from_ckpt ckpt_dir/ \
    --train_chain_data_cache_path chain_data_cache.json \
    --obsolete_pdbs_file_path obsolete.dat
```

where `--template_release_dates_cache_path` is a path to the mmCIF cache. 
Note that `template_mmcif_dir` can be the same as `mmcif_dir` which contains
training targets. A suitable DeepSpeed configuration file can be generated with 
`scripts/build_deepspeed_config.py`. The training script is 
written with [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) 
and supports the full range of training options that entails, including 
multi-node distributed training, validation, and so on. For more information, 
consult PyTorch Lightning documentation and the `--help` flag of the training 
script.

Note that, despite its variable name, `mmcif_dir` can also contain PDB files 
or even ProteinNet .core files. 

To emulate the AlphaFold training procedure, which uses a self-distillation set 
subject to special preprocessing steps, use the family of `--distillation` flags.

In cases where it may be burdensome to create separate files for each chain's
alignments, alignment directories can be consolidated using the scripts in 
`scripts/alignment_db_scripts/`. First, run `create_alignment_db.py` to
consolidate an alignment directory into a pair of database and index files.
Once all alignment directories (or shards of a single alignment directory)
have been compiled, unify the indices with `unify_alignment_db_indices.py`. The
resulting index, `super.index`, can be passed to the training script flags
containing the phrase `alignment_index`. In this scenario, the `alignment_dir`
flags instead represent the directory containing the compiled alignment
databases. Both the training and distillation datasets can be compiled in this
way. Anecdotally, this can speed up training in I/O-bottlenecked environments.

## Testing

To run unit tests, use

```bash
scripts/run_unit_tests.sh
```

The script is a thin wrapper around Python's `unittest` suite, and recognizes
`unittest` arguments. E.g., to run a specific test verbosely:

```bash
scripts/run_unit_tests.sh -v tests.test_model
```

Certain tests require that AlphaFold (v2.0.1) be installed in the same Python
environment. These run components of AlphaFold and OpenFold side by side and
ensure that output activations are adequately similar. For most modules, we
target a maximum pointwise difference of `1e-4`.

## Building and using the docker container

### Building the docker image

Openfold can be built as a docker container using the included dockerfile. To build it, run the following command from the root of this repository:

```bash
docker build -t openfold .
```

### Running the docker container 

The built container contains both `run_pretrained_openfold.py` and `train_openfold.py` as well as all necessary software dependencies. It does not contain the model parameters, sequence, or structural databases. These should be downloaded to the host machine following the instructions in the Usage section above. 

The docker container installs all conda components to the base conda environment in `/opt/conda`, and installs openfold itself in `/opt/openfold`,

Before running the docker container, you can verify that your docker installation is able to properly communicate with your GPU by running the following command:


```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

Note the `--gpus all` option passed to `docker run`. This option is necessary in order for the container to use the GPUs on the host machine.

To run the inference code under docker, you can use a command like the one below.  In this example, parameters and sequences from the alphafold dataset are being used and are located at `/mnt/alphafold_database` on the host machine, and the input files are located in the current working directory. You can adjust the volume mount locations as needed to reflect the locations of your data. 

```bash
docker run \
--gpus all \
-v $PWD/:/data \
-v /mnt/alphafold_database/:/database \
-ti openfold:latest \
python3 /opt/openfold/run_pretrained_openfold.py \
/data/fasta_dir \
/database/pdb_mmcif/mmcif_files/ \
--uniref90_database_path /database/uniref90/uniref90.fasta \
--mgnify_database_path /database/mgnify/mgy_clusters_2018_12.fa \
--pdb70_database_path /database/pdb70/pdb70 \
--uniclust30_database_path /database/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
--output_dir /data \
--bfd_database_path /database/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
--model_device cuda:0 \
--jackhmmer_binary_path /opt/conda/bin/jackhmmer \
--hhblits_binary_path /opt/conda/bin/hhblits \
--hhsearch_binary_path /opt/conda/bin/hhsearch \
--kalign_binary_path /opt/conda/bin/kalign \
--openfold_checkpoint_path /database/openfold_params/finetuning_ptm_2.pt
```

## Copyright notice

While AlphaFold's and, by extension, OpenFold's source code is licensed under
the permissive Apache Licence, Version 2.0, DeepMind's pretrained parameters 
fall under the CC BY 4.0 license, a copy of which is downloaded to 
`openfold/resources/params` by the installation script. Note that the latter
replaces the original, more restrictive CC BY-NC 4.0 license as of January 2022.

## Contributing

If you encounter problems using OpenFold, feel free to create an issue! We also
welcome pull requests from the community.

## Citing this work

Please cite our paper:

```bibtex
@article {Ahdritz2022.11.20.517210,
	author = {Ahdritz, Gustaf and Bouatta, Nazim and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O{\textquoteright}Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccolò and Zhang, Bo and Nowaczynski, Arkadiusz and Wang, Bei and Stepniewska-Dziubinska, Marta M and Zhang, Shang and Ojewole, Adegoke and Guney, Murat Efe and Biderman, Stella and Watkins, Andrew M and Ra, Stephen and Lorenzo, Pablo Ribalta and Nivon, Lucas and Weitzner, Brian and Ban, Yih-En Andrew and Sorger, Peter K and Mostaque, Emad and Zhang, Zhao and Bonneau, Richard and AlQuraishi, Mohammed},
	title = {OpenFold: Retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization},
	elocation-id = {2022.11.20.517210},
	year = {2022},
	doi = {10.1101/2022.11.20.517210},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {AlphaFold2 revolutionized structural biology with the ability to predict protein structures with exceptionally high accuracy. Its implementation, however, lacks the code and data required to train new models. These are necessary to (i) tackle new tasks, like protein-ligand complex structure prediction, (ii) investigate the process by which the model learns, which remains poorly understood, and (iii) assess the model{\textquoteright}s generalization capacity to unseen regions of fold space. Here we report OpenFold, a fast, memory-efficient, and trainable implementation of AlphaFold2, and OpenProteinSet, the largest public database of protein multiple sequence alignments. We use OpenProteinSet to train OpenFold from scratch, fully matching the accuracy of AlphaFold2. Having established parity, we assess OpenFold{\textquoteright}s capacity to generalize across fold space by retraining it using carefully designed datasets. We find that OpenFold is remarkably robust at generalizing despite extreme reductions in training set size and diversity, including near-complete elisions of classes of secondary structure elements. By analyzing intermediate structures produced by OpenFold during training, we also gain surprising insights into the manner in which the model learns to fold proteins, discovering that spatial dimensions are learned sequentially. Taken together, our studies demonstrate the power and utility of OpenFold, which we believe will prove to be a crucial new resource for the protein modeling community.},
	URL = {https://www.biorxiv.org/content/10.1101/2022.11.20.517210},
	eprint = {https://www.biorxiv.org/content/early/2022/11/22/2022.11.20.517210.full.pdf},
	journal = {bioRxiv}
}
```

Any work that cites OpenFold should also cite AlphaFold.
