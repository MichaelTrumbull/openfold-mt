import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--protien_name', type=str, default="6t1z")
parser.add_argument('--vari', type=float, default=10)
parser.add_argument('--representation', type=str, default='s')
args = parser.parse_args()



#@markdown ### Enter the amino acid sequence to fold ⬇️
#pdbname = '4JA4' #@param {type:"string"}
pdbname = args.protien_name
if pdbname == '6t1z':
  sequence = 'GKEFWNLDKNLQLRLGIVFLGAFSYGTVFSSMTIYYNQYLGSAITGILLALSAVATFVAGILAGFFADRNGRKPVMVFGTIIQLLGAALAIASNLPGHVNPWSTFIAFLLISFGYNFVITAGNAMIIDASNAENRKVVFMLDYWAQNLSVILGAALGAWLFRPAFEALLVILLLTVLVSFFLTTFVMTETFKPTVKVDNIFQAYKTVLQDKTYMIFMGANIATTFIIMQFDNFLPVHLSNSFKTITFWGFEIYGQRMLTIYLILACVLVVLLMTTLNRLTKDWSHQKGFIWGSLFMAIGMIFSFLTTTFTPIFIAGIVYTLGEIVYTPSVQTLGADLMNPEKIGSYNGVAAIKMPIASILAGLLVSISPMIKAIGVSLVLALTEVLAIILVLVAVNRHQKTKLNLEVLFQG'
if pdbname == '4JA4':# NOT ENOUGH MEM
  sequence = 'GNTQYNSSYIFSITLVATLGGLLFGYDTAVISGTVESLNTVFVAPQNLSESAANSLLGFCVASALIGCIIGGALGGYCSNRFGRRDSLKIAAVLFFISGVGSAWPELGFTSINPDNTVPVYLAGYVPEFVIYRIIGGIGVGLASMLSPMYIAELAPAHIRGKLVSFNQFAIIFGQLLVYCVNYFIARSGDASWLNTDGWRYMFASECIPALLFLMLLYTVPESPRWLMSRGKQEQAEGILRKIMGNTLATQAVQEIKHSLDHGRKTGGRLLMFGVGVIVIGVMLSIFQQFVGINVVLYYAPEVFKTLGASTDIALLQTIIVGVINLTFTVLAIMTVDKFGRKPLQIIGALGMAIGMFSLGTAFYTQAPGIVALLSMLFYVAAFAMSWGPVCWVLLSEIFPNAIRGKALAIAVAAQWLANYFVSWTFPMMDKNSWLVAHFHNGFSYWIYGCMGVLAALFMWKFVPETKGKTLEELEALWEPETKKT'

###########################################################

NAME_DETAILS = 'z_vari_100_' + pdbname + '_'

folderpath = r'myscripts/MSAs/' #location of .dbs file that contains the msa

output_dir = 'output/'



#from IPython.utils import io
import os
import subprocess
import tqdm.notebook

import pickle
dbspath = folderpath + pdbname + ".dbs"
run_jkhmmer = not os.path.exists(dbspath)

#import unittest.mock
import sys

# Allows us to skip installing these packages
unnecessary_modules = [
  "dllogger",
  "pytorch_lightning",
  "pytorch_lightning.utilities",
  "pytorch_lightning.callbacks.early_stopping",
  "pytorch_lightning.utilities.seed",
]
#for unnecessary_module in unnecessary_modules:
#  sys.modules[unnecessary_module] = unittest.mock.MagicMock()
import os
#from urllib import request
from concurrent import futures
#from google.colab import files
import json
#from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import py3Dmol
import torch

# A filthy hack to avoid slow Linear layer initialization
#import openfold.model.primitives
def __default_linear_init__(self, *args, **kwargs):
    return torch.nn.Linear.__init__(
      self, 
      *args[:2], 
      **{k:v for k,v in kwargs.items() if k == "bias"}
    )

#openfold.model.primitives.Linear.__init__ = __default_linear_init__
#print(os.getcwd())

from openfold import config
from openfold.data import feature_pipeline
from openfold.data import parsers
from openfold.data import data_pipeline
from openfold.data.tools import jackhmmer
from openfold.model import model
from openfold.np import protein
from openfold.utils.import_weights import import_jax_weights_
from openfold.utils.tensor_utils import tensor_tree_map

#from IPython import display
#from ipywidgets import GridspecLayout
#from ipywidgets import Output



##############################################################################################################
#  BLOCK  BEGIN
##############################################################################################################


weight_set = 'OpenFold' #@param ["OpenFold", "AlphaFold"]
relax_prediction = True #@param {type:"boolean"}

if(relax_prediction):
  from openfold.np.relax import relax
  from openfold.np.relax import utils

# Remove all whitespaces, tabs and end lines; upper-case
sequence = sequence.translate(str.maketrans('', '', ' \n\t')).upper()
aatypes = set('ACDEFGHIKLMNPQRSTVWY')  # 20 standard aatypes
if not set(sequence).issubset(aatypes):
  raise Exception(f'Input sequence contains non-amino acid letters: {set(sequence) - aatypes}. OpenFold only supports 20 standard amino acids as inputs.')


##############################################################################################################
#  BLOCK  INSTALL THIRD PARTY SOFTWARE
##############################################################################################################
# note: most has been deleted



TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'


##############################################################################################################
# BLOCK INSTALL OPENFOLD
##############################################################################################################
# note: OF preinstalled. Just using the paths in this block

# Define constants
#GIT_REPO='https://github.com/aqlaboratory/openfold'
GIT_REPO='https://github.com/MichaelTrumbull/openfold-mt'####### NEW CODE
ALPHAFOLD_PARAM_SOURCE_URL = 'https://storage.googleapis.com/alphafold/alphafold_params_2022-01-19.tar'
OPENFOLD_PARAMS_DIR = 'openfold/resources/openfold_params'
ALPHAFOLD_PARAMS_DIR = 'openfold/resources/params'
ALPHAFOLD_PARAMS_PATH = os.path.join(
  ALPHAFOLD_PARAMS_DIR, os.path.basename(ALPHAFOLD_PARAM_SOURCE_URL)
)

##############################################################################################################
# BLOCK IMPORT PYTHON PACKAGES
##############################################################################################################

# moved up..

##############################################################################################################
# BLOCK PICKLE
##############################################################################################################

#@title pickle
# pickle is used to save the dbs file generated by jkhmmer
# moved up...

##############################################################################################################
# BLOCK SEARCH AGAINST GENETIC DATABASE
##############################################################################################################

## Removing any jackhammer done
print('Skipping JKHMMER. Loading msa dbs instead.')
# need this pbar code or things break...
with tqdm.notebook.tqdm(total=1, bar_format=TQDM_BAR_FORMAT) as pbar:
  pbar.set_description('Searching mgnify')
with open(dbspath, "rb") as f:
  dbs = pickle.load(f)

# --- Extract the MSAs and visualize ---
# Extract the MSAs from the Stockholm files.
# NB: deduplication happens later in data_pipeline.make_msa_features.

mgnify_max_hits = 501

msas = []
deletion_matrices = []
full_msa = []
for db_name, db_results in dbs:
  unsorted_results = []
  for i, result in enumerate(db_results):
    msa, deletion_matrix, target_names = parsers.parse_stockholm(result['sto'])
    e_values_dict = parsers.parse_e_values_from_tblout(result['tbl'])
    e_values = [e_values_dict[t.split('/')[0]] for t in target_names]
    zipped_results = zip(msa, deletion_matrix, target_names, e_values)
    if i != 0:
      # Only take query from the first chunk
      zipped_results = [x for x in zipped_results if x[2] != 'query']
    unsorted_results.extend(zipped_results)
  sorted_by_evalue = sorted(unsorted_results, key=lambda x: x[3])
  db_msas, db_deletion_matrices, _, _ = zip(*sorted_by_evalue)
  if db_msas:
    if db_name == 'mgnify':
      db_msas = db_msas[:mgnify_max_hits]
      db_deletion_matrices = db_deletion_matrices[:mgnify_max_hits]
    full_msa.extend(db_msas)
    msas.append(db_msas)
    deletion_matrices.append(db_deletion_matrices)
    msa_size = len(set(db_msas))
    print(f'{msa_size} Sequences Found in {db_name}')

deduped_full_msa = list(dict.fromkeys(full_msa))
total_msa_size = len(deduped_full_msa)
print(f'\n{total_msa_size} Sequences Found in Total\n')

aa_map = {restype: i for i, restype in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ-')}
msa_arr = np.array([[aa_map[aa] for aa in seq] for seq in deduped_full_msa])
num_alignments, num_res = msa_arr.shape

#fig = plt.figure(figsize=(12, 3))
#plt.title('Per-Residue Count of Non-Gap Amino Acids in the MSA')
#plt.plot(np.sum(msa_arr != aa_map['-'], axis=0), color='black')
#plt.ylabel('Non-Gap Count')
#plt.yticks(range(0, num_alignments + 1, max(1, int(num_alignments / 3))))
#plt.show()

##############################################################################################################
# BLOCK RUN OPENFOLD AND DOWNLOAD PREDICTION
##############################################################################################################

# Color bands for visualizing plddt
PLDDT_BANDS = [
  (0, 50, '#FF7D45'),
  (50, 70, '#FFDB13'),
  (70, 90, '#65CBF3'),
  (90, 100, '#0053D6')
]

# --- Run the model ---
model_names = [ 
  'finetuning_3.pt', 
  'finetuning_4.pt', 
  'finetuning_5.pt', 
  'finetuning_ptm_2.pt',
  'finetuning_no_templ_ptm_1.pt'
]

def _placeholder_template_feats(num_templates_, num_res_):
  return {
      'template_aatype': np.zeros((num_templates_, num_res_, 22), dtype=np.int64),
      'template_all_atom_positions': np.zeros((num_templates_, num_res_, 37, 3), dtype=np.float32),
      'template_all_atom_mask': np.zeros((num_templates_, num_res_, 37), dtype=np.float32),
      'template_domain_names': np.zeros((num_templates_,), dtype=np.float32),
      'template_sum_probs': np.zeros((num_templates_, 1), dtype=np.float32),
  }

plddts = {}
pae_outputs = {}
unrelaxed_proteins = {}

with tqdm.notebook.tqdm(total=len(model_names) + 1, bar_format=TQDM_BAR_FORMAT) as pbar:
  for i, model_name in list(enumerate(model_names)):
    pbar.set_description(f'Running {model_name}')
    num_templates = 1 # dummy number --- is ignored
    num_res = len(sequence)
    
    feature_dict = {}
    feature_dict.update(data_pipeline.make_sequence_features(sequence, 'test', num_res))
    feature_dict.update(data_pipeline.make_msa_features(msas, deletion_matrices=deletion_matrices))
    feature_dict.update(_placeholder_template_feats(num_templates, num_res))

    if(weight_set == "AlphaFold"):
      config_preset = f"model_{i}"
    else:
      if("_no_templ_" in model_name):
        config_preset = "model_3"
      else:
        config_preset = "model_1"
      if("_ptm_" in model_name):
        config_preset += "_ptm"

    cfg = config.model_config(config_preset)
    openfold_model = model.AlphaFold(cfg)
    openfold_model = openfold_model.eval()
    if(weight_set == "AlphaFold"):
      params_name = os.path.join(
        ALPHAFOLD_PARAMS_DIR, f"params_{config_preset}.npz"
      )
      import_jax_weights_(openfold_model, params_name, version=config_preset)
    elif(weight_set == "OpenFold"):
      params_name = os.path.join(
        OPENFOLD_PARAMS_DIR,
        model_name,
      )
      d = torch.load(params_name)
      openfold_model.load_state_dict(d)
    else:
      raise ValueError(f"Invalid weight set: {weight_set}")

    openfold_model = openfold_model.cuda()

    pipeline = feature_pipeline.FeaturePipeline(cfg.data)
    processed_feature_dict = pipeline.process_features(
      feature_dict, mode='predict'
    )

    processed_feature_dict = tensor_tree_map(
        lambda t: t.cuda(), processed_feature_dict
    )

    with torch.no_grad():
      prediction_result = openfold_model(processed_feature_dict)

    # Move the batch and output to np for further processing
    processed_feature_dict = tensor_tree_map(
      lambda t: np.array(t[..., -1].cpu()), processed_feature_dict
    )
    prediction_result = tensor_tree_map(
      lambda t: np.array(t.cpu()), prediction_result
    )

    mean_plddt = prediction_result['plddt'].mean()

    if 'predicted_aligned_error' in prediction_result:
      pae_outputs[model_name] = (
          prediction_result['predicted_aligned_error'],
          prediction_result['max_predicted_aligned_error']
      )
    else:
      # Get the pLDDT confidence metrics. Do not put pTM models here as they
      # should never get selected.
      plddts[model_name] = prediction_result['plddt']

    # Set the b-factors to the per-residue plddt.
    final_atom_mask = prediction_result['final_atom_mask']
    b_factors = prediction_result['plddt'][:, None] * final_atom_mask
    unrelaxed_protein = protein.from_prediction(
      processed_feature_dict, prediction_result, b_factors=b_factors
    )
    unrelaxed_proteins[model_name] = unrelaxed_protein

    # Delete unused outputs to save memory.
    del openfold_model
    del processed_feature_dict
    del prediction_result
    pbar.update(n=1)

  # Find the best model according to the mean pLDDT.
  best_model_name = max(plddts.keys(), key=lambda x: plddts[x].mean())
  best_pdb = protein.to_pdb(unrelaxed_proteins[best_model_name])

  # --- AMBER relax the best model ---
  if(relax_prediction):
    pbar.set_description(f'AMBER relaxation')
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=0,
        tolerance=2.39,
        stiffness=10.0,
        exclude_residues=[],
        max_outer_iterations=20,
        use_gpu=False,
    )
    relaxed_pdb, _, _ = amber_relaxer.process(
        prot=unrelaxed_proteins[best_model_name]
    )

    # Write out the prediction
    #NAME_DETAILS = 's_vari0_1' ################### NEW CODE
    pred_output_path = os.path.join(output_dir, NAME_DETAILS + '_prediction.pdb')
    with open(pred_output_path, 'w') as f:
      f.write(relaxed_pdb)

    best_pdb = relaxed_pdb

  pbar.update(n=1)  # Finished AMBER relax.

# Construct multiclass b-factors to indicate confidence bands
# 0=very low, 1=low, 2=confident, 3=very high
banded_b_factors = []
for plddt in plddts[best_model_name]:
  for idx, (min_val, max_val, _) in enumerate(PLDDT_BANDS):
    if plddt >= min_val and plddt <= max_val:
      banded_b_factors.append(idx)
      break
banded_b_factors = np.array(banded_b_factors)[:, None] * final_atom_mask
to_visualize_pdb = utils.overwrite_b_factors(best_pdb, banded_b_factors)

# --- Visualise the prediction & confidence ---
show_sidechains = True
def plot_plddt_legend():
  """Plots the legend for pLDDT."""
  thresh = [
            'Very low (pLDDT < 50)',
            'Low (70 > pLDDT > 50)',
            'Confident (90 > pLDDT > 70)',
            'Very high (pLDDT > 90)']

  colors = [x[2] for x in PLDDT_BANDS]

  plt.figure(figsize=(2, 2))
  for c in colors:
    plt.bar(0, 0, color=c)
  plt.legend(thresh, frameon=False, loc='center', fontsize=20)
  plt.xticks([])
  plt.yticks([])
  ax = plt.gca()
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  plt.title('Model Confidence', fontsize=20, pad=20)
  return plt

# Color the structure by per-residue pLDDT
color_map = {i: bands[2] for i, bands in enumerate(PLDDT_BANDS)}
view = py3Dmol.view(width=800, height=600)
view.addModelsAsFrames(to_visualize_pdb)
style = {'cartoon': {
    'colorscheme': {
        'prop': 'b',
        'map': color_map}
        }}
if show_sidechains:
  style['stick'] = {}
view.setStyle({'model': -1}, style)
view.zoomTo()

#grid = GridspecLayout(1, 2)
#out = Output()
#with out:
#  view.show()
#grid[0, 0] = out

#out = Output()
#with out:
#  plot_plddt_legend().show()
#grid[0, 1] = out

#display.display(grid)

# Display pLDDT and predicted aligned error (if output by the model).
if pae_outputs:
  num_plots = 2
else:
  num_plots = 1

#plt.figure(figsize=[8 * num_plots, 6])
#plt.subplot(1, num_plots, 1)
#plt.plot(plddts[best_model_name])
#plt.title('Predicted LDDT')
#plt.xlabel('Residue')
#plt.ylabel('pLDDT')

if num_plots == 2:
  plt.subplot(1, 2, 2)
  pae, max_pae = list(pae_outputs.values())[0]
  plt.imshow(pae, vmin=0., vmax=max_pae, cmap='Greens_r')
  plt.colorbar(fraction=0.046, pad=0.04)
  plt.title('Predicted Aligned Error')
  plt.xlabel('Scored residue')
  plt.ylabel('Aligned residue')

# Save pLDDT and predicted aligned error (if it exists)
pae_output_path = os.path.join(output_dir, NAME_DETAILS + '_predicted_aligned_error.json')
if pae_outputs:
  # Save predicted aligned error in the same format as the AF EMBL DB
  rounded_errors = np.round(pae.astype(np.float64), decimals=1)
  indices = np.indices((len(rounded_errors), len(rounded_errors))) + 1
  indices_1 = indices[0].flatten().tolist()
  indices_2 = indices[1].flatten().tolist()
  pae_data = json.dumps([{
      'residue1': indices_1,
      'residue2': indices_2,
      'distance': rounded_errors.flatten().tolist(),
      'max_predicted_aligned_error': max_pae.item()
  }],
                        indent=None,
                        separators=(',', ':'))
  with open(pae_output_path, 'w') as f:
    f.write(pae_data)


# --- Download the predictions ---
#!zip -q -r {output_dir}.zip {output_dir}
#files.download(f'{output_dir}.zip')

#########################################

# if we saved an s_tensor we need to rename it to this runname:
s_filename = 'output/s_tensor'
if (os.path.exists(s_filename) == True):
  os.rename(s_filename, s_filename + NAME_DETAILS + '.pt')
print('######### FINISHED ###########')