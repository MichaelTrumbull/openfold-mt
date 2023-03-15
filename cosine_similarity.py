'''
Latent spae data from the unperturbed run generated ~35gb of data. Too much for my laptop.
Here I will generate the data I am interested in. Cosine similarity grid.

'''
import pickle
import os
import torch
import torch.nn as nn
import numpy as np

cos = nn.CosineSimilarity(dim=0)

PTH = "predictions-latentspace-unperturbed/"
save_dir = 'predictions/'

def latent_space_cosine_grid(protein, rep):
    data = np.empty([32,32])
    for r1 in range(4):
        for i1 in range(8):
            for r2 in range(4):
                for i2 in range(8):
                    file_1 = PTH + protein + '_lspace' + '/' + rep + '_iter_' + str(i1) + '_recy_' + str(r1) + '_.pt'
                    file_2 = PTH + protein + '_lspace' + '/' + rep + '_iter_' + str(i2) + '_recy_' + str(r2) + '_.pt'
                    t1 = torch.flatten(torch.load(file_1))
                    t2 = torch.flatten(torch.load(file_2))
                    data[(r1*8)+i1, (r2*8)+i2] = cos(t1,t2)
    with open(save_dir + protein + '_cosinesim_' + rep, 'wb') as f:
        pickle.dump(data, f)

for file in os.listdir(PTH):
    if os.path.isdir(PTH + file):
        latent_space_cosine_grid(file[0:6], 's')
