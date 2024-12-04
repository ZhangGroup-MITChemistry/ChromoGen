import torch
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
from os import listdir, chdir
from os.path import isfile, join
import sys
from pandas import read_pickle
import pickle
import subprocess

GM_produced_conformations = '/home/gridsan/zlao/binz_group_shared/gkks/with_Zhuohan/produce_samples/GM/full_scan/corrected/'
Tan_conformations = '/home/gridsan/zlao/binz_group_shared/gkks/with_Zhuohan/tan_full_scan/full_scan/'

mol_info = read_pickle('/home/gridsan/zlao/binz_group_shared/gkks/with_Zhuohan/conformations/rosetta_stone.pkl')

fraction = float(sys.argv[1])
number_gen = int(sys.argv[2])

subprocess.call(["mkdir -p %.1f"%fraction],shell=True,stdout=subprocess.PIPE)

def compute_dismat(configs):
    distance_feature = []
    for conf in configs:
        distance_mat = distance_matrix(conf, conf)
        curr_feature = []
        n_dist = len(distance_mat)
        for i in range(n_dist):
            for j in range(i+1, n_dist):
                curr_feature.append(distance_mat[i][j])

        distance_feature.append(curr_feature)

    distance_feature = np.array(distance_feature)
    return distance_feature

files = np.loadtxt('files_TAN.txt', dtype=str)
dict_log_all_files = {}
all_features = []

num_rows_to_select = int(number_gen*fraction)
curr_idx = 0

for file in files:
    guidance_1 = torch.load(GM_produced_conformations+file).numpy()
    mask = np.isnan(guidance_1).any(axis=(1, 2))
    guidance_1 = guidance_1[~mask]
    if np.any(np.isnan(guidance_1)):
        print('bad sample'+file)
        continue
    curr_file = file[:-3].split('_')
    chrom = curr_file[-1]
    pos = int(mol_info[chrom][int(curr_file[1])][2])/1000000
    guidance_5 = torch.load(GM_produced_conformations+curr_file[0]+'_'+curr_file[1]+'_5.0_8.0_120_'+curr_file[5]+'.pt').numpy()
    mask = np.isnan(guidance_5).any(axis=(1, 2))
    guidance_5 = guidance_5[~mask]
    if np.any(np.isnan(guidance_5)):
        print('bad sample'+file)
        continue

    random_indices = np.random.choice(guidance_1.shape[0], size=num_rows_to_select, replace=False)
    new_guidance_1 = guidance_1[random_indices]
    random_indices = np.random.choice(guidance_5.shape[0], size=number_gen-num_rows_to_select, replace=False)
    new_guidance_5 = guidance_5[random_indices]
    all_features.append(compute_dismat(new_guidance_1))
    all_features.append(compute_dismat(new_guidance_5))

    curr_tan = curr_file[0]+'_'+curr_file[1]+'_'+curr_file[5]+'.pt'
    if isfile(Tan_conformations+curr_tan):
        tan_data = torch.load(Tan_conformations+curr_tan).numpy()
        all_features.append(compute_dismat(tan_data))

    dict_log_all_files['chrom_%s_%.2f'%(chrom, pos)] = [curr_idx, curr_idx + number_gen + len(tan_data), curr_tan]
    curr_idx += number_gen + len(tan_data)

with open('%.1f/dict_log_all_files_all_results_%d.pkl'%(fraction, number_gen), 'wb') as f:
    pickle.dump(dict_log_all_files, f)

np.savetxt('%.1f/all_features_all_results_%d.txt'%(fraction, number_gen), np.vstack(all_features), fmt='%.6f')
