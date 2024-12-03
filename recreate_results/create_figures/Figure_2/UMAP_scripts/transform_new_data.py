import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
from pandas import read_pickle
import subprocess
import seaborn as sns
import joblib
import torch
from scipy.spatial import distance_matrix

mpl.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.size'] = 24

lst = [('395','1'), ('1325','1'), ('464', '22')]

data = np.loadtxt('all_features_800.txt')

mean_value = np.mean(data, axis=0)
std_value = np.std(data, axis=0)

data = (data-mean_value)/std_value

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

    return distance_feature

k = []

for curr_, chrom in lst:
    weak = torch.load('/home/gridsan/zlao/binz_group_shared/gkks/with_Zhuohan/conformations/GM/sample_%s_1.0_0.0_120_%s.pt'%(curr_,chrom)).numpy()
    strong = torch.load('/home/gridsan/zlao/binz_group_shared/gkks/with_Zhuohan/conformations/GM/sample_%s_5.0_8.0_120_%s.pt'%(curr_,chrom)).numpy()
    weak = compute_dismat(weak)
    strong = compute_dismat(strong)
    all_data = weak+strong
    all_data = np.array(all_data)
    all_data = (all_data-mean_value)/std_value
    l1 = len(data)
    mask = ~np.isnan(all_data).any(axis=1)
    all_data = all_data[mask]
    data = np.vstack((data, all_data))
    l2 = len(data)
    
    point = torch.load('/home/gridsan/zlao/binz_group_shared/gkks/with_Zhuohan/conformations/Tan/sample_%s_%s.pt'%(curr_,chrom)).numpy()
    point = compute_dismat(point)
    point = np.array(point)
    point = (point-mean_value)/std_value
    data = np.vstack((data, point))
    l3 = len(data)
    k.append([l1,l2,l3])

u = joblib.load('umap_model_all_results_15_0.1_800.sav')

new_embeddings = u.transform(data[2357920:])
np.savetxt('new_embeddings.txt', np.array(new_embeddings), fmt='%.6f')
