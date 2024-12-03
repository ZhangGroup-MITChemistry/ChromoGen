import umap.umap_ as umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
from pandas import read_pickle
import subprocess
import seaborn as sns
import subprocess
import sys

mpl.rcParams['pdf.fonttype'] = 42

plt.rcParams['font.size'] = 24

fit = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='euclidean'
)

fraction = float(sys.argv[1])
number_gen = int(sys.argv[2])

data = np.loadtxt('%.1f/all_features_all_results_%d.txt'%(fraction, number_gen))
mol_info = read_pickle('%.1f/dict_log_all_files_all_results_%d.pkl'%(fraction, number_gen))

mean_value = np.mean(data, axis=0)
std_value = np.std(data, axis=0)

feature_z_score = (data-mean_value)/std_value

u = fit.fit(feature_z_score)
import joblib
filename = '%.1f/umap_model_all_results_15_0.1_%d.sav'%(fraction, number_gen)
joblib.dump(u, filename)
