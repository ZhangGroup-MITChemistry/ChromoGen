import pandas as pd
import pickle
import sys
sys.path.insert(0,'../data_utils/')
from HiCDataset import HiCDataset

import matplotlib.pyplot as plt
plt.style.use('/home/gridsan/gschuette/universal/matplotlib/plot_style.txt')

save_folder='HiC/'

exp_hic = HiCDataset()

full_map = exp_hic.fetch(adaptive_coarsegraining=True,interp_nans=True)

pickle.dump(full_map,open(save_folder+"full_GM.pkl",'wb'))





