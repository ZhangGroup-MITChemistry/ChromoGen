import pandas as pd
import os
import sys
sys.path.insert(1,'/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/code/data_utils')
from Sample import Sample

files = os.listdir('./')
files.sort()
i = 0 
for file in files: 
    if 'sample' not in file or '.pkl' not in file or 'backup' in file: 
        continue

    if i > 0:
        print('')
    print(file)
    print(pd.read_pickle(file).unflatten().shape)
    i+=1


