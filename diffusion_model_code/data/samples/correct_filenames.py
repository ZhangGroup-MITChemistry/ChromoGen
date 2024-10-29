
import os

dirs = [fp+'/dcd_files/' for fp in os.listdir('./') if os.path.isdir(fp+'/dcd_files') and len(os.listdir(fp+'/dcd_files'))>1]

for directory in dirs: 
    for file in os.listdir(directory):
        f = file.split('_')
        for i in range(2,4):
            f[i] = f[i][0] + '.' + f[i][1:]
        f = '_'.join(f)
        os.rename(directory+file,directory+f)


