import shutil
import os

def convert_filename(f):
    ff = f.split('_')
    chrom = ff[1]
    region_idx = int(ff[2])
    return f'sample_{region_idx}_5.0_8.0_120_{chrom}.pt'

dirs = [
    ('./GM/','/home/gridsan/gschuette/binz_group_shared/gkks/with_Zhuohan/conformations/GM/'),
    ('./IMR/','/home/gridsan/gschuette/binz_group_shared/gkks/with_Zhuohan/conformations/IMR/')
]
n=0
for start_dir,end_dir in dirs:

    files = [f for f in os.listdir(start_dir) if os.path.isfile(start_dir+f)]
    for f in files:
        new_fname = convert_filename(f)
        n+=1
        shutil.copyfile(start_dir + f,end_dir + new_fname)


