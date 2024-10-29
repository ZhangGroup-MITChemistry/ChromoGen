import os
from pandas import read_pickle

# File locations
#embeddings_main_dir = '../../../data/raw_embeddings/' # Where Zhuohan's embeddings are located 
#embeddings_main_dir = '/home/gridsan/gschuette/binz_group_shared/zlao/for_greg/hg19/'
embeddings_main_dir = '/home/gridsan/gschuette/binz_group_shared/zlao/for_greg/hg19_with_new_hic_optim/node_embedding/'
save_fp = './missing_files.tsv' # .tsv file where we should document the missing embeddings
#zhuohan_dict_fp = '/home/gridsan/gschuette/binz_group_shared/zlao/for_greg/hg19/my_dict.pickle'
zhuohan_dict_fp = '/home/gridsan/gschuette/binz_group_shared/zlao/for_greg/hg19_with_new_hic_optim/node_embedding/my_dict.pickle'

# Load Zhuohan's dictionary 
zhuohan_dict = read_pickle(zhuohan_dict_fp)

# Create a new file and write a formatted 
with open(save_fp, 'w') as f: # Using 'w' will overwrite the file if it already exists
    f.write('Chrom\tIndex\n')
    for chrom in zhuohan_dict: 
    
        path = lambda idx: embeddings_main_dir+f'run_scripts_{chrom}/chr_{chrom}_{idx}.pt'
        idxs = [ int(fn.split('_')[-1].split('.')[0]) for fn in os.listdir(embeddings_main_dir+f'run_scripts_{chrom}/') ]
        for k in range(len(zhuohan_dict[chrom][:,0])):
            try:
                idxs.remove(k)
            except:
                f.write(f'{chrom}\t{k}\n')
