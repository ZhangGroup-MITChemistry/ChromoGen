import pandas as pd
import pickle

chroms = [*[str(k) for k in range(1,23)],'X']

rosetta_stone = {}
for chrom in chroms: 
    #embedding = pd.read_pickle(f'./chrom_{chrom}.tar.gz')
    #embedding.index.to_pickle(f'./chrom_{chrom}_index.tar.gz')
    rosetta_stone[chrom] = pd.read_pickle(f'./chrom_{chrom}.tar.gz').index

pickle.dump(rosetta_stone,open('rosetta_stone.pkl','wb'))

