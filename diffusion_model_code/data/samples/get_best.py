import pandas as pd
import os
import torch

n_best = 5

files = [
    f for f in os.listdir('./') if '.pkl' in f
]
files.sort()

results = {
    'File':[],
    'Loss':[],
    'Index':[],
    'Frame':[]
}
for f in files:

    temp = pd.read_pickle(f)

    for i in range(len(temp)):
        results['File'].append(f)
        results['Loss'].append(temp[i]['Loss'])
        results['Index'].append(temp[i]['Index'])
        results['Frame'].append(i)

results['Loss'] = torch.tensor(results['Loss']).flatten()
i=0
while i < n_best:
    val,idx = results['Loss'].min(0)
    if idx.numel() > 1:
        idx = idx[0]
        val = val[0]
    print(f'{i}th Best:')
    print('\t'+f"File: {results['File'][int(idx)]}")
    print('\t'+f"Loss: {results['Loss'][int(idx)]}")
    print('\t'+f"Index: {results['Index'][int(idx)]}")
    print('\t'+f"Frame: {results['Frame'][int(idx)]}")
    i+=1
    results['Loss'][idx] = results['Loss'].max()
    
