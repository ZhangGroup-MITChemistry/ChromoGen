import torch
import pandas as pd

for fn in open('data_filenames.txt','r').readlines():
    if 'overlap' in fn:
        continue
    fn = fn.replace('%20','_').strip()
    df = pd.read_csv(
        fn,
        header=1
    )
    dfs = []
    if 'Cell cycle' in df:
        for cc in df['Cell cycle'].unique():
            dfs.append(df[df['Cell cycle']==cc])
    else:
        dfs.append(df)

    conformations = []
    for df in dfs:
        xyz = df[['X','Y','Z']]
        conformations.extend([ # pandas errantly converts the stored ints to np.float64's -> torch.double's. More convenient to have floats anyway, but use 32 bit for storage space
            torch.from_numpy(xyz[df['Chromosome index']==idx].values).float() for idx in df['Chromosome index'].unique()
        ])

    torch.save(torch.stack(conformations,dim=0),fn.replace('.csv','.pt'))



