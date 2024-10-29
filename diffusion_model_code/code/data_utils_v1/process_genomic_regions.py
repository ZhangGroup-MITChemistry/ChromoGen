'''
Given a pre-processed dataset, this will find the genomic regions corresponding to each of the 
identified starting positions (adjust parameters below, if desired). The information is saved 
to a tsv file for interfacing with Zhuohan's stuff. 
'''

from DataLoader import DataLoader

nbeads = 128 #65
dataset_filepath = '../../data/processed_data.hdf5'
batch_size = 64
two_channels = False 
allow_overlap = True 
csv_filepath = '../../data/genomic_regions_128.tsv'

dl = DataLoader(
    dataset_filepath,
    segment_length=nbeads,
    batch_size=batch_size,
    normalize_distances=True,
    geos=None,
    organisms=None,
    cell_types=None,
    cell_numbers=None,
    chroms=None,
    replicates=None,
    shuffle=True,
    allow_overlap=allow_overlap, 
    two_channels=two_channels,
    try_GPU=True,
    mean_dist_fp='../../data/mean_dists.pt',
    mean_sq_dist_fp='../../data/squares.pt'
)

indices = dl.get_genomic_regions()

# Sort them for later convenience
sub_idx = indices['Chromosome'] != 'X' # only non-numbered chromosome included in the dataset
indices.loc[sub_idx,'Chromosome'] = pd.to_numeric(indices.loc[sub_idx,'Chromosome'])
indices = indices.sort_values(['Chromosome','Start'],ignore_index=True)
indices['Chromosome'] = indices['Chromosome'].astype(str)

indices.to_csv(csv_filepath,sep='\t')

