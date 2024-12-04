import pandas as pd
import torch
import os
import sys
from pathlib import Path
from ChromoGen.model.Diffuser.training_utils import ConfigDataset

save_dir = Path(__file__).parent /'full_scan'
save_dir.mkdir(exist_ok=True,parents=True)

# Load the Dip-C conformations from the HDF5 file
config_fp = str(Path(__file__).parent.parent.parent.parent / 'downloaded_data/conformations/DipC/processed_data.h5')
segment_length=64
config_ds = ConfigDataset(
    config_fp,
    segment_length=segment_length,
    remove_diagonal=False,
    batch_size=0,
    normalize_distances=False,
    geos=None,
    organisms=None,
    cell_types=None,
    cell_numbers=None,
    chroms=None,
    replicates=None,
    shuffle=True,
    allow_overlap=True,
    two_channels=False,
    try_GPU=True,
    mean_dist_fp=None,
    mean_sq_dist_fp=None
)

# Find all regions with no more than 250 kb overlap to scan over the whole genome
rosetta = pd.read_pickle(Path(__file__).parent/'rosetta_stone.pkl')
regions = {}
for chrom in rosetta:
    indices = []
    stop = -1
    for i,(_,_,start) in enumerate(rosetta[chrom]):
        if start >= stop:
            indices.append(i)
            stop = start + 1_030_000

    regions[chrom] = indices

# Where still necessary, fetch and save all conformations in the same format as the generated full_scan conformations
for chrom,indices in regions.items():
    for region_idx in indices:
        f = save_dir / f'sample_{region_idx}_{chrom}.pt'
        if f.exists():
            continue
        genomic_index = rosetta[chrom][region_idx][-1]
        _,coords = config_ds.fetch_specific_coords(chrom,genomic_index)
        torch.save(coords.cpu().float(),f)



