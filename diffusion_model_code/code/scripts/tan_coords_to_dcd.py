import sys
sys.path.insert(0,'../data_utils')
from diffusion_dists_to_dcd import convert_Tan_data

configs_filepath = '../../data/processed_data.hdf5'
destination_directory = '../../data/samples/Tan/'

rosetta_stones = { # region_length:path/to/embedding_rosetta_stone
    64:'../../data/embeddings_64_after_transformer/rosetta_stone.pkl'
}

regions_to_convert = { # chrom:region_idx
    '1':[144,200,265,330,395,460,525,590,730,795,860,1260,1325],
    'X':[100,236,381,445,553,610,675,810,900,965,1060,1125,1200]
}

for chrom,region_indices in regions_to_convert.items():
    for region_length,rosetta_stone_fp in rosetta_stones.items():
        convert_Tan_data(configs_filepath,rosetta_stone_fp,destination_directory,chrom,region_indices,region_length)


