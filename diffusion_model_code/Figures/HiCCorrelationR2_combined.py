#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=compute_correlations
##SBATCH --partition=debug-gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=20
#SBATCH --array=0-2
#SBATCH --output=./log_files/HiCCorrelationR2_combined_%a.log

#########################
# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import pickle
from torcheval.metrics.functional import r2_score
from tqdm.auto import tqdm 
import os
import sys
sys.path.insert(0,'../code/data_utils/SampleClass/')
from Coordinates import Coordinates
from Distances import Distances
sys.path.insert(1,'../code/data_utils/')
from HiCMap import HiCMap
from HiCDataset import HiCDataset
from ConfigDataset import ConfigDataset
plt.style.use('/home/gridsan/gschuette/universal/matplotlib/plot_style_2.txt')

#########################
# Basic setup 

# State basic facts about the generated data 
resolution = 20_000
data_dir1 = '/home/gridsan/gschuette/binz_group_shared/gkks/with_Zhuohan/produce_samples/GM/full_scan/corrected/'
cond_scale1=1.
rescaled_phi1=0.
data_dir2 = '../data/samples/origami_64_no_embed_reduction/full_scan/corrected/'
cond_scale2=5.
rescaled_phi2=8.

imr_dir = '/home/gridsan/gschuette/binz_group_shared/gkks/with_Zhuohan/produce_samples/full_scan_IMR/corrected/'

milestone=120
save_folder = './HiCCorrelationR2_combined/'

# Chromosomes to analyze
chroms = [*[str(k) for k in range(1,23)],'X']

# Where to save processed data
data_fp = lambda cell_type: save_folder + f'data_{cell_type}.pkl'

# Parameters to use while converting distances to contact probabilities
r_c = 1.5 
sigma = 3.72

# Perform computations on the GPU if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Place known folders in lists to generalize to multiple data folders in case this is later desired
gm_directories = [
    data_dir1
]
imr_directories = [
    imr_dir
]

# Create the save folder if necessary
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Get all filenames for the IMR/GM data
all_gm_coord_files = []
[
    all_gm_coord_files.extend(
        [ 
            d + f for f in os.listdir(d) 
            if ('sample_' in f and f.split('_')[-1].split('.')[0] in chroms  ) 
            or ('chr_' in f and f.split('/')[-1].split('_')[1] in chroms  )
        ]
    ) for d in gm_directories
]
all_imr_coord_files = []

[
    all_imr_coord_files.extend(
        [ 
            d + f for f in os.listdir(d) 
            if ('sample_' in f and f.split('_')[-1].split('.')[0] in chroms  ) 
            or ('chr_' in f and f.split('/')[-1].split('_')[1] in chroms  )
        ]
    ) for d in imr_directories
]

#########################
# Load data that will be used repeatedly 

# Experimental Hi-C 
gm_hic = HiCDataset('../data/outside/GM12878_hg19.mcool')
imr_hic = HiCDataset('../data/outside/IMR90_hg19.mcool')

# Dip-C dataset
config_fp='../data/processed_data.hdf5'
num_bins=64
config_ds = ConfigDataset(
    config_fp,
    segment_length=num_bins,
    remove_diagonal=False,
    batch_size=0,
    normalize_distances=False,
    geos=None,
    organisms=None,
    cell_types=None,
    cell_numbers=None,
    chroms=None,#[chrom for chrom in regions],
    replicates=None,
    shuffle=False,
    allow_overlap=True,
    two_channels=False,
    try_GPU=True,
    mean_dist_fp=None,#mean_dist_fp,
    mean_sq_dist_fp=None,#mean_sq_dist_fp
)

# Rosetta stone to convert between our region indices and genomic indices
rosetta = pd.read_pickle(f'../data/embeddings_{num_bins}_after_transformer/rosetta_stone.pkl')

#########################
# Define various support functions

# Convert region index to genomic index
def get_genomic_index(chrom,region_idx,rosetta=rosetta):
    return rosetta[chrom][region_idx][-1]

# Get the chromosome, region, etc., information contained in a filepath
def parse_filename(f):
    f = f.split('/')[-1]
    f = f.split('_')
    if f[0] == 'sample':
        chrom = f[-1].split('.')[0]
        region_idx = int(f[1])
    elif f[0] == 'chr':
        chrom = f[1]
        region_idx = int(f[2])
    else:
        raise Exception(f"File {'_'.join(f)} cannot be interpreted")
    genomic_index = get_genomic_index(chrom,region_idx)
    return chrom, region_idx, genomic_index

# Convert distances into Hi-C contact probabilities
def conformations_to_probs(conformations,sigma=sigma,r_c=r_c,average=True):
    p = conformations.distances.values.clone()
    mask = p < r_c
    p[mask] = ( (sigma*(r_c-p[mask])).tanh() + 1 )/2
    mask^= True
    #p[mask] = (r_c/p[mask])**4 / 2
    p[mask] = (r_c/p[mask])**3.45 / 2
    if average:
        p = HiCMap(p.mean(0))
    return p

# Compute the correlation coefficient between data in one large batch (not computational batch, but all pairs in two aligned lists)
def batch_corrcoef(vals1,vals2):
    n = len(vals1)
    assert n == len(vals2), 'vals1 and vals2 have a different number of traces'
    return torch.stack([
        torch.corrcoef(
            torch.stack(
                [vals1[i],vals2[i]],
                dim=0
            )
        )[0,1]
    for i in range(n)
    ])

# Same as above but for R-squared values
def batch_r2(input,target):
    n = len(input)
    assert n == len(target), 'vals1 and vals2 have a different number of traces'
    return torch.stack([r2_score(input[i],target[i]) for i in range(n)])

# Load/format specific coordinate files at both specified guidance strengths 
def get_gen_coords(coord_fp1,data_dir2=data_dir2,cond_scale1=cond_scale1,rescaled_phi1=rescaled_phi1,
                cond_scale2=cond_scale2,rescaled_phi2=rescaled_phi2,device=device):

    coords1 = Coordinates(coord_fp1)
    coord_fp2 = data_dir2 + coord_fp1.split('/')[-1]
    coord_fp2 = coord_fp1.replace(str(float(cond_scale1)),str(float(cond_scale2)))
    coord_fp2 = coord_fp2.replace(str(float(rescaled_phi1)),str(float(rescaled_phi2)))
    coords2 = Coordinates(coord_fp2)

    return coords1.to(device), coords2.to(device)

#########################
# Perform the computation

# To hold the data. Not actually used in this folder, but keeping since I copied this code from the Jupyter Notebook that DOES use it
corr_dist = {}
r2_prob = {}
corr_prob = {}

# There are multiple ways to construct the combined generated structure, so bootstrap the median value. 
# This defines the number of bootstrapping resamplings to perform. 
n_bootstrap_resamples = 1000

# Perform computations on the GPU if possible
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# Fraction of conformations in the combined dataset to pull from the w=1 dataset
fraction_w1s_to_test = [.5] 

# Use these to distribute work across all slurm jobs
nnn = 0 
task_id = int(os.environ['SLURM_ARRAY_TASK_ID'])

# Loop through all options
for cell_type,files,exp_hic in [
    ('GM',all_gm_coord_files,gm_hic),
    ('IMR',all_imr_coord_files,imr_hic),
    ('Tan',all_gm_coord_files,gm_hic)
]:

    # Distribute the workload across the slurm jobs
    nnn+= 1
    if nnn%3 != task_id:
        continue

    data_fp_ = data_fp(cell_type)

    if os.path.exists(data_fp_):
        temp_data = pickle.load(open(data_fp_,'rb'))
        temp_corr_dist = temp_data['corr_dist']
        temp_r2_prob = temp_data['r2_prob']
        temp_corr_prob = temp_data['corr_prob']
        del temp_data
    else:
        temp_corr_dist = {}
        temp_r2_prob = {}
        temp_corr_prob = {}

    # Decide which w=1 vs w=5 fractions to look at
    if cell_type == 'Tan':
        fractions = [0] # Experimental data has no weighting split to consider
    else:
        fractions = fraction_w1s_to_test

    # Track whether changes are made such that the data will need to be saved
    modified = False 
    for fraction_w1 in fractions:
        if fraction_w1 in temp_corr_dist:
            continue

        modified = True # The data WILL change/need to be saved
        
        gen_median_dists = []
        gen_probs = []
        exp_probs = []
        exp_log_probs = []

        for f in tqdm(files):
            
            # Determine the chromosome, region index, and genomic index
            chrom, region_idx, genomic_index = parse_filename(f)
    
            # Load the generated coordinates
            if cell_type == 'Tan':
                coords1 = Coordinates(config_ds.fetch_specific_coords(chrom,genomic_index)[1])
            else:
                coords1 = Coordinates(f)
            coords1 = coords1.to(device) 
            inferred_prob_map = conformations_to_probs(coords1)
                
            coords1.to_(inferred_prob_map.device)
            
            # Load the experimental Hi-C interaction frequencies
            start = genomic_index
            stop = start + resolution * coords1.num_beads
            exp_prob_map = exp_hic.fetch(chrom,start,stop)
    
            # Get the indices for the upper triangle of each set of data, excluding the diagonal
            n = coords1.num_beads
            i,j = torch.triu_indices(n,n,1,device=exp_prob_map.device)
    
            #####
            # Place data into the respective lists
    
            # Interaction frequency
            exp_probs.append(exp_prob_map.prob_map[i,j])
            
            # Remove values associated with NaN Hi-C data
            valid_idx = torch.where(exp_probs[-1].isfinite())[0]
            exp_probs[-1] = exp_probs[-1][valid_idx]
            i,j = i[valid_idx], j[valid_idx]
    
            # Normalize the experimental interaction frequencies to nearest neighbors to
            # obtain contact probabilities consistent with our distance-to-probability 
            # conversion
            exp_probs[-1]/=  torch.nanmean(exp_prob_map.prob_map[range(n-1),range(1,n)])
            
            # Log probability
            exp_log_probs.append(exp_probs[-1].log10())
            
            # Generated probabilities
            gen_probs.append(inferred_prob_map.prob_map[i,j])
            
            # Interactions with recorded probability 0 become undefined, so remove those points
            valid_idx = torch.where(exp_log_probs[-1].isfinite())[0]
            exp_log_probs[-1] = exp_log_probs[-1][valid_idx]
            i,j = i[valid_idx], j[valid_idx]
    
            # Generated distances
            if cell_type != 'Tan':
                coords1,coords2 = get_gen_coords(f)
                dists1 = coords1.distances
                dists2 = coords2.distances
                dists1._values*= 100
                dists2._values*= 100
                gen_median_dists.append(dists1.append(dists2).median.values[0,i,j])
                '''
                n1 = len(dists1)
                n2 = len(dists2)
                N = n1 + n2
                median_dist_resamples = []
                for _ in range(n_bootstrap_resamples):
    
                    # Decide how many samples to draw from each 
                    # set of conformations
                    nn1 = ( torch.rand(N) < ratio ).sum()
                    nn2 = N - nn1
    
                    # Draw samples to construct new dataset, find median dists, record result
                    median_dist_resamples.append(
                        dists1[torch.randint(n1,(nn1,))].append(
                            dists2[torch.randint(n2,(nn2,))]
                        ).median.values
                    )
            
                # Record the median of medians
                gen_median_dists.append(
                    torch.cat(median_dist_resamples,dim=0).median(0).values[i,j]
                )
                '''
            else:
                gen_median_dists.append(coords1.distances.median.values[0,i,j])
    
        # Compute the desired statistics
        if len(gen_median_dists) > 0:
            cd = batch_corrcoef(gen_median_dists,exp_log_probs).cpu()
            r2p = batch_r2(gen_probs,exp_probs).cpu()
            cp = batch_corrcoef(gen_probs,exp_probs).cpu()
        else:
            cd = []
            r2p = []
            cp = []

        del gen_median_dists, exp_log_probs, gen_probs, exp_probs

        temp_corr_dist[fraction_w1] = cd
        temp_r2_prob[fraction_w1] = r2p
        temp_corr_prob[fraction_w1] = cp

    if modified:
        pickle.dump(
            {
                'corr_dist':temp_corr_dist,
                'r2_prob':temp_r2_prob,
                'corr_prob':temp_corr_prob
            },
            open(data_fp_,'wb')
        )
    
    corr_dist[cell_type] = temp_corr_dist
    r2_prob[cell_type] = temp_r2_prob
    corr_prob[cell_type] = temp_corr_prob



