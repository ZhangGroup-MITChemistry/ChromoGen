#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=get_stats
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-10
#SBATCH --output=./log_files/get_stats_%a.log

import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import pickle
from torcheval.metrics.functional import r2_score
import os
import sys
sys.path.insert(0,'../code/data_utils/SampleClass/')
from Coordinates import Coordinates
from Distances import Distances
sys.path.insert(1,'../code/data_utils/')
from HiCMap import HiCMap
from HiCDataset import HiCDataset
plt.style.use('/home/gridsan/gschuette/universal/matplotlib/plot_style_2.txt')

resolution = 20_000
data_dir1 = '/home/gridsan/gschuette/binz_group_shared/gkks/with_Zhuohan/produce_samples/GM/full_scan/corrected/'
cond_scale1=1.
rescaled_phi1=0.
data_dir2 = '../data/samples/origami_64_no_embed_reduction/full_scan/corrected/'
cond_scale2=5.
rescaled_phi2=8.

milestone=120
save_folder = './HiCCorrelationR2_combined/'
data_fp = save_folder + 'data.pkl'

chroms = [*[str(k) for k in range(1,23)],'X']

r_c = 1.5
sigma = 3.72

'''
gm_directories = [
    '../data/samples/origami_64_no_embed_reduction/eval_mode/',
    '../data/samples/origami_64_no_embed_reduction/cross_validation/GM/',
    '../data/samples/origami_64_no_embed_reduction/active_inactive_repressed/'
]
imr_directories = [
    '../data/samples/origami_64_no_embed_reduction/cross_validation/IMR/',
]
'''
gm_directories = [
    data_dir1
]
imr_directories = []

gm_hic = HiCDataset('../data/outside/GM12878_hg19.mcool')
imr_hic = HiCDataset('../data/outside/IMR90_hg19.mcool')

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

def get_genomic_index(chrom,region_idx,rosetta=pd.read_pickle('../data/embeddings_64_after_transformer/rosetta_stone.pkl')):
    return rosetta[chrom][region_idx][-1]

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

def batch_r2(input,target):
    n = len(input)
    assert n == len(target), 'vals1 and vals2 have a different number of traces'
    return torch.stack([r2_score(input[i],target[i]) for i in range(n)])
    
def plot_region(
    coord_fp1,
    coord_fp2,
    exp_hic,
    r_c=r_c,#1.5, # CUTOFF 2 because https://doi.org/10.1038/nature21429
    sigma=sigma,
    resolution=resolution,
    data_dir1=data_dir1,
    data_dir2=data_dir2,
    cond_scale1=cond_scale1,
    rescaled_phi1=rescaled_phi1,
    cond_scale2=cond_scale2,
    rescaled_phi2=rescaled_phi2,
    milestone=milestone,
    ratio=.632, # Fraction from coord_fp1
    fig=None,
    ax=None,
    choose_exp_vmin=False
):

    # Get the chromosome, region index, and genomic index
    chrom, region_idx, genomic_index = parse_filename(coord_fp1)

    # Load the generated coordinates
    coords1 = Coordinates(coord_fp1)
    coords2 = Coordinates(coord_fp2)

    # Convert generated coordinates into Hi-C interaction frequencies
    gen_map = conformations_to_probs(coords1)
    gen_map.prob_map*= ratio
    gen_map.prob_map+= conformations_to_probs(coords2).prob_map * (1-ratio)

    # Get experimental Hi-C
    start = genomic_index
    stop = start + resolution * coords1.num_beads
    exp_map = exp_hic.fetch(chrom,start,stop)

    # Normalize the experimental Hi-C to nearest neighbor contact probabilities
    n = exp_map.prob_map.shape[-1]
    exp_map.prob_map/= torch.nanmean(exp_map.prob_map[range(n-1),range(1,n)])

    vmin = exp_map.prob_map[exp_map.prob_map.isfinite()].min() if choose_exp_vmin else None
    fig,ax,im,cbar = exp_map.plot_with(gen_map,fig=fig,ax=ax,vmin=vmin)
    ax.set_xlabel('Genomic index')
    ax.set_ylabel('Genomic index')
    start_Mb = round(start/1e6,3)
    stop_Mb = round(stop/1e6,3)
    ax.set_title(f'Chromosome {chrom}:' + '\n' + f'{start_Mb}-{stop_Mb} Mb')
    cbar.set_label('Interaction frequency')

    return fig, ax, im, cbar

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
[all_imr_coord_files.extend([ d + f for f in os.listdir(d) if 'sample_' in f or 'chr_' in f ]) for d in imr_directories];

def get_gen_coords(coord_fp1,data_dir2=data_dir2,cond_scale1=cond_scale1,rescaled_phi1=rescaled_phi1,
                cond_scale2=cond_scale2,rescaled_phi2=rescaled_phi2):

    coords1 = Coordinates(coord_fp1)
    coord_fp2 = data_dir2 + coord_fp1.split('/')[-1]
    coord_fp2 = coord_fp1.replace(str(float(cond_scale1)),str(float(cond_scale2)))
    coord_fp2 = coord_fp2.replace(str(float(rescaled_phi1)),str(float(rescaled_phi2)))
    coords2 = Coordinates(coord_fp2)

    return coords1, coords2

'''
def get_gen_data(coord_fp1,data_dir2=data_dir2,cond_scale1=cond_scale1,rescaled_phi1=rescaled_phi1,
                cond_scale2=cond_scale2,rescaled_phi2=rescaled_phi2,ratio=.632):

    # Load all conformations
    dists1 = Coordinates(coord_fp1).distances
    coord_fp2 = data_dir2 + coord_fp1.split('/')[-1]
    coord_fp2 = coord_fp1.replace(str(float(cond_scale1)),str(float(cond_scale2)))
    coord_fp2 = coord_fp2.replace(str(float(rescaled_phi1)),str(float(rescaled_phi2)))
    dists2 = Coordinates(coord_fp2).distances
'''
def get_gen_data(coord_fp1,fraction=0.8):

    coords1,coords2 = get_gen_coords(coord_fp1)
    dists1 = coords1.distances
    dists2 = coords2.distances


    # Contact probabilities
    gen_map = conformations_to_probs(dists1)
    gen_map.prob_map*= fraction
    gen_map.prob_map+= conformations_to_probs(dists2).prob_map * (1-fraction)

    # Mean distances
    mean_dists = dists1.mean
    mean_dists._values*= fraction
    mean_dists._values+= dists2.mean.values * (1-fraction)

    return gen_map, mean_dists

fig,axes = plt.subplots(ncols=3,nrows=3,layout='constrained',figsize=(8,8))
k=0
while k < 9:
    ax = axes[k//3,k%3]

    coord_fp1 = all_gm_coord_files[-k-1]
    coord_fp2 = data_dir2 + coord_fp1.split('/')[-1]
    coord_fp2 = coord_fp2.replace(str(float(cond_scale1)),str(float(cond_scale2)))
    coord_fp2 = coord_fp2.replace(str(float(rescaled_phi1)),str(float(rescaled_phi2)))

    fig, ax, im, cbar = plot_region(coord_fp1,coord_fp2,gm_hic,fig=fig,ax=ax)
    k+=1


'''
fig,axes = plt.subplots(ncols=3,nrows=3,layout='constrained',figsize=(8,8))
k=0
n=0
while k < 9:
    n+=1
    if not '_2' in all_gm_coord_files[-n]:
        continue
    ax = axes[k//3,k%3]
    fig, ax, im, cbar = plot_region(all_gm_coord_files[-n],gm_hic,fig=fig,ax=ax)
    k+=1
'''

fig,axes = plt.subplots(ncols=3,nrows=3,layout='constrained',figsize=(8,8.2))
k=0
n=0
ratio = 0.5
while k < 9:
    n+=1
    if not '_2' in all_gm_coord_files[-n]:
        continue
    ax = axes[k//3,k%3]

    coord_fp1 = all_gm_coord_files[-n]
    coord_fp2 = data_dir2 + coord_fp1.split('/')[-1]
    coord_fp2 = coord_fp2.replace(str(float(cond_scale1)),str(float(cond_scale2)))
    coord_fp2 = coord_fp2.replace(str(float(rescaled_phi1)),str(float(rescaled_phi2)))
    
    fig, ax, im, cbar = plot_region(coord_fp1,coord_fp2,gm_hic,fig=fig,ax=ax,ratio=ratio)
    k+=1
fig.suptitle(f'Fraction $w=1$ contribution: {ratio}')
    

'''
fig,axes = plt.subplots(ncols=3,nrows=3,layout='constrained',figsize=(8,8))
k=0
n=0
while k < 9:
    n+=1
    if not '_2' in all_gm_coord_files[-n]:
        continue
    ax = axes[k//3,k%3]
    fig, ax, im, cbar = plot_region(all_gm_coord_files[-n],gm_hic,fig=fig,ax=ax)
    k+=1
'''

fig,axes = plt.subplots(ncols=3,nrows=3,layout='constrained',figsize=(10,8))
k=0
n=0
ratio = 0.5
i,j = torch.triu_indices(64,64,1)
while k < 9:
    n+=1
    if not '_2' in all_gm_coord_files[-n]:
        continue
    ax = axes[k//3,k%3]

    coord_fp1 = all_gm_coord_files[-n]
    coord_fp2 = data_dir2 + coord_fp1.split('/')[-1]
    coord_fp2 = coord_fp2.replace(str(float(cond_scale1)),str(float(cond_scale2)))
    coord_fp2 = coord_fp2.replace(str(float(rescaled_phi1)),str(float(rescaled_phi2)))

    mixed_dists = Coordinates(coord_fp1).distances.mean
    mixed_dists._values*= ratio
    mixed_dists._values+= Coordinates(coord_fp2).distances.mean.values * (1-ratio)
    mixed_dists._values*= 100
    
    fig, ax, im, cbar = mixed_dists.plot(fig=fig,ax=ax,cbar_orientation='horizontal') #plot_region(coord_fp1,coord_fp2,gm_hic,fig=fig,ax=ax,ratio=ratio)
    cbar.set_label('')
    
    chrom,_,start = parse_filename(coord_fp1)
    stop = start + 1_280_000
    exp_map = gm_hic.fetch(chrom,start,stop,interp_nans=True)
    exp_map.prob_map[j,i] = torch.nan
    *_,cbar = exp_map.plot(fig=fig,ax=ax)
    cbar.set_label('')
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    
    k+=1
fig.suptitle(f'Fraction $w=1$ contribution: {ratio}')
    

from tqdm.auto import tqdm 

# Number of resamplings to perform during bootstrapping to get median distance
n_bootstrap_resamples = 100

'''
if os.path.exists(data_fp):
    data = pickle.load(open(data_fp,'rb'))
    #corr_dist = data['corr_dist']
    corr_dist = {}
    corr_mean_dist = data['corr_mean_dist']
    r2_prob = data['r2_prob']
    corr_prob = data['corr_prob']
else:
    #r2_dist = {}
    corr_dist = {}
    r2_prob = {}
    corr_prob = {}
'''

ratio_ = int(os.environ['SLURM_ARRAY_TASK_ID']) / 10

data_fp = data_fp.replace('.pkl',f'_{ratio}.pkl')

corr_dist = {}
r2_prob = {}
corr_prob = {}

for ratio in [ratio_]:#[0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.]:

    if ratio not in corr_dist:
        #r2_dist[ratio] = {}
        corr_dist[ratio] = {}
        r2_prob[ratio] = {}
        corr_prob[ratio] = {}

    if 'GM' in corr_dist[ratio] and 'IMR' in corr_dist[ratio]:
        continue

    for cell_type,files,exp_hic in [('GM',all_gm_coord_files,gm_hic),('IMR',all_imr_coord_files,imr_hic)]:
    
        gen_median_dists = []
        gen_probs = []
        exp_probs = []
        exp_log_probs = []
        
        for f in tqdm(files):
            
            # Determine the chromosome, region index, and genomic index
            chrom, region_idx, genomic_index = parse_filename(f)
    
            # Load the generated coordinates
            coords1 = Coordinates(f)
    
            # Convert coordinates to distances and obtain the average contact probabilities
            #dists = coords.distances
            #gen_prob_map = conformations_to_probs(dists)
            gen_prob_map,mean_dists = get_gen_data(f,fraction=ratio)
            
            # Load the experimental Hi-C interaction frequencies
            start = genomic_index
            stop = start + resolution * coords1.num_beads
            exp_prob_map = exp_hic.fetch(chrom,start,stop)
    
            # Get the indices for the upper triangle of each set of data, excluding the diagonal
            n = coords1.num_beads
            i,j = torch.triu_indices(n,n,1)
    
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
            gen_probs.append(gen_prob_map.prob_map[i,j])
            
            # Interactions with recorded probability 0 become undefined, so remove those points
            valid_idx = torch.where(exp_log_probs[-1].isfinite())[0]
            exp_log_probs[-1] = exp_log_probs[-1][valid_idx]
            i,j = i[valid_idx], j[valid_idx]
    
            # Generated distances
            coords1,coords2 = get_gen_coords(f)
            dists1 = coords1.distances
            dists2 = coords2.distances
            dists1._values*= 100
            dists2._values*= 100
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
            
            #gen_median_dists.append(dists.median.values[0,i,j])
            #gen_median_dists.append(mean_dists.values[0,i,j])
    
        # Compute the desired statistics
        if len(gen_median_dists) > 0:
            #r2_dist[ratio][cell_type] = batch_r2(gen_median_dists,exp_log_probs) 
            corr_dist[ratio][cell_type] = batch_corrcoef(gen_median_dists,exp_log_probs)
            r2_prob[ratio][cell_type] = batch_r2(gen_probs,exp_probs)
            corr_prob[ratio][cell_type] = batch_corrcoef(gen_probs,exp_probs)
        else:
            #r2_dist[ratio][cell_type] = []
            corr_dist[ratio][cell_type] = []
            r2_prob[ratio][cell_type] = []
            corr_prob[ratio][cell_type] = []
pickle.dump(
    {
        #'corr_mean_dist':corr_mean_dist,
        'corr_dist':corr_dist,
        'r2_prob':r2_prob,
        'corr_prob':corr_prob
    },
    open(data_fp.replace('.pt',f'_{ratio_}.pt'),'wb')
)



