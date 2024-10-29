#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=align_files
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --output=./log_files/pairwise_alignment.py

import mdtraj as md
import torch
import pandas as pd
import copy
import os
import pickle
from tqdm.auto import tqdm
import sys
sys.path.insert(0,'./')
from NewSample import coords_to_dists, loss_fcn, coord_to_dcd, smooth_transition_loss_by_sample

def to_list(object):
    return object if type(object) == list else [object]

def parse_filename(filename):
    filename = filename.split('/')[-1]  # remove directory path
    components = filename.split('_')
    region_idx = int(components[1])
    cond_scale = float(components[2]) if '.' in components[2] else int(components[2])
    rescaled_phi = float(components[3]) if '.' in components[3] else int(components[3])
    milestone = int(components[4])
    chrom = components[5].split('.')[0]

    return region_idx, cond_scale, rescaled_phi, milestone, chrom

def is_valid(
    filename,
    region_idx,
    cond_scale,
    rescaled_phi,
    milestone,
    chrom
):

    try:
        ri, cs, rp, ms, ch = parse_filename(filename) 
    except:
        return False, (None, None, None, None, None)
    
    for desired,actual in [(region_idx,ri),(cond_scale,cs),
                         (rescaled_phi,rp),(milestone,ms),(chrom,ch)]:
        if desired is not None and actual not in to_list(desired):
            return False, (ri, cs, rp, ms, ch)
    
    return True, (ri, cs, rp, ms, ch)
    
def get_all_sample_types(
    sample_directory,
    *,
    region_idx=None,
    cond_scale=None,
    rescaled_phi=None,
    milestone=None,
    chrom = None
):
    samples = os.listdir(sample_directory)

    to_process = pd.DataFrame({
        'region_idx':[],
        'cond_scale':[],
        'rescaled_phi':[],
        'milestone':[],
        'chrom':[],
    })
    
    for f in samples:
        valid, properties = is_valid(f,region_idx,cond_scale,rescaled_phi,milestone,chrom) 
        if valid: 
            to_process.loc[len(to_process)] = properties

    to_process.sort_values(
        ['milestone','chrom','region_idx','cond_scale','rescaled_phi'], # sorts in this order
        inplace=True,
        ignore_index=True,
    )

    return [*to_process.itertuples(index=False,name=None)]

''' Original
def get_best_alignments(t_ref,t_sample,r_c=1.,long_scale=1/8,use_gpu=True,high_precision=False):

    ref_dists = coords_to_dists(torch.from_numpy(t_ref.xyz))
    ref_dists = ref_dists.unsqueeze(1).expand(-1,len(t_sample),-1,-1)
    sample_dists = coords_to_dists(torch.from_numpy(t_sample.xyz))
    sample_dists = sample_dists.unsqueeze(0).expand(len(t_ref),-1,-1,-1)

    losses = smooth_transition_loss_by_sample(sample_dists,ref_dists,r_c,long_scale,use_gpu,high_precision)

    best_loss,best_loss_idx = losses.min(1)
    
    coords = torch.empty(*t_ref.xyz.shape)
    for frame,idx in enumerate(best_loss_idx):
        ts1 = t_sample[idx]
        ts2 = copy.deepcopy(ts1)
        ts2.xyz[...,-1]*= -1 # Reflect, in case we computed the isomer with backward chirality
        ts1.superpose(t_ref,frame=frame)
        ts2.superpose(t_ref,frame=frame)

        diff1 = ((t_ref[frame].xyz - ts1.xyz)**2).sum()
        diff2 = ((t_ref[frame].xyz - ts2.xyz)**2).sum()
        if diff1 < diff2: 
            coords[frame,...] = torch.from_numpy(ts1.xyz).squeeze(0)
        else:
            coords[frame,...] = torch.from_numpy(ts2.xyz).squeeze(0)

    best_info = [
        {
            'Loss':best_loss[i],
            'Index':best_loss_idx[i]
        } for i in range(len(best_loss))
    ]
    
    return best_info,coords
'''
'''
def get_best_alignments(t_ref,t_sample,r_c=1.,long_scale=1/8,use_gpu=True,high_precision=False):
    '#''
    Simply align all conformations, see which have the smallest disagreement with the reference
    '#''
    best_indices = []
    best_measures = []
    coords = torch.empty(*t_ref.xyz.shape)
    for frame in range(len(t_ref)):    
        for i in range(2):
            if i == 1:
                t_sample.xyz[...,-1]*= -1 # Reflect the conformations in case we have a stereoisomer
            else:
                running_best_measure = None
                
            # Align conformations with the reference
            t_sample.superpose(t_ref,frame=frame)
    
            # Place xyz in torch so GPU can be used if desired
            sample_xyz = torch.from_numpy(t_sample.xyz)
            ref_xyz = torch.from_numpy(t_ref.xyz[frame:frame+1])
            if use_gpu and torch.cuda.is_available(): 
                sample_xyz = sample_xyz.cuda()
                ref_xyz = ref_xyz.cuda()
            if high_precision:
                sample_xyz = sample_xyz.double()
                ref_xyz = ref_xyz.double()
    
            ### 
            # Get overall alignment, measured by magnitude of projection / magnitude of starting structure & relative magnitudes of vecotrs in each case
    
            # Used several times, so compute once
            ref_norms = torch.linalg.norm(ref_xyz,dim=-1)
            sample_norms = torch.linalg.norm(sample_xyz,dim=-1)
            
            # magnitude of projection / magnitude of starting structure -- Higher is better, so multiply by -1 to treat as loss
            losses = -torch.linalg.vecdot(ref_xyz,sample_xyz,dim=-1) / sample_norms # coefficient for projection of each vector
            losses/= ref_norms # Normalize to the length of original vectors
    
            # Difference in vector magnitudes -- Lower is better, so add it to maintain consistency with the above
            '#''
            mask = ref_norms!=0
            mask1 = mask.expand_as(losses)
            mask2 = mask.expand_as(sample_norms)
            losses[mask1]+= (ref_norms[mask]-sample_norms[mask2]).abs() / ref_norms[mask]
            '#''
            mask = ref_norms!=0
            ref_norms = ref_norms.expand_as(losses)
            mask = mask.expand_as(losses)
            losses[mask]+= (ref_norms[mask]-sample_norms[mask]).abs() / ref_norms[mask]
    
            # Average over all monomers in each structure to measure alignment quality
            losses.nan_to_num_(torch.inf)
            losses = losses.mean(-1)

            ###
            # Decide which conformation best matches the reference
            best_measure,best_idx = losses.min(0)

            # Record the conformation
            if running_best_measure is None or best_measure < running_best_measure:
                running_best_measure = best_measure
                running_best_idx = best_idx
                conformation = sample_xyz[best_idx,...]

        # Now that both reflected & non-reflected conformations have been tested,
        # record the best-aligned conformation
        coords[frame,...] = conformation.cpu().float()
        best_indices.append(running_best_idx)
        best_measures.append(running_best_measure)

    best_info = [
        {
            'Loss':best_measures[i],
            'Index':best_indices[i]
        } for i in range(len(best_measures))
    ]
    return best_info,coords
'''
def get_best_alignments(t_ref,t_sample,r_c=1.,long_scale=1/8,use_gpu=True,high_precision=False):
    '''
    Simply compute RMSD values. Leaving r_c, etc., arguments so I don't have to change the
    other functions that call this for now. 
    '''

    # Make a copy of the reference frames to ensure the superposition afterwards 
    # relates to the orientation saved in the referenced dcd file. 
    t_ref1 = copy.deepcopy(t_ref)
    rmsds = torch.stack(
        [
            torch.from_numpy(md.rmsd(t_sample,frame)) for frame in t_ref1
        ],
        dim=1
    )

    # Identify the generated conformations that can be best aligned with the reference conformations
    best_rmsds,best_rmsd_indices = rmsds.min(0)

    # Superpose the best-matched conformations with the reference conformation and place them into a coordinates object
    coords = torch.cat(
        [
            torch.from_numpy(t_sample[idx].superpose(t_ref,frame=frame).xyz) for frame,idx in enumerate(best_rmsd_indices)
        ],
        dim=0
    )

    # save the indexing, rmsd information 
    best_info = {
        'RMSD':best_rmsds,
        'Index':best_rmsd_indices
    }

    return best_info,coords

def get_gen_filepaths(
    sample_dir,
    region_idx,
    cond_scale,
    rescaled_phi,
    milestone,
    chrom
):
    
    filepath = sample_dir
    if filepath != '' and filepath[-1] != '/':
        filepath+= '/'
    filepath+= f'sample_{region_idx}_{float(cond_scale)}_{float(rescaled_phi)}'
    filepath+= f'_{milestone}_{chrom}'
    return filepath+'.dcd', filepath+'.psf'

def get_tan_filepaths(
    sample_dir,
    chrom,
    region_idx,
    nbins
):
    filepath = sample_dir
    if filepath != '' and filepath[-1] != '/':
        filepath+= '/'
    filepath+= f'chrom_{chrom}_region_{region_idx}_nbins_{nbins}'
    return filepath+'.dcd',filepath+'.psf'

def align_many_samples(
    sample_dir,
    reference_dir,
    *,
    chroms=None,
    milestones=None,
    region_idxs=None,
    cond_scales=None,
    rescaled_phis=None,
    r_c=1.,
    long_scale=1/8,
    use_gpu=True,
    high_precision=False,
):
    sample_types = get_all_sample_types(
        sample_directory=sample_dir,
        region_idx=region_idxs,
        cond_scale=cond_scales,
        rescaled_phi=rescaled_phis,
        milestone=milestones,
        chrom = chroms
    )

    if sample_dir != '' and sample_dir[-1] != '/':
        sample_dir = sample_dir + '/'
    dest_dir = sample_dir + 'aligned/'

    for region_idx, cond_scale, rescaled_phi, milestone, chrom in tqdm(sample_types,desc='Sample Alignment Progress'):
        # Get filenames for generated samples
        gen_dcd,gen_psf = get_gen_filepaths(sample_dir,region_idx,cond_scale,rescaled_phi,milestone,chrom)

        # Check if this sample has already been processed; continue if so
        sample_name = '.'.join(gen_dcd.split('/')[-1].split('.')[:-1])
        info_fp = dest_dir+sample_name+'_loss_info.pkl'
        dest_fp1 = dest_dir + sample_name + '.dcd'
        dest_fp2 = dest_dir + sample_name + '.psf'
        if os.path.exists(info_fp) and os.path.exists(dest_fp1) and os.path.exists(dest_fp2):
            continue

        # Load the generated conformations
        t_gen = md.load(gen_dcd,top=gen_psf)
        t_gen.xyz*= 10

        # Load reference samples
        nbins = t_gen.xyz.shape[-2]
        ref_dcd,ref_psf = get_tan_filepaths(reference_dir,chrom,region_idx,nbins)
        t_ref = md.load(ref_dcd,top=ref_psf)
        t_ref.xyz*=10 

        # Align the data, place configurations into a 
        info,coords = get_best_alignments(t_ref,t_gen,r_c,long_scale,use_gpu,high_precision)

        # Save the information
        coord_to_dcd(coords,dest_dir,sample_name)
        pickle.dump(info,open(info_fp,'wb'))

if __name__ == '__main__':

    align_many_samples(
        '../../data/samples/origami_64_no_embed_reduction/dcd_files/',
        '../../data/samples/Tan/',
        chroms=None,#'1',
        milestones=120,
        region_idxs=None,#330,
        cond_scales=None,#3.0,
        rescaled_phis=None#.5
    )
    
