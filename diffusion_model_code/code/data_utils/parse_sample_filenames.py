import os
import pandas as pd 

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

    
    