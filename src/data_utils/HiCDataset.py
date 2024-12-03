'''
Greg Schuette 2023
'''
import os
import cooler
from cooltools.lib.numutils import adaptive_coarsegrain, interp_nan

import sys
sys.path.insert(1,'./') 
from .HiCMap import HiCMap

##############################
# Check validity of cooler filepath. Change .mcool <-> .cool if needed. 

# Ugly code, so use a descriptive function name when this operation is performed. 
flip_cool_mcool = lambda fp: fp[:-4]+'mcool' if fp[-5:]=='.cool' else fp[:-5] + 'cool' 

# Main function
def check_cooler_fp(fp):
    
    ft = fp.split('.')[-1]
    assert ft in ['cool','mcool'], f'The Cooler filepath must end in .mcool or .cool. Received .{ft}'

    if os.path.exists(fp): 
        return fp, ft

    # If here, then the passed file cannot be found. 
    # Check if the file exists with the alternative Cooler type. 
    file_root = fp[:-len(ft)-1]
    alt_ft = 'cool' if ft=='mcool' else 'mcool' 
    alt_fp = file_root + '.' + alt_ft
    assert os.path.exists(alt_fp), 'The specified cooler filepath does not exist.'

    # If here, then the alternative filepath DOES exist
    return alt_fp, alt_ft

##############################
# Initialize the cooler object

def load_cooler(uri):
    '''
    If this is a cooler uri, initialize a Cooler object located at that position. 
    Otherwise, return False. 
    '''
    if cooler.fileops.is_cooler(uri):
        return cooler.Cooler(uri)
    else:
        return False

def get_cooler_resolution(fp):

    if cooler.fileops.is_cooler(fp):
        # Single resolution cooler file 
        return [ cooler.Cooler(fp).info['bin-size'] ]
    elif cooler.fileops.is_multires_file(fp): 
        # Multi-resolution cooler file
        return [int(c.split('/')[-1]) for c in cooler.fileops.list_coolers(fp)]
    else: 
        raise Exception('The provided filepath does not correspond to a Cooler object!')

def init_cooler(fp,resolution): 

    if cooler.fileops.is_multires_file(fp): 
        # Multires cooler file. Must add ending
        fp = fp + f'::/resolutions/{resolution}'

    return cooler.Cooler(fp) 
    
def file_has_resolution(fp,resolution): 
    return resolution in get_cooler_resolution(fp) 

#def zoomify(fp,resolution): 

def get_Cooler(fp,resolution): 

    # Ensure filepath exists. Flip .mcool <-> .cool if necessary
    fp,extension = check_cooler_fp(fp) 

    # If the passed file contains the requested resolution, initialize the cooler
    if file_has_resolution(fp,resolution):
        return fp, init_cooler(fp,resolution)
    
    # If not, check if an alternative file exists (same root name) with the resolution
    fp2 = flip_cool_mcool(fp)
    if cooler.fileops.is_cooler(fp2) or cooler.fileops.is_multires_file(fp2): 
        if file_has_resolution(fp2):
            return fp2, init_cooler(fp2,resolution) 

    # The cooler must be made via the Zoomify function 
    # Deal with that later. 
    raise Exception(f'Cannot find contact data at resolution {resolution} at the indicated filepath')
    

'''
# Todo: Separate this into multiple smaller functions for readability/simplicity. 
# Deprecated version 
def get_Cooler(fp,resolution):

    fp,_ = check_cooler_fp(fp) 

    while True:

        if cooler.fileops.is_cooler(fp): 
            # Single resolution cooler file 
            clr = cooler.Cooler(fp)
            res = clr.info['bin-size']
            if res != resolution: 
                # Check if alternative path exists with multiple resolutions
                #fp2 = fp[:-4] + 'mcool' 
                fp2 = flip_cool_mcool(fp)
                if os.path.exists(fp2) and cooler.fileops.is_multires_file(fp2):
                    fp = fp2 # We'll want to load from the multires file/not overwrite it with less data 
                    continue 
            
            resolutions = [res] 
            
        else: 
            # Multires file 
            assert cooler.fileops.is_multires_file(fp), 'The provided filepath does not correspond to a cooler file!'
            
            resolutions = [int(c.split('/')[-1]) for c in cooler.fileops.list_coolers(fp)]
            if resolution in resolutions: 
                clr = cooler.Cooler(fp+f'::/resolutions/{resolution}')
                res = resolution
            else: 
                res = -1
            
        if res == resolution: 
            return fp, clr 
        print("The desired resolution isn't available in the provided file. Coarsening the data to obtain contact data at the desired resolution.",flush=True)
        print("This might take a while...",flush=True)
        
        # Coarsen the data to the desired resolution. 
        fp_new = fp if fp[-5:] == 'mcool' else fp[:-4] + 'mcool' # in-place if already have multires cooler file
        cooler.zoomify_cooler( # ONLY WORKS IF GIVEN SPECIFIC URI 
            fp,
            fp_new,
            resolutions=resolutions.append(resolution),
            chunksize=1000000
        )
        fp = fp_new 
'''


class HiCDataset:

    def __init__(
        self,
        cooler_fp='../../data/outside/GM12878_hg19.mcool',
        resolution=20000,
    ):

        self.cooler_fp, self.clr = get_Cooler(cooler_fp,resolution)
        self.resolution = resolution

    @property
    def chroms(self):
        return self.clr.bins()[:]['chrom'].unique()

    def fetch(
        self,
        chrom=None,
        start=None,
        stop=None,
        balance=True,
        adaptive_coarsegraining=False, # At least for cooltools <= 0.6.1, this does not work 
                                       # with NumPy >= 24 because np.int was removed. Either load
                                       # with older NumPy version or check if more recent cooltools version has been released. 
                                       # As of 1/5/2023, the relevant bug is located at line 1318 of cooltools/cooltools/lib/numutils.py: 
                                       # https://github.com/open2c/cooltools/blob/8af2b1087f7302d282dc77a82047ac0a3a8339c1/cooltools/lib/numutils.py#L1318C1-L1319C1
        adaptive_coarsegraining_cutoff=3,
        adaptive_coarsegraining_max_levels=8,
        interp_nans=False
    ):
        '''
        Should take the 
        '''
        clr = self.clr

        ''' # Need to convert the mcool file's chromosome definitions from '1', '2', ... to 'chr1', ...
        if type(chrom) == int: 
            chrom = f'chr{chrom}'
        elif type(chrom) == str:
            if (len(chrom) <= 3) or (chrom[:3] != 'chr'):
                chrom = f'chr{chrom}'
        else: 
            raise Exception('invalid chrom identifier')
        '''

        if chrom is None: 
            region = ''
        else: 
            region = str(chrom)
            if start is not None or stop is not None: 
                region+= ':'
                region+= '0' if start is None else str(start)
                region+= '-'
                region+= '' if stop is None else str(stop)

        if adaptive_coarsegraining: 
            if region != '': 
                mat = adaptive_coarsegrain(
                    clr.matrix(balance=True).fetch(region),
                    clr.matrix(balance=False).fetch(region),
                    cutoff=adaptive_coarsegraining_cutoff, 
                    max_levels=adaptive_coarsegraining_max_levels
                )
            else: 
                mat = adaptive_coarsegrain(
                    clr.matrix(balance=True)[:],
                    clr.matrix(balance=False)[:],
                    cutoff=adaptive_coarsegraining_cutoff, 
                    max_levels=adaptive_coarsegraining_max_levels
                )
        elif region != '': 
            mat = clr.matrix(balance=balance).fetch(region)
        else: 
            mat = clr.matrix(balance=balance)[:] 

        if interp_nans: 
            mat = interp_nan(mat)

        return HiCMap(
            mat, 
            chrom=chrom,
            start=start,
            stop=stop,
            includes_self_interaction=True,
        )