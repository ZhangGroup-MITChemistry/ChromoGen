import numpy as np
import torch
import pandas as pd
import random 

import sys
sys.path.insert(1,'./')
from HiCMap import HiCMap 

def reduce_coord_info(coord_info,geos,organisms,cell_types,cell_numbers,chroms,replicates):
    '''
    Reduce the coord_info object to the rows that satisfy all desired restrictions. 
    '''

    key_vals = {
        'Accession':geos,
        'Organism':organisms,
        'Cell_Type':cell_types,
        'Cell':cell_numbers,
        'Replicate':replicates,
        'Chromosome':chroms
    }

    for col, restrictions in key_vals.items():
        assert len(coord_info) > 0, "No data satisfies your desired restrictions."
        if restrictions is None:
            continue
        idx = np.zeros(len(coord_info),dtype=bool)
        vals = coord_info[col].values
        for r in restrictions: 
            idx|= vals == r
        idx = np.where(idx)[0]
        coord_info = coord_info.iloc[idx]

    # Reset the index in the coord_info DataFrame to make indexing it later more straightforward. 
    coord_info.reset_index(drop=True,inplace=True)
    
    return coord_info 

def load_coords(filepath,coord_info):

    index_ranges = [[coord_info['idx_min'].iloc[0],coord_info['idx_max'].iloc[0]+1]]
    for k,row in coord_info.iterrows(): 
        if k == 0:
            continue
        if index_ranges[-1][1] == row['idx_min']:
            index_ranges[-1][1] = row['idx_max']+1
        else: 
            index_ranges.append([row['idx_min'],row['idx_max']+1])

    data = []
    for range in index_ranges: 
        data.append(
            pd.read_hdf(
                filepath,
                key='Coordinates',
                start=range[0],
                stop=range[1]
            )
        )

    data = pd.concat(
        data,
        axis=0,
        ignore_index=True
    )
    data.reset_index(drop=True,inplace=True)
    
    return data

def get_valid_starts(permitted_lengths,segment_length,allow_overlap=False): 
        '''
        Find all indices where, including that bead, there are segment_length
        sequential beads in a row (i.e. not removed during Dip-C clean process)

        If allow_overlap is set to False, the indices returned will represent
        the starting points for non-overlapping regions only
        '''

        pl = permitted_lengths 
    
        # If overlapping regions are fine, simply need segments long enough
        # to satisfy the following 
        if allow_overlap:
            return np.where(pl>segment_length-2)[0]#pl[pl > segment_length-2]

        # A monomer at the extreme end of one segment can appear as the start of the next 
        # segment per this setup 
        perlens_for_no_overlap = np.arange(segment_length-1,pl.max()+.5,segment_length-1)
        perlens_for_no_overlap = perlens_for_no_overlap.reshape(1,len(perlens_for_no_overlap))

        pl = pl.reshape(len(pl),1)
        return np.where( (pl == perlens_for_no_overlap).any(1) )[0]

def reset_coord_info_indices(coord_info):
    
    # Make the indices match the DataFrame containing actual coordinates 
    coord_info.loc[0,'idx_max'] = coord_info.loc[0,'idx_max'] - coord_info.loc[0,'idx_min']
    coord_info.loc[0,'idx_min'] = 0 
    for i in range(1,len(coord_info)):
        length = coord_info.loc[i,'idx_max'] - coord_info.loc[i,'idx_min']
        coord_info.loc[i,'idx_min'] = coord_info.loc[i-1,'idx_max'] + 1
        coord_info.loc[i,'idx_max'] = coord_info.loc[i,'idx_min'] + length

    return coord_info

class ConfigDataset:

    def __init__(
        self,
        filepath,
        segment_length=64,
        batch_size=64,
        normalize_distances=True,
        geos=None,
        organisms=None,
        cell_types=None,
        cell_numbers=None,
        chroms=None,
        replicates=None,
        shuffle=True,
        allow_overlap=False,
        two_channels=False,
        try_GPU=True,
        mean_dist_fp='../../data/mean_dists.pt',
        mean_sq_dist_fp='../../data/squares.pt'
    ):
        '''
        filepath: Dataset location. Should be formatted as designed for this study
        segment_length: The number of monomers relevant to the distance maps 
        batch_size: Number of configurations per batch 
        shuffle: Should the sample indices be shuffled before each epoch? 
        allow_overlap: Can the dataset include overlapping regions (True), 
                       or should they all be fully independent (False)? 
        two_channels: Does the data include the maternal & paternal structures
                        in the same sample (two channels of the image)? Otherwise, 
                        returns one or the other. 

        To choose a subset of the overall dataset, use the following variables. In all cases, 
        a value of None means no restriction on this parameter. Otherwise, a list of those 
        parameters to be INCLUDED should be provided. 
            1. geos: GEO accession numbers
            2. organisms: Organism, e.g. 'Human' or 'Mouse'
            3. cell_types: Cell type 
            4. cell_numbers: Cell number within a dataset specified above. Should be np.int64
            5. chroms: Chromosomes. Should be passed as a string. 
            6. replicates: Replicates from Dip-C procedure on the same cell's data
        '''

        # Assign qualities where relevant 
        self.filepath = filepath 
        self.seg_len = segment_length 
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.allow_overlap = allow_overlap
        self.two_channels = two_channels
        self.norm_dists = normalize_distances

        # torch.cuda.is_available() doesn't work properly on 
        # SuperCloud, so use this approach instead. 
        try: 
            assert try_GPU
            self.device = torch.empty(1).cuda().device
        except:
            self.device = torch.empty(1).device

        # Load the information object to help us decide which portion of the dataset to load from storage
        coord_info = pd.read_hdf(
            self.filepath,
            key='coord_info'
        )

        # Find the indices to load from memory
        coord_info = reduce_coord_info(coord_info,geos,organisms,cell_types,cell_numbers,chroms,replicates)

        # Fetch the desired rows from memory
        coord_df = load_coords(filepath,coord_info)

        # Place information from the coord_df object into objects which improve speed downstream 
        self.coords = torch.from_numpy(coord_df[['mat_x','mat_y','mat_z','pat_x','pat_y','pat_z']].values).to(torch.double)
        self.genomic_index = coord_df['Genomic_Index'].values

        # Get the indices of valid starting positions to obtain uninterrupted regions of the referenced dimensions
        self.start_indices = get_valid_starts(coord_df['Permitted_Lengths'].values,segment_length,allow_overlap)

        # Set the indices of the coord_info object to match the loaded dataset
        self.coord_info = reset_coord_info_indices(coord_info)

        # Track the sample indices. This can be shuffled without perturbing the main dataset 
        #self.data_index = 
        if two_channels: 
            self.data_index = self.start_indices
        else: 
            n = len(self.start_indices)
            self.data_index = np.empty((2*n,2),dtype=np.int64)
            self.data_index[:n,0] = self.start_indices # Rows in coords object
            self.data_index[:n,1] = 0 # Maternal is in columns 0,1,2
            self.data_index[n:,0] = self.start_indices 
            self.data_index[n:,1] = 3 # Paternal is in columns 3,4,5

        # Define some internal indexing objects to keep track of the dataset
        self.epoch = 0     # Which epoch are we on? 
        self.inner_idx = 0 # Keep track of which row in the dataset we're considering
        self.reset_index() # This will shuffle the index (if desired) and increase epoch to 1
        self.triu_indices = torch.triu_indices(segment_length,segment_length,1) 

        # The following is used to index the distance objects, which are indexed such that
        # sep[i] corresponds to i+1, so subtract the 1
        self.sep_idx = self.triu_indices[1] - self.triu_indices[0] - 1  
        

        # Initialize some objects used during batch fetching/manipulation
        self.batch_coords = torch.empty(self.batch_size,self.seg_len,3*(1+int(self.two_channels)),
                                        device=self.device,dtype=self.coords.dtype)
        self.batch_dists = torch.empty(self.batch_size,1+int(self.two_channels),self.seg_len-1,self.seg_len-1,
                                      device=self.device,dtype=torch.float)

        # Load the distance vs genomic separation relationships used to normalize the distance data
        # for use in the signmoid mod. Afterwards, process the values we use at each iteration
        if normalize_distances:
            dt = self.batch_coords.dtype
            mean_dist = torch.load(mean_dist_fp,map_location=self.device).flatten()[:self.seg_len].to(dt)
            mean_square_dist = torch.load(mean_sq_dist_fp,map_location=self.device).flatten()[:self.seg_len].to(dt)
            self.dist_std = (mean_square_dist - mean_dist**2).sqrt()
            self.inv_beta = torch.sqrt( 2*mean_square_dist/3 )
            self.inv_beta_sigmoid = torch.sigmoid( -self.inv_beta/self.dist_std )
            self.complement_inv_beta_sigmoid = 1 - self.inv_beta_sigmoid

    def reset_index(self):

        if self.shuffle: 
            n_unused = len(self) - self.inner_idx
            if n_unused > 0 and n_unused < self.batch_size: 
                # Place the unused data at the front, if the epoch
                # hasn't *totally* completed but is less than a 
                # batch length away. 
                temp = self.data_index[-n_unused:,...].copy()
                idx = np.arange(len(self)-n_unused)
                self.data_index[n_unused:,...] = self.data_index[idx,...]
                self.data_index[:n_unused,...] = temp 
            else:
                idx = np.arange(self.data_index.shape[0])
                random.shuffle(idx)
                self.data_index = self.data_index[idx,...]

        self.epoch+=1
        self.inner_idx = 0

    def normalize_dists(self,dists):
        if not self.norm_dists:
            return dists
        sep = self.sep_idx
        i,j = self.triu_indices
        bs = dists.shape[0] #self.batch_size
        j = j-1
        dists-= self.inv_beta[sep].repeat(bs,1) # Should eventually replace with expand to save memory 
        dists/= self.dist_std[sep].repeat(bs,1)
        dists.sigmoid_()
        dists-= self.inv_beta_sigmoid[sep].repeat(bs,1)
        dists/= self.complement_inv_beta_sigmoid[sep].repeat(bs,1)
        return dists 

    def get_genomic_regions(self):
    
        coord_info = self.coord_info
        start_indices = self.start_indices
        gen_idx = self.genomic_index
        nbeads = self.seg_len
    
        regions = pd.DataFrame({
            'Start':gen_idx[start_indices]
        })
        regions['Stop'] = regions['Start'] + (gen_idx[start_indices[0]+1]- gen_idx[start_indices[0]]) * nbeads
        regions.insert(0,'Chromosome','')
    
        for _,row in coord_info.iterrows():
            idx = (start_indices >= row['idx_min']) & (start_indices <= row['idx_max'])
            if idx.any(): 
                regions.loc[idx,'Chromosome'] = row['Chromosome']
    
        regions = regions.drop_duplicates(ignore_index=True)#,inplace=True)
        
        regions = regions.sort_values(by=['Chromosome','Start'],axis=0,ignore_index=True) 
        
        return regions

    def _fetch_one_coord_(
        self, 
        chrom,
        start_idx,
        cell,
        replicate
    ):

        inner_min = self.coord_info[
            (self.coord_info.Cell == cell) & 
            (self.coord_info.Chromosome == chrom) &
            (self.coord_info.Replicate == replicate)
        ]
        if len(inner_min) == 0: # The selection is invalid 
            return False, False

        inner_min, inner_max = int(inner_min.idx_min.values), int(inner_min.idx_max.values)

        temp_gen_idx = self.genomic_index[inner_min:inner_max]
        if temp_gen_idx[-1] < start_idx: 
            return False, False # The requested region is out of range
        
        start = np.where(
            temp_gen_idx == start_idx
        )[0] + inner_min

        if len(start) == 0:
            return False, False # The index doesn't exist 
        start = int(start) # for indexing below

        if (start != self.start_indices).all(): 
            return False, False # The region would be incomplete 

        mat, pat = self.coords[start:start+self.seg_len,:3], self.coords[start:start+self.seg_len,3:]
        
        return mat, pat
    
    def fetch_specific_coords(
        self,
        chrom,           # Chromosome of interest. Should be in string form, i.e. '1', '2', '3', etc. 
        start_idx,       # Integer representing the starting position within the context (in bp) 
        *, 
        cells=None,      # If None, ignore cell number & return all that are relevant. Otherwise, str or list of strs. 
        replicates=None, # If None, ignore replicate number & return this region everwhere it's valid. Otherwise, int or list of ints. 
        lineage=None,    # Paternal vs maternal copy. Either None (return both), 'mat' (return maternal copy), or 'pat' (return paternal copy). 
    ):
        '''
        If a single value is requested, i.e., cells/replicates/lineage are all specified singularly, return 
        the coordinates (if they're valid) or False (if the requested region is nonexistent or non-continuous). 

        Otherwise, return a pandas DataFrame with columns specifying which cell, replicate, and lineage each entry corresponds to in
        a torch tensor with size BxNx3 (B==batch size/number of regions of this type that could be fetched, N=number beads, 3 for x/y/z coords). 
        '''
        
        # Format the selections
        chrom = str(chrom) 
        start_idx = int(start_idx) 
        
        all_specified = True
        if cells is None: 
            all_specified = False
            cells = list(self.coord_info.Cell.unique())
        else: 
            if type(cells) == list: 
                cells = [str(cell) for cell in cells]
            else: 
                cells = [str(cells)]
        if replicates is None: 
            all_specified = False
            replicates = list(self.coord_info.Replicate.unique())
        else: 
            if type(replicates) == list: 
                replicates = [int(replicate) for replicate in replicates]
            else: 
                replicates = [int(replicates)]
        if lineage is None: 
            lineage = ['mat','pat']
            all_specified = False
        else:
            assert lineage in ['mat','pat'], f"lineage must equal 'mat' or 'pat', but received {lineage}"

        # If one specific region was requested, provide it without the pd DataFrame 
        if len(cells)==1 and len(replicates) == 1 and lineage in ['mat','pat']:
            return self._fetch_one_coord_(chrom,start_idx,cells[0],replicates[0])[int(lineage=='pat')]

        cell_nums = []
        replicate_nums = []
        lineage_nums = []
        coords = []
        for cell in cells: 
            for replicate in replicates:
                mat, pat = self._fetch_one_coord_(chrom,start_idx,cell,replicate)
                if type(mat) != torch.Tensor:  # It's False
                    continue
                
                for lin,data in [('mat',mat),('pat',pat)]: 
                    if lin in lineage: 
                        cell_nums.append(cell)
                        replicate_nums.append(replicate)
                        lineage_nums.append(lin)
                        coords.append(data)

        if len(cell_nums) == 0:
            return None, None 
        
        coords = torch.stack(coords,dim=0) 
        info = pd.DataFrame(
            {
            'Cell':cell_nums,
            'Replicate':replicate_nums,
            'Lineage':lineage_nums
            },
            index = np.arange(len(cell_nums))
        )

        return info, coords
        

    def fetch_coords(self,start_idx): 
        
        if type(start_idx) == tuple: 
            #assert start[1] in ['mat','pat']
            y = start_idx[0]
            x_idxs = [3*(start_idx[1] == 'pat')]
        else:
            y = start_idx
            x_idxs = [0,3] 

        coords = []
        for x_idx in x_idxs: 
            coords.append(self.coords[y:y+self.seg_len,x_idx:x_idx+3])

        return torch.stack(coords,dim=0)
            

    def fetch(self,start_indices):

        # Get the relevant coordinates
        if type(start_indices) == int or type(start_indices) == tuple: 
            start_indices = [start_indices]

        coords = torch.stack([self.fetch_coords(i) for i in start_indices],dim=0).to(self.device)

        # Compute distances; ignore duplicates
        i,j = self.triu_indices
        dists = torch.cdist(coords,coords)[...,i,j]

        # Normalize distances (if desired) 
        if self.norm_dists: 
            s = dists.shape
            dists = self.normalize_dists(dists.flatten(0,-2)).to(self.batch_dists.dtype).reshape(s)

        b,c,h = dists.shape[0],dists.shape[1],self.seg_len-1
        batch = torch.empty((b,c,h,h),dtype=torch.float,device=self.device)
        j = j-1
        batch[:,:,i,j] = dists.to(torch.float)
        batch[:,:,j,i] = dists.to(torch.float)
        
        return batch 
    
    def __len__(self):
        return self.data_index.shape[0]
    
    def __next__(self):

        # Avoid out of range issues
        if self.inner_idx + self.batch_size >= len(self):
            self.reset_index()

        # Get the section of the main dataset we're pulling from 
        idx = self.data_index[self.inner_idx:self.inner_idx+self.batch_size,...]
        
        # Get the distance map associated with the inquired regions
        if self.two_channels:
            self.batch_coords[:] = self.coords[idx,:].to(self.device).reshape_as(self.batch_coords)
        else:
            for i in range(self.batch_size):
                i0 = idx[i,0]
                i1 = idx[i,1]
                self.batch_coords[i,:,:] = self.coords[i0:i0+self.seg_len,i1:i1+3]
                #self.batch_coords[sub_idx,i,:] = self.coords[idx[sub_idx,0],:3].to(self.device).reshape_as(self.batch_coords[sub_idx,:,:])
                #sub_idx^= True # Flips to paternal index
                #self.batch_coords[sub_idx,i,:] = self.coords[idx[sub_idx,0],3:].to(self.device).reshape_as(self.batch_coords[sub_idx,:,:])

        i,j = self.triu_indices
        for k in range(self.batch_dists.shape[1]):
            ii = 3*k
            jj = ii+3
            dists = torch.cdist(self.batch_coords[:,:,ii:jj],self.batch_coords[:,:,ii:jj])[:,i,j]
            self.batch_dists[:,k,i,j-1] = self.normalize_dists(dists).to(self.batch_dists.dtype) 
        self.batch_dists[:,:,j-1,i] = self.batch_dists[:,:,i,j-1]

        self.inner_idx+= self.batch_size 

        return self.batch_dists 
            
        
        