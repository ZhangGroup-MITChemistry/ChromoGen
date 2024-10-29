import pandas as pd
import numpy as np
import torch 

# Should be replaced with relative imports eventually 
import sys
sys.path.insert(1,'/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/code/data_utils_v2/')
from ConfigDataset import ConfigDataset
#from EmbeddedRegions import EmbeddedRegions
from HiCDataset import HiCDataset

def load_processed_hic(folder,hic_maps_dict):
    
    for chrom in hic_maps_dict: 
        try: 
            hic_maps_dict[chrom] = pd.read_pickle(folder+f'/hic_data_chrom_{chrom}.pkl')[chrom]
        except:
            print(f'Preprocessed Hi-C data for chromosome {chrom} data is not available in directory {folder}.')
    return hic_maps_dict 

class DataLoader:

    def __init__(
        self,
        config_ds,
        exp_hic,
        drop_unmatched_pairs=True, # Should we drop the unneeded rows in the embedder object?. Don't think it does anthing here
        shuffle = True,
        batch_size=64,
        resolution=20_000,
        interp_hic_nans=False,
        processed_hic_folder='../../data/processed_hic'
    ):

        '''
        NOTE: This assumes that any unmatched region lengths have already been removed from the embeddedregions
        object. 
        '''

        # Basic attributes
        self.configs = config_ds
        self.exp_hic = exp_hic
        self.shuffle = shuffle
        self.batch_size = batch_size 
        self.epoch = 0 
        self.internal_idx = 0 
        self.resolution = resolution
        self.interp_hic_nans=interp_hic_nans 

        # Ease our notation. 
        con, hic = self.configs, self.exp_hic
        
        # Ensure input objects are correct
        msg = ''
        if type(con) != ConfigDataset:
            msg+= f'Expected config_ds to be a ConfigDataset object. Received {type(con)}.'
        if type(hic) != HiCDataset:
            if msg != '':
                msg+= '\n'
            msg+= f'Expected exp_hic to be of type HiCDataset. Received {type(hic)}.'
        #if type(er) != EmbeddedRegions and er is not None: 
        #    if msg != '':
        #        msg+='\n'
        #    msg+=f'Expected embeddedregions to be an EmbeddedRegions object or NoneType. Received {type(er)}.'
        #if len(er.length_index.unique()) > 1: 
        #    if msg != '':
        #        msg+='\n'
        #    msg+= f'The EmbeddedRegions object must have only one region to match the capabilities of the Coordinates object.'
        if msg!= '':
            raise Exception(msg) 
        
        # Combine the relevant information from con into a DataFrame for easy manipulation/
        df = pd.DataFrame()
        df['coord_idx'] = con.start_indices # Starting position, first index position of con.coords
        df['Chromosome'] = ''
        df['Genomic_Index'] = con.genomic_index[con.start_indices] # Starting position, genomic index of chromosome
        
        # Fill in the chromosome position
        s = df['coord_idx'] 
        todo = np.ones(len(df),dtype=bool) # Saves some time with comparisons in the loop
        for _,row in con.coord_info.iterrows():
            idx = todo & (s <= row['idx_max']) & (s >= row['idx_min'])
            df.loc[idx,'Chromosome'] = row['Chromosome'] 
            todo&= ~idx

        # Handle the EmbeddedRegions object's indices, if it exists
        '''
        if er is not None: 
            # Place the data in a DataFrame for easy comparison with the configuration data
            df2 = pd.DataFrame()
            df2['Chromosome'] = er.chrom_index.values
            df2['Genomic_Index'] = er.genomic_index.values

            # Free memory by dropping the unused embeddings
            if drop_unmatched_pairs: 
                #drop_idx = er.index[df.merge(df2,indicator=True,how='right')['_merge'] == 'right_only']
                drop_idx = df2.merge(df,indicator=True,how='left')
                drop_idx = drop_idx[ drop_idx['_merge'].values == 'left_only' ][['Chromosome','Genomic_Index']].drop_duplicates()
                drop_idx = [(er.length_index[0],*drop_idx.iloc[i].values) for i in range(len(drop_idx))]
                self.embedded_regions.drop(index=drop_idx) # Pretty sure we can just do er.drop, but pointer vs not always gets me...
                del drop_idx
        
            # This reduces our choices to the regions present in both objects 
            df = df.merge(df2,how='inner')
            del df2

            # Get the index for embedding
            length = er.length_index[0]
            embed_idx = np.array([
                [length for k in range(len(df))],
                df['Chromosome'].values,
                df['Genomic_Index'].values
            ]).T
            df['embed_idx'] = pd.MultiIndex.from_tuples(
                list(map(tuple,embed_idx)),
                names=list(er.index.names)
            )
            del embed_idx
        '''

        if not con.two_channels: 
            # Must account for maternal vs paternal copies/duplicates
            idx1 = df['coord_idx'].values
            coord_idx = [(i,'mat') for i in idx1]
            coord_idx.extend([(i,'pat') for i in idx1])
            df = pd.concat([df,df])
            df['coord_idx'] = coord_idx
            del coord_idx 

        # this will store maps after their first load to avoid the overhead associated with loading/processing HiC maps
        self.hic_maps = {
            chrom:{} for chrom in df.Chromosome.unique()
        }
        self.hic_maps = load_processed_hic(processed_hic_folder,self.hic_maps)
        
        ##################
        # Band-aid to avoid issues with regions that couldn't be loaded from the .mcool file.
        # Simply remove indices to regions not present in the processed hic data!
        for _,item in self.hic_maps[[*self.hic_maps.keys()][0]].items():
            self.null_hic_map = torch.empty_like(item).fill_(-1)
            break
        ''' SUPER SLOW, so just return null map instead when an invalid region is crossed
        df = df.reset_index(drop=True) # to minimize the risk of issues here
        to_drop = []
        for row_idx,row in df.iterrows():
            chrom,genomic_index = row[['Chromosome','Genomic_Index']].values
            try:
                _ = self.hic_maps[chrom][genomic_index]
            except:
                to_drop.append(row_idx)
        if len(to_drop) > 0:
            df = df.drop(index=to_drop).reset_index(drop=True)
        '''

        # Finally, make the index an attribute of this object and 
        # shuffle the index for the first epoch (if desired). 
        self.index = df
        if shuffle:
            self.reshuffle()
    '''
    @property
    def embed_idx(self):
        return self.index['embed_idx']
    '''
    @property
    def coord_idx(self):
        return self.index['coord_idx']

    @property
    def device(self):
        return self.configs.device

    @property
    def dtype(self):
        return self.configs.batch_dists.dtype

    @property
    def nbeads(self):
        return self.configs.seg_len

    def __len__(self):
        return len(self.index)
        
    def reshuffle(self):
        self.index = self.index.sample(frac=1)

    def reset_index(self):
        self.internal_idx = 0
        self.epoch+= 1
        if self.shuffle:
            self.reshuffle()

    def load_hic_map(self,chrom,start_idx):
        probs = self.exp_hic.fetch(
            chrom=chrom,
            start=start_idx,
            stop=start_idx + self.nbeads * self.resolution,
            interp_nans = self.interp_hic_nans
        ).prob_map.to(device=self.device,dtype=self.dtype)
        probs[probs.isnan()] = -1
        return probs

    def get_hic_map(self,chrom,start_idx):
        try: 
            return self.hic_maps[chrom][start_idx].to(self.device)
        except: # ~~should~~ never happen now that the configs w/o loaded hic data are removed 
            # Except now this is called since I got rid of the cleaning process
            #return self.null_hic_map.to(self.device)
            #''' 
            hic_map = self.load_hic_map(chrom,start_idx)
            i,j = torch.triu_indices(hic_map.shape[-1],hic_map.shape[-1],0)
            self.hic_maps[chrom][start_idx] = hic_map[i,j].reshape(1,1,len(i)) 
            return self.hic_maps[chrom][start_idx].to(self.device)
            #'''
    def __next__(self):
        i = self.internal_idx
        j = min(i+self.batch_size,len(self))
        self.internal_idx = j         
        '''
        # Get the embeddings (if desired) 
        if self.embedded_regions is None:
            embeddings = None
        else:
            embeddings = self.embedded_regions.fetch(self.embed_idx[i:j]).to(self.device) 
            ''#'
            if j-i == 1: # NOTE TO FUTURE SELF: This should really be done in the EmbeddedRegions class
                embeddings = embeddings.unsqueeze(0) #embeddings.reshape(1,*embeddings.shape)
            else:
                embeddings = torch.stack(embeddings,0)
            ''#'
        '''
        
        # Get the embeddings (AKA Hi-C maps) 
        hic_maps = []
        for ii in range(i,j):
            hic_maps.append(
                self.get_hic_map(
                    #*self.index[['Chromosome','Genomic_Index']].iloc[ii].values
                    *self.index.iloc[ii][['Chromosome','Genomic_Index']].values
                )
            )
            #hic_maps.append(
            #    self.get_hic_map(
            #        chrom = self.index['Chromosome'][ii],
            #        start_idx = self.index['Genomic_Index'][ii]
            #    )
            #)
        hic_maps = torch.stack(hic_maps,dim=0)#.unsqueeze(1)
        
        # Get the distance maps
        # NOTE: self.coord_idx[0:1] returns data associated with self.coord_idx.iloc[0], which is what we want
        # HOWEVER, self.coord_idx[0] returns data associated with self.coord_idx.loc[0], which is NOT what we want
        dist_maps = self.configs.fetch(self.coord_idx[i:j].tolist())
        
        # Reset the index, if necessary
        if self.internal_idx == len(self):
            self.reset_index()
        

        return dist_maps, hic_maps
