import pandas as pd
import numpy as np
import torch 

# Should be replaced with relative imports eventually 
import sys
sys.path.insert(1,'/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/code/data_utils_v2/')
from ConfigDataset import ConfigDataset
from EmbeddedRegions import EmbeddedRegions

class DataLoader:

    def __init__(
        self,
        config_ds,
        embedded_regions=None,
        drop_unmatched_pairs=True, # Should we drop the unneeded rows in the embedder object?
        shuffle = True,
        batch_size=64
    ):

        '''
        NOTE: This assumes that any unmatched region lengths have already been removed from the embeddedregions
        object. 
        '''

        # Basic attributes
        self.configs = config_ds
        self.embedded_regions = embedded_regions
        self.shuffle = shuffle
        self.batch_size = batch_size 
        self.epoch = 0 
        self.internal_idx = 0 
        
        # Ease our notation. 
        con, er = self.configs, self.embedded_regions
        
        # Ensure input objects are correct
        msg = ''
        if type(con) != ConfigDataset:
            msg+= f'Expected config_ds to be a ConfigDataset object. Received {type(con)}.'
        if type(er) != EmbeddedRegions and er is not None: 
            if msg != '':
                msg+='\n'
            msg+=f'Expected embeddedregions to be an EmbeddedRegions object or NoneType. Received {type(er)}.'
        if len(er.length_index.unique()) > 1: 
            if msg != '':
                msg+='\n'
            msg+= f'The EmbeddedRegions object must have only one region to match the capabilities of the Coordinates object.'
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

        if not con.two_channels: 
            # Must account for maternal vs paternal copies/duplicates
            idx1 = df['coord_idx'].values
            coord_idx = [(i,'mat') for i in idx1]
            coord_idx.extend([(i,'pat') for i in idx1])
            df = pd.concat([df,df])
            df['coord_idx'] = coord_idx
            del coord_idx 

        # Finally, make the index an attribute of this object and 
        # shuffle the index for the first epoch (if desired). 
        self.index = df
        if shuffle: 
            self.reshuffle()

    @property
    def embed_idx(self):
        return self.index['embed_idx']

    @property
    def coord_idx(self):
        return self.index['coord_idx']

    @property
    def device(self):
        return self.configs.device

    def __len__(self):
        return len(self.index)
        
    def reshuffle(self):
        self.index = self.index.sample(frac=1)

    def reset_index(self):
        self.internal_idx = 0
        self.epoch+= 1
        if self.shuffle:
            self.reshuffle()

    def __next__(self):
        i = self.internal_idx
        j = min(i+self.batch_size,len(self))
        self.internal_idx = j 

        # Get the embeddings (if desired)
        if self.embedded_regions is None:
            embeddings = None
        else:
            embeddings = self.embedded_regions.fetch(self.embed_idx[i:j]).to(self.device) 
            '''
            if j-i == 1: # NOTE TO FUTURE SELF: This should really be done in the EmbeddedRegions class
                embeddings = embeddings.unsqueeze(0) #embeddings.reshape(1,*embeddings.shape)
            else:
                embeddings = torch.stack(embeddings,0)
            '''

        # Get the distance maps 
        dist_maps = self.configs.fetch(self.coord_idx[i:j].tolist())
        
        # Reset the index, if necessary
        if self.internal_idx == len(self):
            self.reset_index()

        return dist_maps, embeddings 
