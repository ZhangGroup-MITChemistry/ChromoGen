import pandas as pd

def load_embeddings(directory,chroms):
    '''
    chroms: Either None or list of chromosomes to load. 
    '''

    if chroms is None: 
        chroms = [f'{k}' for k in range(1,23)]
        chroms.append('X')

    if directory[-1] != '/':
        directory = directory + '/' 

    filepath = lambda chrom: directory + f'chrom_{chrom}.tar.gz'

    dfs = []
    for chrom in chroms: 
        dfs.append( pd.read_pickle(filepath(chrom)) )

    return pd.concat(dfs)


class EmbeddedRegions:

    def __init__(
        self,
        directory,
        chroms=None
    ):
        self.data = load_embeddings(directory,chroms)

    @property
    def index(self):
        return self.data.index

    @property
    def chrom_index(self):
        return self.index.get_level_values('Chromosome')

    @property
    def genomic_index(self):
        return self.index.get_level_values('Genomic_Index')

    @property
    def length_index(self):
        return self.index.get_level_values('Region_Length')

    def __len__(self):
        return len(self.data) 

    def drop(
        self,
        index
    ): 
        # Doing this in place is WAY faster than out of place. 
        self.data.drop(index=index,inplace=True)

    def append(
        self,
        new_embedded_regions # EmbeddedRegions object or iterable full of them
    ):
        y = new_embedded_regions

        if type(y) == EmbeddedRegions:
            self.data = pd.concat([self.data,y.data])
        elif hasattr(y, '__iter__'):
            for i,obj in enumerate(y):
                assert type(obj) == EmbeddedRegions, \
                f'Expected iterable to contain EmbeddedRegions objects, but it contains {type(obj)} at position {i}.'
            self.data = pd.concat([self.data,*[yy.data for yy in y]])
        else:
            raise Exception('EmbeddedRegions.append must receive another EmbeddedRegions object or an iterable of them')

    def fetch(
        self,
        index
    ):
        if len(index) > 1:
            return list(self.data.loc[index,'Data'].values)
        return self.data.loc[index,'Data']

    def ifetch(
        self,
        idx 
    ):
        '''
        Return the data at a specific index
        '''
        return self.data.iloc[idx].values
    