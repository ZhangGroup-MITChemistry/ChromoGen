import pandas as pd 
import numpy as np
import torch

def remove_inter_chromosomal_interactions(df):
    df = df[df['Chrom0'] == df['Chrom1']]
    #df = df[df['Haplotype0']==df['Haplotype1']]
    df = df.drop(
        ['Chrom1'],#,'Haplotype1'],
        axis = 1
    ).rename(
        columns={
            'Chrom0':'Chromosome'#,
            #'Haplotype0':'Haplotype'
        }
    )
    return df

def remove_ignored_values(df,key,items_to_keep):

    all_items = df[key]
    indices_to_keep = np.zeros(len(all_items),dtype=bool)
    for item in items_to_keep: 
        indices_to_keep|= all_items == item

    return df[indices_to_keep]

''' 
# May eventually be helpful, but for now, give the full scHi-C dataset and try to predict both structures
def refine_paternal_maternal(df):

    # Remove contacts where the parentage is unknown
    return df
'''

def divide_into_dicts(df):
    as_dict = {}
    for cell in df['Cell'].unique():
        as_dict[cell] = {}
        temp = df[df['Cell'] == cell]
        for chrom in temp['Chromosome'].unique():
            as_dict[cell][chrom] = temp[temp['Chromosome'] == chrom][['Genomic_Index0','Genomic_Index1']].values
    return as_dict

class scHiCDataset:

    def __init__(
        self,
        filepath='../../data/processed_scHiC.hdf5',
        resolution=20_000,
        chroms = None, # Otherwise, string or iterable of strings
        cell_type = 'GM12878',
        cells = None 
    ):
        self.resolution = resolution
        # Load the file
        df = pd.read_hdf(filepath)
        
        # Could perhaps make this optional later, but for now, always remove interchromosomal interactions
        df = remove_inter_chromosomal_interactions(df)

        # If certain values are to be ignored, do that here
        for key,values_to_keep in [('Chromosome',chroms),('Cell_Type',cell_type),('Cell',cells)]:
            if values_to_keep is None:
                continue
            if type(values_to_keep) in [int,str]:
                values_to_keep = [values_to_keep]
            df = remove_ignored_values(df,key,values_to_keep)

        # Bin the data
        for key in ['Genomic_Index0','Genomic_Index1']: # NOTE: genomic_index0 <= genomic_index1 everywhere already
            df[key]//= resolution
        
        # Sort the DataFrame
        df = df.sort_values(
            ['Cell','Chromosome','Genomic_Index0','Genomic_Index1']
        )

        # Divide the DataFrame into dictionaries for faster searching
        self.data = divide_into_dicts(df) 

    def get_scHiC_map(self,cell,chrom,start_idx,stop_idx,is_genomic_index=True):
        if is_genomic_index:
            start_idx = start_idx // self.resolution
            stop_idx = stop_idx // self.resolution
        cell = int(cell)
        chrom = str(chrom)

        # Get the relevant data
        c = self.data[cell][chrom]
        idx = (c[:,0] >= start_idx) & (c[:,1] < stop_idx)
        i,j = c[idx,:].T - start_idx

        # Initialize the map 
        n = stop_idx - start_idx
        scHiC_map = torch.zeros(n,n,dtype=int)

        # Populate the map with 1's where contacts were observed 
        scHiC_map[i,j] = 1
        scHiC_map[j,i] = 1

        return scHiC_map
        
        

