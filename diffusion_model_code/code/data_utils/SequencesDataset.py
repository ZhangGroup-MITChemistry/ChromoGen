import os
import numpy as np
import torch
import pandas as pd
from scipy.sparse import load_npz

##########
# Load functions for the genome. Adapted from what Zhuohan sent me 
def pad_seq_matrix(matrix, pad_len=300):
    paddings = np.zeros((1, 4, pad_len)).astype('int8')
    dmatrix = np.concatenate((paddings, matrix[:, :, -pad_len:]), axis=0)[:-1, :, :]
    umatrix = np.concatenate((matrix[:, :, :pad_len], paddings), axis=0)[1:, :, :]
    return np.concatenate((dmatrix, matrix, umatrix), axis=2)
    
def load_ref_genome(fp):
    #try:
    #    ref_gen_data = load_npz(fp).toarray().reshape(4, -1, 1000).swapaxes(0, 1)
    #except:
    #    print(fp) 
    #    return []
    ref_gen_data = load_npz(fp).toarray().reshape(4, -1, 1000).swapaxes(0, 1)
    return torch.tensor(pad_seq_matrix(ref_gen_data))

##########
# Load functions for DNase data. Adapted from what Zhuohan sent me 
def pad_signal_matrix(matrix, pad_len=300):
    paddings = np.zeros(pad_len).astype('float32')
    dmatrix = np.vstack((paddings, matrix[:, -pad_len:]))[:-1, :]
    umatrix = np.vstack((matrix[:, :pad_len], paddings))[1:, :]
    return np.hstack((dmatrix, matrix, umatrix))
    
def load_dnase(fp,chroms):
    chroms_ = [int(chrom[3:]) if chrom[3:].isnumeric() else chrom[3:] for chrom in chroms]
    dnase_seq_ = pd.read_pickle(fp)
    dnase_seq = {}
    for chrom in chroms: 
        chr = int(chrom[3:]) if chrom[3:].isnumeric() else chrom[3:]
        dnase_seq[chrom] = dnase_seq_[chr]
    del dnase_seq_
    
    for chr in dnase_seq: 
        dnase_seq[chr] = torch.tensor(np.expand_dims(pad_signal_matrix(dnase_seq[chr].toarray().reshape(-1, 1000)), axis=1))

    return dnase_seq

##########
# Class by Greg
class SequencesDataset:

    def __init__(
        self,
        cell_type = 'GM12878',
        alignment = 'hg19',
        data_dir = '../../data/',
        resolution=20_000,
        region_length=1_300_000,
        batch_size=64,
        chroms = None
    ):
        self.dnase_fp = data_dir + f'outside/{cell_type}_{alignment}.pkl' 
        self.res = resolution // 1000
        self.length = region_length // 1000
        self.batch_size = batch_size

        if chroms is None: 
            chroms = [f'chr{k}' for k in [*range(1,23),'X']]
        elif type(chroms) == str:
            chroms = [chroms] 
        elif type(chroms) == int: 
            chroms = [f'chr{chroms}']
        '''
        else: 
            raise Exception('More work for greg to do here for generalization')
            # Probably use an external function to do the two elif statements & iterate through them here
        '''
        self.chroms = chroms
        
        # Ensure all files exist 
        self.genome_fps = {}
        for chrom in chroms: 
            self.genome_fps[chrom] = data_dir + f'outside/{alignment}/{chrom}.npz'
            assert os.path.exists(self.genome_fps[chrom]), self.genome_fps[chrom]

        assert os.path.exists(self.dnase_fp), self.dnase_fp

        # Load the data
        print(f'Loading sequencing data')
        self.load_genome()
        self.load_dnase()
        print(f'Sequencing data loading complete')

        # Prepare to iterate
        self.curr_chrom = 0
        self.curr_idx = 0 
        self.inner_idx = 0 

    ##################################################
    # Loading functions
    def load_genome(self):
        self.genome = {}
        for chrom,fp in self.genome_fps.items():
            self.genome[chrom] = load_ref_genome(fp)

    def load_dnase(self):
        self.dnase = load_dnase(self.dnase_fp,self.chroms)

    ##################################################
    # Data loading functions 
    def _update_sample_idx_(self):
        
        self.curr_idx+= self.res

        c = self.chroms[self.curr_chrom]
        if self.curr_idx + self.length >= seq_ds.genome[c].shape[0]:
            self.curr_idx = 0
            self.curr_chrom = (self.curr_chrom+1) % len(self.chroms)

    def _get_sample_(self,idx=None):
        if idx is None:
            c,i,l = self.chroms[self.curr_chrom],self.curr_idx,self.length
        else:
            c,i,l = idx # uses kb resolution for i,l (see self.fetch for comparison) 

        ref_seq = self.genome[c][i:i+l,...]
        dnase_seq = self.dnase[c][i:i+l,...]

        return ref_seq, dnase_seq 
    
    def _region_is_valid_(self):

        ref_seq, dnase_seq = self._get_sample_()

        return ~(ref_seq==0).all(1).any(), ref_seq, dnase_seq
        
    def __iter__(self):
        self.curr_chrom = 0
        self.curr_idx = 0
        self.inner_idx = 0
        return self
    
    def __next__(self):

        if (self.curr_chrom == 0) and (self.curr_idx == 0) and self.inner_idx > 0:
            self.inner_idx == 0 
            raise StopIteration # Back to the start!

        batch_ref_seq = []
        batch_dnase_seq = []
        while len(batch_ref_seq) < self.batch_size:

            is_valid, ref_seq, dnase_seq = self._region_is_valid_()
            if is_valid:
                batch_ref_seq.append(ref_seq)
                batch_dnase_seq.append(dnase_seq)
                self.inner_idx+= 1

            self._update_sample_idx_()

            if (self.curr_chrom == 0) and (self.curr_idx == 0):
                # Went back to the start. This epoch is over! 
                break

        if len(batch_ref_seq) == 0:
            # We must have broken out of the while loop before finding any valid samples
            self.inner_idx == 0 
            raise StopIteration

        # Stack the subobjects as desired.
        batch_ref_seq = torch.stack(batch_ref_seq,dim=0) 
        batch_dnase_seq = torch.stack(batch_dnase_seq,dim=0) 
        
        return batch_ref_seq, batch_dnase_seq

    # For easier interfacing with my other classes. 
    def _fetch_one_(self,idx):
        '''
        idx = tuple(chrom,start_idx (in bp), region length (in bp))
        '''
        c,s,l = idx

        assert s%1000 == 0 
        s//= 1000
        assert l%1000 == 0
        l//= 1000
        if c not in self.chroms:
            c = f'chr{c}'
        assert c in self.chroms

        return self._get_sample_(idx=(c,s,l))
        #ref_seq, dnase_seq = self._get_sample_(idx=(c,s,l))
        #return ref_seq, dnase_seq
        
    def is_valid(self,idx):
        '''
        idx = same as described in self._fetch_one_
        '''
        ref_seq, dnase_seq = self._fetch_one_(idx)
        
        return ~(ref_seq==0).all(1).any(), ref_seq, dnase_seq
    
    def fetch(self,idxs):
        '''
        idxs is a list of tuples with the shape desribed for idx in the self._fetch_one_ function above.
        '''

        seqs = []
        for idx in idxs: 
            seqs.append(torch.cat(self._fetch_one_(idx),dim=-2))

        return torch.stack(seqs,dim=0) 
            
        
        '''
        ref_seqs = []
        dnase_seqs = []
        idxs_return = []
        for idx in idxs: 
            is_valid, ref_seq, dnase_seq = self.is_valid(idx) 
            if return_invalid or is_valid: 
                idxs_return.append(idx)
                ref_seqs.append(ref_seq)
                dnase_seqs.append(dnase_seq)

        ref_seqs = torch.stack(ref_seqs,dim=0)
        dnase_seqs = torch.stack(dnase_seqs,dim=0) 

        return idxs_return, ref_seqs, dnase_seqs
        '''
