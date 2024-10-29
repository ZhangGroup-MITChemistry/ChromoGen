###########################################################################################
# Adapted from https://github.com/liu-bioinfo-lab/EPCOT/blob/main/Input/dnase_processing.py
import pyBigWig
import numpy as np
import pickle
from scipy.sparse import csr_matrix, save_npz
import os 
import pycurl
import time
import gzip

def process_DNase(dnase_file):
    bw = pyBigWig.open(dnase_file)
    signals = {}
    for chrom, length in bw.chroms().items():
        try:
            if chrom == 'chrX':
                chr = 'X'
            else:
                chr = int(chrom[3:])
        except Exception:
            continue
        temp = np.zeros(length)
        intervals = bw.intervals(chrom)
        for interval in intervals:
            temp[interval[0]:interval[1]] = interval[2]
            
        seq_length = length // 1000 * 1000
        signals[chr] = csr_matrix(temp.astype('float32')[:seq_length])
        #print(dnase_file, seq_length, np.mean(signals[chr]))
        print(f'Chromosome {chr} complete.')

    print('Processing complete.',flush=True)
    print(f"Saving processed data to {dnase_file.replace('bigWig', 'pkl')}",flush=True)
    with open(dnase_file.replace('bigWig', 'pkl'), 'wb') as file:
        pickle.dump(signals, file)

    os.remove(dnase_file)

###########################################################################################
# Adapted from https://github.com/liu-bioinfo-lab/EPCOT/blob/main/Input/reference_genome.py
def process_fasta(fp,chroms,dest_dir=None):

    print("Loading fasta file.",flush=True) 
    with gzip.open(fp,mode='rt') as fn: #open(fp,'r') as fn:
        lines = fn.readlines()

    if dest_dir is None:
        f = fp.split('/')[-1].split('.')[0] # yields 'hg19', 'hg38', etc. 
        dir = ''.join(fp.replace( fp.split('/')[-1], '' ))
        if len(dir) == 0:
            dir = '.'
        dir+= f + '/'
    else: 
        dir = dest_dir

    if not os.path.exists(dir):
        os.mkdir(dir)
    
    print("Processing fasta file.")
    conv = {'A':0,'C':1,'G':2,'T':3,'N':-1}
    n = 0 
    nn = 0 
    N = len(chroms) 
    while len(chroms) > 0 and n < len(lines):
        if lines[n][0] == '>' and lines[n][1:-1] in chroms: # must ignore the '\n'
            t = -time.time()
            chrom = lines[n][4:-1]
            try: 
                chrom = int(chrom)
            except: 
                pass
            print(f'Processing chromosome {chrom}',flush=True) 

            m = n+1
            while m < len(lines) and lines[m][0] != '>':
                m+=1

            chrom_data = ''.join(lines[n+1:m]).replace('\n','').upper()

            i = np.array([conv[val] for val in chrom_data]) 
            ii = i > -1 
            j = np.arange(len(i))[ii]
            i = i[ii] 

            chrom_data = np.zeros((4,len(chrom_data)),dtype=np.int8)
            chrom_data[(i,j)] = 1

            seq_length =chrom_data.shape[1]//1000*1000
            chrom_data = csr_matrix(chrom_data[:,:seq_length])

            t+= time.time()
            print(f'Processed chromosome {chrom} in {t} seconds. Now saving',flush=True)
            
            save_npz(dir+f'chr{chrom}.npz',chrom_data)
            n = m

            chroms.remove(f'chr{chrom}')
            
        else:
            n+=1

###########################################################################################
# Higher-level code

def get_hg_files(alignment):
    
    if alignment == 'hg19': 
        url = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz'
    elif alignment == 'hg38': 
        url = 'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz'
    else: 
        raise Exception(f'URL for sequence file for {alignment} is unknown.')

    if alignment in ['hg19','hg38']:
        
        # Human chromosomes 
        chroms = [f'chr{k}' for k in range(1,23)]
        chroms.extend(['chrX','chrY'])
    else: 
        raise Exception(f'Chromosomes not specified for file {fp}')

    fps = []
    chr2rm = []
    for chrom in chroms: 
        fps.append(f'../../data/outside/{alignment}/{chrom}.npz')
        if os.path.exists(f'../../data/outside/{alignment}/{chrom}.npz'):
            chr2rm.append(chrom)

    if len(chr2rm) < len(chroms):
        fp = f'../../data/outside/{alignment}.fa.gz'
        for chrom in chr2rm:
            chroms.remove(chrom)

        if os.path.exists(fp):
            print(f"Must process the fasta file for alignment {alignment}. This might take a moment.",flush=True)
        else: 
            print(f"Must download and process the fasta file for alignment {alignment}. This might take a moment.",flush=True)
            curl = pycurl.Curl()
            curl.setopt(pycurl.URL, url)
            curl.setopt(pycurl.WRITEDATA, open(fp, "wb"))
            curl.setopt(curl.NOPROGRESS, False)
            curl.perform()
            curl.close()
            print("Download Complete",flush=True)

        process_fasta(fp,chroms) 
        #os.remove(fp) 
        print('Alignment data is now processed',flush=True)

    return fps


def get_DNase_file(cell_type,alignment): 

    if cell_type == 'GM12878' and alignment == 'hg19': 
        url = 'https://encode-public.s3.amazonaws.com/2017/09/06/e2259e48-add5-4b57-bc75-e27b290b954f/ENCFF901GZH.bigWig'
        fp = '../../data/outside/GM12878_hg19'
    elif cell_type == 'IMR90' and alignment == 'hg19': # https://www.encodeproject.org/files/ENCFF291DOH/
        url = 'https://encode-public.s3.amazonaws.com/2017/09/27/ab857d71-030e-41f1-9034-69c1355a6821/ENCFF291DOH.bigWig'
        fp = '../../data/outside/IMR90_hg19'
        
    #elif OTHERS
    else: 
        raise Exception('Filetype not recognized')

    if os.path.exists(fp+'.pkl'): 
        return

    if os.path.exists(fp+'.bigWig'):
        print(f"Must process the bigWig file for cell type {cell_type}. This might take a moment.",flush=True)
    else: 
        print(f"Must download and process the bigWig file for cell type {cell_type}. This might take a moment.",flush=True)
        curl = pycurl.Curl()
        curl.setopt(pycurl.URL, url)
        curl.setopt(pycurl.WRITEDATA, open(fp+'.bigWig', "wb"))
        curl.setopt(curl.NOPROGRESS, False)
        curl.perform()
        curl.close()
        print("Download Complete",flush=True)

    print(f"Processing")
    process_DNase(fp+'.bigWig')

    return fp+'.pkl'

###################
# For the initial download
if __name__ == '__main__':
    _ = get_hg_files('hg19')
    _ = get_DNase_file('GM12878','hg19')
    