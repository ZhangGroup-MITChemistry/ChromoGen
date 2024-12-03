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

def get_hg_files(alignment,dest_dir='./',download_only=False):
    
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
        fps.append(dest_dir + '/{alignment}/{chrom}.npz')
        if os.path.exists(dest_dir + '/{alignment}/{chrom}.npz'):
            chr2rm.append(chrom)

    if len(chr2rm) < len(chroms):
        fp = dest_dir + f'/{alignment}.fa.gz'
        must_process=True
        for chrom in chr2rm:
            chroms.remove(chrom)

        if not os.path.exists(fp): 
            print(f"Must download the fasta file for alignment {alignment}. This might take a moment.",flush=True)
            curl = pycurl.Curl()
            curl.setopt(pycurl.URL, url)
            curl.setopt(pycurl.WRITEDATA, open(fp, "wb"))
            curl.setopt(curl.NOPROGRESS, False)
            curl.perform()
            curl.close()
            print("Download Complete",flush=True)

        if download_only:
            print(f'FASTA file for alignment {alignment} not processed, per user request',flush=True)
            return
        else:    
            print(f"Must process the fasta file for alignment {alignment}. This might take a moment.",flush=True)
            fp = dest_dir + f'/{alignment}.fa.gz'
            process_fasta(fp,chroms) 
            os.remove(fp) 
            print(f'Alignment data is now processed ({alignment})',flush=True)


    return #fps, fp # processed files, unprocessed file


def get_DNase_file(cell_type,alignment,download_fp=None,download_only=False): 
    fp = download_fp
    if cell_type == 'GM12878' and alignment == 'hg19': 
        url = 'https://encode-public.s3.amazonaws.com/2017/09/06/e2259e48-add5-4b57-bc75-e27b290b954f/ENCFF901GZH.bigWig'
        if fp is None:
            fp = 'GM12878_hg19'
    elif cell_type == 'IMR90' and alignment == 'hg19': # https://www.encodeproject.org/files/ENCFF291DOH/
        url = 'https://encode-public.s3.amazonaws.com/2017/09/27/ab857d71-030e-41f1-9034-69c1355a6821/ENCFF291DOH.bigWig'
        if fp is None:
            fp = 'IMR90_hg19'
        
    #elif OTHERS
    else: 
        raise Exception('Filetype not recognized')

    if os.path.exists(fp+'.pkl'):
        print('DNase seq data for alignment {alignment} in {cell_type} cells already exists. Skipping.',flush=True)
        return fp+'.pkl'

    if not os.path.exists(fp+'.bigWig'):
        print(f"Must download the bigWig file for cell type {cell_type}. This might take a moment.",flush=True)
        curl = pycurl.Curl()
        curl.setopt(pycurl.URL, url)
        curl.setopt(pycurl.WRITEDATA, open(fp+'.bigWig', "wb"))
        curl.setopt(curl.NOPROGRESS, False)
        curl.perform()
        curl.close()
        print("Download Complete for DNase-seq in {cell_type} cells",flush=True)

    if download_only:
        print(f'Data remains unprocessed for DNase-seq in {cell_type} cells, per user request.',flush=True)
        return fp + '.bigWig'

    print(f"Processing the bigWig file for cell type {cell_type}. This might take a moment.",flush=True)
    
    process_DNase(fp+'.bigWig')

    return fp+'.pkl', True

###################
# For the initial download
if __name__ == '__main__':
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path

    def get_and_process_fasta(
        alignment,
        dest_dir=None,
        chroms=[str(k) for k in range(1,23)] + ['X'],
        download_only=False
    ):
        # Download data. The fps are essentially files we want to create
        # afterwards during processing
        fps = get_hg_files(alignment,dest_dir)
        if download_only:
            return
        print("AHHHH 1")
        with ThreadPoolExecutor() as executor:
            for fp in fps:
                executor.submit(process_fasta,fp=fp,chroms=chroms,dest_dir=dest_dir)

    def get_and_process_DNase_seq(
        cell_type,
        alignment,
        dest_dir=None,
        filename=None,
        download_only=False,
    ):
        if filename is not None:
            if dest_dir is not None:
                fp = dest_dir + '/' + filename
            else:
                fp = filename
        elif dest_dir is not None:
            fp = dest_dir + '/' + cell_type
        else:
            fp = None

        dnase_file = get_DNase_file(cell_type,alignment,fp)
        if download_only:
            return
        print("AHH")
        process_DNase(dnase_file)

    
    this_dir = '/'.join(__file__.split('/')[:-1]) + '/'
    details = {
        # alignments as keys
        'hg19':[
            # List of cell types as values
            'GM12878',
            'IMR90'
        ]
    }

    with ThreadPoolExecutor() as executor:
     
        for alignment,cell_types in details.items():
            Path(this_dir+f'{alignment}/').mkdir(exist_ok=True,parents=True)
            #get_and_process_fasta(alignment,dest_dir=this_dir+f'{alignment}/',download_only=True)
            #executor.submit(
            #    get_and_process_fasta,
            #    alignment=alignment,
            #    dest_dir=this_dir+f'{alignment}/',
            #    download_only=True
            #)
            executor.submit(
                get_hg_files,
                alignment=alignment,
                dest_dir=this_dir+f'{alignment}/',
                download_only=True#False
            )

            '''
            for cell_type in cell_types:
                Path(this_dir+f'DNase_seq/{alignment}/').mkdir(exist_ok=True,parents=True)
                #get_and_process_DNase_seq(
                #    cell_type=cell_type,
                #    alignment=alignment,
                #    dest_dir=this_dir+f'DNase_seq/{alignment}/',
                #    filename=cell_type,
                #    download_only=True
                #)
                #executor.submit(
                #    get_and_process_DNase_seq,
                #    cell_type=cell_type,
                #    alignment=alignment,
                #    dest_dir=this_dir+f'DNase_seq/{alignment}/',
                #    download_only=True
                #)
                executor.submit(
                    get_DNase_file,
                    cell_type=cell_type,
                    alignment=alignment,
                    download_fp=this_dir+f'DNase_seq/{alignment}/{cell_type}',
                    download_only=False
                )
            '''
    '''
    # Processing separately due to memory issues 
    for alignment,cell_types in details.items():    
        get_hg_files(
            alignment=alignment,
            dest_dir=this_dir+f'{alignment}/',
            download_only=False
        )
        for cell_type in cell_types:
            get_DNase_file(
                cell_type=cell_type,
                alignment=alignment,
                download_fp=this_dir+f'DNase_seq/{alignment}/{cell_type}',
                download_only=False
            )
    '''


    
