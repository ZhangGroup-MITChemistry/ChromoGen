'''
So that we could use the published EPCOT code, we followed their same approach
and placed DNase-seq data in pickle files. However, we change the processing 
dramatically to improve efficiency at this step, which is otherwise incredibly slow. 
'''
import pycurl
import time
import pyBigWig
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

class ParseDNaseSeq:

    ########################################################################
    # Downoad the data

    @staticmethod
    def __download_bigWig(url,bigWig_fp,max_download_time,max_time_to_wait_for_connection):
        curl = pycurl.Curl()
        curl.setopt(curl.URL, url)
        if isinstance(max_download_time,int):
            curl.setopt(curl.TIMEOUT, max_download_time)
        if isinstance(max_time_to_wait_for_connection,int):
            curl.setopt(curl.CONNECTTIMEOUT, max_time_to_wait_for_connection)
        curl.setopt(curl.WRITEDATA, bigWig_fp.open('wb'))
        curl.setopt(curl.NOPROGRESS, False)
        curl.perform()
        curl.close()

    ########################################################################
    # Create csr matrices, dump in pkl files. 
    @staticmethod
    def __process_one_chrom(bigwig_values,chrom):

        # Time process for reference. 
        t = -time.time()

        # Convert to NumPy ndarray & replace NaN values with 0, equivalent to EPCOT paper's processing.
        vals = np.array(bigwig_values,dtype=np.float32)
        vals = np.nan_to_num(vals, copy=False, nan=0.0, posinf=None, neginf=None)

        # Place in csr matrix
        csr = csr_matrix(vals) 

        # Report to user
        t+= time.time()
        if t > 1:
            t = round(t,1)
        else:
            t = f'{t:.4e}'
        print(f'Chromosome {chrom} processed in {t} seconds.',flush=True)

        # Return data
        return csr

    ########################################################################
    # Initiate full pipeline
    @staticmethod
    def __format_int(x,variable_name):
        if x is not None:
            if isinstance(x,float):
                x = int(x) if int(x)==x else x
            assert isinstance(x,int), f'{variable_name} must be an integer. Received {type(x)}.'
            assert x > 0, f'{variable_name} must be positive-valued. Received {x}.'
        return x
    
    @staticmethod
    def from_file(bigWig_filepath,pickle_filepath=None,max_workers=None,overwrite=False):
        filepath = Path(bigWig_filepath)
        if sfx:=filepath.suffix:
            assert sfx in ['.bw','.bigWig'], f'Expected bigWig file to have extension .bw or .bigWig, but it has extension {sfx}.'
            assert filepath.is_file(), f'Provided bigWig filepath ({filepath}) does not exist.'
        elif filepath:= filepath.with_suffix('.bigWig').is_file():
            pass
        elif filepath:= filepath.with_suffix('.bw').is_file():
            pass
        else:
            raise Exception(
                f'Provided filepath ({filepath.with_suffix("")}) contained no extensions, '
                'and no file with that name seems to exist with extensions .bw or .bigWig.'
            )

        if pickle_filepath is None:
            save_fp = filepath.with_suffix('.pkl')
            file_was = 'inferred'
        else:
            save_fp = Path(pickle_filepath).with_suffix('.pkl')
            file_was = 'provided'

        if not overwrite and save_fp.exists():
            print(f'The {file_was} save file ({save_fp}) already exists. Skipping.',flush=True)
            return

        max_workers = ParseDNaseSeq.__format_int(max_workers,'max_workers')
        assert isinstance(overwrite,bool), f'overwrite should be a bool. Received {type(overwrite)}.'

        print(f'Loading and formatting DNase-seq data from bigWig file {filepath.name}',flush=True)
        t1 = -time.time()
        bw = pyBigWig.open(str(filepath))
        signals = {}
        for chrom, length in bw.chroms().items():
            # Skip over the 'chr' and save with integer, 'X', or 'Y' key in the pickle file
            chrom1 = chrom.replace('chr','') 
            try:
                chrom1 = int(chrom1)
            except:
                #if chrom1 not in ['X','Y']:
                if chrom1 != 'X':
                    # Skip any non-standard chromosomes (AND chrom Y, which we originally did for one reason or another)
                    continue

            # Load the data.
            # Round region down to an even 1000.
            print(f'Loading chromosome {chrom1} data from bigWig file',flush=True)
            t = -time.time()
            values = bw.values(chrom,0,length//1000 * 1000)
            t+= time.time()
            if t >= 1:
                t = round(t,1)
            else:
                t = f'{t:.4e}'
            print(f'Chromosome {chrom1} data loaded in {t} seconds.')
            
            # Process. Do this concurrently so that we can start loading the next chromosome's data. 
            signals[chrom1] = ParseDNaseSeq.__process_one_chrom(values, chrom1)

        print(f'All DNase-seq data loaded and formatted in {round(time.time()+t1,1)} seconds.',flush=True)
        print(f'Now saving formatted data to {save_fp.name}.', flush=True)
        t = -time.time()
        with save_fp.open('wb') as f:
            pickle.dump(signals,f)
        t+= time.time()
        if t >= 1:
            t = round(t,1)
        else:
            t = f'{t:.4e}'
        print(f'Formatted data saved in {t} seconds.',flush=True)

        t1+= time.time()
        print(f'DNase-seq data loaded, formatted, and saved in {round(t1,1)} seconds. Process is complete.',flush=True)

    @staticmethod
    def from_url(
        url,
        save_filepath,         # filepath in which to save the processed pkl files. 
        max_workers=4,         # Max threads to use when processing/saving data. Having too many sems to make them conflict!
                               # None defaults to ThreadPoolExecutor default (number CPUs + 4).
                               # Actually, this doesn't do anything anymore. Using multiple threads really slowed this down.
        max_download_time=600, # Maximum time (in seconds) to allow download to proceed before 
                               # cancelling the job. Positive integer, 0, or None (use pycurl default).
        max_time_to_wait_for_connection=5, # Maximum time (in seconds) to wait for a connection to 
                                           # be established before quitting. Positive integer or 0 
                                           # or None (use pycurl default).
        delete_bigWig=True,    # Whether to keep (False) or delete (True) the bigWig file after it's been processed
        overwrite=True         # Whether to overwrite the end pkl file if it exists. 
    ):
        
        # Validate/format input
        # Will let pycurl's exception handler deal with any issues/raise exceptions for the URL
        assert isinstance(filepath:=save_filepath,(str,Path)), f'Expected filepath to be string or pathlib.Path instance. Received {type(filepath)}.'
        filepath = Path(filepath).with_suffix('.pkl')
        max_download_time = ParseDNaseSeq.__format_int(max_download_time,'max_download_time')
        max_time_to_wait_for_connection = ParseDNaseSeq.__format_int(max_time_to_wait_for_connection,
                                                                'max_time_to_wait_for_connection')

        # Download
        bigWig_fp = filepath.with_suffix('.bigWig')
        if bigWig_fp.exists():
            # from_file behavior will preserve the bigWig file no matter what. Otherwise, same here. 
            print(f'Inferred bigWig filepath, {bigWig_fp}, already exists. Skipping download and defaulting to from_file behavior.',flush=True)
            delete_bigWig = False
        else:
            filepath.parent.mkdir(exist_ok=True,parents=True)
            print('Downloading the bigWig file. This might take a moment.',flush=True)
            t = -time.time()
            ParseDNaseSeq.__download_bigWig(url,bigWig_fp,max_download_time,max_time_to_wait_for_connection)
            t+= time.time()
            print(f'bigWig file downloaded in {round(t,1)} seconds.',flush=True)

        ParseDNaseSeq.from_file(bigWig_filepath=bigWig_fp,pickle_filepath=None,max_workers=max_workers,overwrite=overwrite)

        if delete_bigWig:
            print(f'Deleting bigWig file {bigWig_fp}.', flush=True)
            bigWig_fp.unlink()


if __name__ == '__main__':
    dest_dir = Path(__file__).parent / 'hg19'
    for cell_type,url in [
        ('GM12878','https://encode-public.s3.amazonaws.com/2017/09/06/e2259e48-add5-4b57-bc75-e27b290b954f/ENCFF901GZH.bigWig'),
        ('IMR90','https://encode-public.s3.amazonaws.com/2017/09/27/ab857d71-030e-41f1-9034-69c1355a6821/ENCFF291DOH.bigWig')
    ]:
        #'''
        with ThreadPoolExecutor() as executor:
            executor.submit(
                ParseDNaseSeq.from_url,
                url=url,
                save_filepath=dest_dir / cell_type,
                max_workers=24, # Number of chromosomes to process and, coincidentally, half the available cores. 
                max_download_time=600, 
                max_time_to_wait_for_connection=5, 
                delete_bigWig=False,
                overwrite=True
            )
        '''
        ParseDNaseSeq.from_url(
            url=url,
            save_filepath=dest_dir / cell_type,
            max_workers=4,      
            max_download_time=600, 
            max_time_to_wait_for_connection=5, 
            delete_bigWig=False,
            overwrite=True
        )
        '''
