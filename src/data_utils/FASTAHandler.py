'''
Greg Schuette 2024
'''
import io
import pycurl
import time
import gzip
import torch
from scipy.sparse import csr_matrix, save_npz
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .H5GenomeFile import H5GenomeFile
from . import __filepath_utils as fpu

class FASTAHandler:

    ########################################################################
    # Load the data
    
    @staticmethod
    def __load_fasta_file(filepath):
        assert fpu.is_fasta(filepath), (
            f'Based on its extension(s), the provided filepath, {filepath}, doesn\'t seem to be a FASTA file.'
        )
        assert filepath.exists(), f'The provided filepath, {filepath}, does not exist.'
        assert not filepath.is_dir(), f'The provided filepath, {filepath}, is a directory.'

        if filepath.suffix == '.gz':
            # Compressed
            sequence = gzip.open(filepath,mode='rb').read()
        else:
            # Uncompressed
            sequence = filepath.open('rb').read()

        return bytearray(sequence) # Improves speed downstream

    @staticmethod
    def __load_fasta_url(url,max_download_time,max_time_to_wait_for_connection,is_gzip):
        t = -time.time()
        buffer = io.BytesIO()
        curl = pycurl.Curl()
        curl.setopt(curl.URL, url)
        if isinstance(max_download_time,int):
            curl.setopt(curl.TIMEOUT, max_download_time)
        if isinstance(max_time_to_wait_for_connection,int):
            curl.setopt(curl.CONNECTTIMEOUT, max_time_to_wait_for_connection)
        curl.setopt(curl.WRITEDATA, buffer)
        curl.setopt(curl.NOPROGRESS, False)
        curl.perform()
        curl.close()
        sequence = buffer.getvalue()
        buffer.close()
        del buffer
        t+= time.time()
        print(f'FASTA file downloaded in {round(t,1)} seconds.',flush=True)
        if is_gzip:
            print('Decompressing FASTA file from gzip format',flush=True)
            t = -time.time()
            sequence = gzip.decompress(sequence)
            t+= time.time()
            print(f'FASTA file decompressed in {round(t,1)} seconds.',flush=True)
        return bytearray(sequence) # Improves speed downstream 

    ########################################################################
    # Format the data
    @staticmethod
    def __split_id_seq(seq):
        try:
            i = min(
                ii for k in [b'A',b'T',b'C',b'G',b'N'] if (ii:=seq.find(k)) > 0 
            )
            return seq[:i], seq[i:]
        except:
            asdf
            # Empty sequence. Not an issue in the hg19 file, at least, but who knows about other genomes... 
            return seq, b''
    
    @staticmethod
    def __parse_fasta(sequence,chroms):
        seqs = sequence.replace(b'\n',b'').split(b'>')
        
        if chroms is None:
            return [FASTAHandler.__split_id_seq(seq) for seq in seqs if seq]
        return [name_seq for seq in seqs if seq and ( name_seq:= FASTAHandler.__split_id_seq(seq) )[0] in chroms]

    @staticmethod
    def __to_tensor(seq_bytearray,round_down):
        # Use the same codec as in EPCOT paper, https://doi.org/10.1093/nar/gkad436,
        # to convert ATCGN values to 4xN binary array with ones in first, second, third, 
        # and fourth row corresponding to A, T, C, and G, respectively. 

        # As in the EPCOT paper, round chromosome length down to an even 1000 (at least, do it for the npz files)
        if round_down:
            seq_bytearray = seq_bytearray[:len(seq_bytearray)//1000*1000]

        # Convert the byte array into a tensor filled with values 0, 1, 2, 3, and 4, 
        # which we'll use to index the final tensor.
        # Using upper makes insertions equivalent to higher-confidence regions for processing purposes
        codec = {
            code:k%5 for k,code in enumerate(bytearray('ACGTNacgtn','UTF-8'))
        }
        idx=torch.frombuffer(seq_bytearray,dtype=torch.uint8)
        for code, encoding in codec.items():
            idx[idx==code] = encoding
        ''' # If memory constraints are relevant, could use this instead, but it takes ~4x as long (still fast...)
        idx = torch.frombuffer(
            seq_bytearray.upper(
            ).replace(b'A',b'\x00'
            ).replace(b'C',b'\x01'
            ).replace(b'G',b'\x02'
            ).replace(b'T',b'\x03'
            ).replace(b'N',b'\x04'
            ),
            dtype=torch.int8
        )
        '''

        # Create the final array, using bools to represent 0's and 1's
        # Include fifth row for known for quick indexing (including N values), then drop it.
        # Note that converting torch.uint8 objects (which are deprecated for indexing usage)
        # to bool() SHOULD have this work correctly. However, it seems like something that 
        # could be altered in a later version of PyTorch, so I'm implementing this with
        # the longer & more memory-intensive long() option. 
        data = torch.zeros(5,len(idx),dtype=bool)
        data[(idx.long(),torch.arange(len(idx)))] = True

        return data[:4,:]

    @staticmethod
    def __postprocess_and_save_one_chrom_npz(
        chrom_name,
        seq_bytearray,
        save_dir,
        # The following two don't do anything in the npz case
        compression,
        overwrite
    ):
        s = chrom_name.decode()
        t = -time.time()
        print(f'Processing chromosome {s}.',flush=True)
        # As in EPCOT paper, switch to float32 at the end.         
        csr_mat = csr_matrix(
            FASTAHandler.__to_tensor(seq_bytearray,round_down=True).numpy()
        ).astype('float32')
        
        t+= time.time()
        if t >= 1:
            t = round(t,1)
        else:
            t = f'{t:.4e}'
        print(f'Chromosome {s} processing completed in {t} seconds.',flush=True)
        file = (save_dir / s).with_suffix('.npz')
        print(f'Now saving processed {s} data to {file}.',flush=True)
        t = -time.time()
        save_npz(file,csr_mat)
        t+= time.time()
        print(f'Save completed for chromosome {s} in {round(t,1)} seconds.',flush=True)

    @staticmethod
    def __postprocess_and_save_one_chrom_hdf5(
        chrom_name,
        seq_bytearray,
        save_dir,
        compression,
        overwrite
    ):
        # save_dir is actually a H5GenomeFile instance in this case, so rename it for clarity
        h5_genome_file = save_dir
        
        s = chrom_name.decode()
        t = -time.time()
        print(f'Processing chromosome {s}.',flush=True)
        data = FASTAHandler.__to_tensor(seq_bytearray,round_down=False)
        t+= time.time()
        if t >= 1:
            t = round(t,1)
        else:
            t = f'{t:.4e}'
        print(f'Chromosome {s} processing completed in {t} seconds.',flush=True)
        print(f"Adding processed chromosome {s} data to H5GenomeFile's save queue.",flush=True)
        h5_genome_file.add_chrom(data=data,chromosome=s,compression=compression,overwrite=overwrite)
        print(f'Save completed for chromosome {s}.',flush=True)
    
    @staticmethod
    def __process_pipeline(
        sequence,
        chroms,
        max_workers,
        save_dir,
        filetype,
        compression,
        overwrite
    ):

        if filetype.lower() in ['npz','.npz']:
            postprocess_fcn = FASTAHandler.__postprocess_and_save_one_chrom_npz
            save_dir.mkdir(exist_ok=True,parents=True)
        else:
            postprocess_fcn = FASTAHandler.__postprocess_and_save_one_chrom_hdf5
            f = fpu.fasta_fp_to_h5_fp(save_dir)
            save_dir = H5GenomeFile(f)
        
        chrom_sequences = FASTAHandler.__parse_fasta(sequence,chroms)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Keep track of futures so that any exceptions are raised
            # when future.result() is called
            futures = []
            for chrom_name, seq_bytearray in chrom_sequences:
                futures.append(
                    executor.submit(
                        postprocess_fcn,
                        chrom_name=chrom_name,
                        seq_bytearray=seq_bytearray,
                        save_dir=save_dir,
                        compression=compression,
                        overwrite=overwrite
                    )
                )
            for f in futures:
                f.result()
    
    ########################################################################
    # To call from the outside

    @staticmethod
    def __format_chroms(chroms):
        if chroms is None:
            return chroms
        assert isinstance(chromosomes,list), 'chromosomes must be a list'
        assert chromosomes, 'chromosomes cannot be empty'
        for i,chrom in enumerate(chroms):
            assert isinstance(chrom,(int,str)), 'Values inside the chromosomes list must be strings or integers.'
            chrom = str(chrom)
            if chrom[:3] != 'chr':
                chrom = 'chr' + chrom
            chroms[i] = chrom
        return [bytes(c,'UTF-8') for c in chroms]

    @staticmethod
    def __format_save_dir(save_dir,filepath=None,file_ok=False):
        if save_dir is not None:
            assert isinstance(save_dir,(str,Path)), f'The save directory (save_dir) must be a string or pathlib.Path instance. Received {type(save_dir)}'
            save_dir = Path(save_dir)
            if not file_ok and save_dir.exists():
                assert save_dir.is_dir(), f'The provided save directory, save_dir={save_dir}, is not a directory.'
        else:
            # If we get here, this must have been called by from_file and a filepath must exist. 
            save_dir = filepath.parent
        save_dir.mkdir(exist_ok=True,parents=True)
        return save_dir

    @staticmethod
    def __format_max_workers(max_workers):
        if max_workers is not None:
            if isinstance(max_workers,float):
                max_workers = int(max_workers) if int(max_workers)==max_workers else max_workers
            assert isinstance(max_workers,int), f'max_workers must be an integer. Received {type(max_workers)}.'
            assert max_workers > 0, f'max_workers must be positive-valued. Received {max_workers}.'
        return max_workers

    @staticmethod
    def __format_int(x,variable_name):
        if x is not None:
            if isinstance(x,float):
                x = int(x) if int(x)==x else x
            assert isinstance(x,int), f'{variable_name} must be an integer. Received {type(x)}.'
            assert x > 0, f'{variable_name} must be positive-valued. Received {x}.'
        return x
                   
    @staticmethod
    def from_file(
        filepath,
        save_dir=None,
        chromosomes=None,
        max_workers=None,
        filetype='h5',
        # The following two don't do anything with npz filetype
        compression='gzip',
        overwrite=True
    ):
        # Validate/format input
        assert isinstance(filepath,(str,Path)), f'Expected filepath to be string or pathlib.Path instance. Received {type(filepath).__name__}.'
        filepath = Path(filepath)
        assert isinstance(filetype, str), f'Expected filetype to be a string. Received {type(filetype).__name__}.'
        if filetype.lower() in ['h5','hdf5','.h5','.hdf5']:
            save_dir = FASTAHandler.__format_save_dir(save_dir,filepath=filepath,file_ok=True)
            if save_dir.is_dir() or not save_dir.suffix():
                save_dir = save_dir / Path(str(filepath)).name
        elif filetype.lower() in ['.npz','npz']:
            save_dir = FASTAHandler.__format_save_dir(save_dir,filepath=filepath,file_ok=False)
        else:
            raise Exception(f'Expected filetype to be "npz" or "h5". Received {filetype}.')
        max_workers = FASTAHandler.__format_max_workers(max_workers)
        chroms = FASTAHandler.__format_chroms(chromosomes)
        
        # Load data from file
        print('Loading the FASTA file. This might take a moment.',flush=True)
        t = -time.time()
        sequence = FASTAHandler.__load_fasta_file(filepath)
        t+= time.time()
        print(f'FASTA file loaded in {round(t,1)} seconds.',flush=True)

        # Perform post-processing, save
        print('Processing and saving the FASTA data. This might take a moment.',flush=True)
        t1 = -time.time()
        FASTAHandler.__process_pipeline(sequence,chroms,max_workers,save_dir,filetype,compression,overwrite)
        t1+= time.time()
        print(f'FASTA file processed and saved in {round(t1,1)} seconds.',flush=True)
        print(f'Full FASTA pipeline complete in {round(t+t1,1)} seconds.',flush=True)

        if filetype.lower() in ['.npz','npz']:
            print(f'The .npz files are located in the following directory: {save_dir}',flush=True)
        else:
            print(f"The HDF5 file's path is: {save_dir}",flush=True)
        return save_dir

    @staticmethod
    def from_url(
        url,
        save_dir='./',         # Directory in which to save the processed npz files. 
        chromosomes=None,      # Which chromosomes to process/save. Default (None) will convert all chromosomes. 
        max_workers=None,      # Max threads to use when processing/saving data. 
                               # None defaults to ThreadPoolExecutor default (number CPUs + 4).
        max_download_time=600, # Maximum time (in seconds) to allow download to proceed before 
                               # cancelling the job. Positive integer, 0, or None (use pycurl default).
        max_time_to_wait_for_connection=5, # Maximum time (in seconds) to wait for a connection to 
                                           # be established before quitting. Positive integer or 0 
                                           # or None (use pycurl default).
        save_fasta=False,       # Whether to save the FASTA file
        is_gzip=True,           # Is the downloaded file in gzip format
        filetype='h5',
        # The following two don't do anything with npz filetype
        compression='gzip',
        overwrite=True
    ):
        # Validate/format input
        # Will let pycurl's exception handler deal with any issues/raise exceptions for the URL existing or not, being malformed, etc. 
        assert isinstance(url,str), f'Expected url to be a string. Received {type(url).__name__}.'
        assert fpu.is_fasta(url), (
            f'Based on its extension(s), the provided url, {url}, doesn\'t seem correspond to a FASTA file.'
        )
        assert isinstance(save_dir,(str,Path)), (
            f'save_dir must be a string or pathlib.Path instance. Received {type(save_dir).__name__}.'
        )
        assert isinstance(filetype, str), f'Expected filetype to be a string. Received {type(filetype).__name__}.'
        if filetype.lower() in ['h5','hdf5','.h5','.hdf5']:
            save_dir = FASTAHandler.__format_save_dir(save_dir,file_ok=True)
            if save_dir.is_dir() or not save_dir.suffix():
                save_dir = save_dir / Path(str(url)).name
        elif filetype.lower() in ['.npz','npz']:
            save_dir = FASTAHandler.__format_save_dir(save_dir,file_ok=False)
        else:
            raise Exception(f'Expected filetype to be "npz" or "h5". Received {filetype}.')
        max_workers = FASTAHandler.__format_int(max_workers,'max_workers')
        chroms = FASTAHandler.__format_chroms(chromosomes)
        max_download_time = FASTAHandler.__format_int(max_download_time,'max_download_time')
        max_time_to_wait_for_connection = FASTAHandler.__format_int(max_time_to_wait_for_connection,
                                                                'max_time_to_wait_for_connection')

        # Load data from file
        print('Downloading the FASTA file. This might take a moment.',flush=True)
        t = -time.time()
        sequence = FASTAHandler.__load_fasta_url(url,max_download_time,max_time_to_wait_for_connection,is_gzip)
        t+= time.time()
        print(f'FASTA file downloaded in {round(t,1)} seconds.',flush=True)

        # Perform post-processing, save
        print('Processing and saving the FASTA data. This might take a moment.',flush=True)
        t1 = -time.time()
        FASTAHandler.__process_pipeline(sequence,chroms,max_workers,save_dir,filetype,compression,overwrite)
        t1+= time.time()
        print(f'FASTA file processed and saved in {round(t1,1)} seconds.',flush=True)
        print(f'Full FASTA pipeline complete in {round(t+t1,1)} seconds.',flush=True)

        if filetype.lower() in ['.npz','npz']:
            print(f'The .npz files are located in the following directory: {save_dir}',flush=True)
        else:
            print(f"The HDF5 file's path is: {save_dir}",flush=True)
        return save_dir
        
