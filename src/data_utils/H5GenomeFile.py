'''
Greg Schuette 2024
'''
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import h5py
import torch

class H5GenomeFile:
    '''
    Should only be used internally, so won't be very careful with verifying arguments, etc. 
    '''

    def __init__(self,filepath,dtype=None,device=None):
        fp = Path(filepath)
        assert not fp.is_dir(), f'The provided filepath, {filepath}, is a directory.'
        if (sfx:=fp.suffix) != '.h5':
            fp = fp.with_suffix(sfx + '.h5')
        self.__filepath = fp

        # We'll use this to prevent outside asyncronous processes from trying to open 
        # the file separately and simultaneously. 
        self.__executor = ThreadPoolExecutor(max_workers=1)

        self.dtype = None
        self.device = None

        self.__executor.submit(self.__set_chroms).result()

    @property
    def filepath(self):
        return self.__filepath

    def __set_chroms(self):
        if self.filepath.is_file():
            with h5py.File(self.filepath, 'r') as hf:
                self.__chroms = {key:hf[key].shape[0] for key in hf}
        else:
            self.__chroms = {}

    def chroms(self,chrom=None):
        # Want to pipe all file access through the one thread to prevent conflicts, 
        # especially because the file can't be open in more than one location
        return self.__chroms.copy()

    def __add_chrom(self,data,key,compression,overwrite):
        assert isinstance(data, torch.Tensor), f'Expected data to be a torch.Tensor. Received {type(data).__name__}.'
        assert data.ndim == 2, (
            'Expected two-dimensional object, but received data object '
            f'with {data.ndim} dimensions for chromosome {key}.'
        )
        # Transpose so that the genomic index is in the first position, which
        # is just generally convenient. This should always need to be done, 
        # but just go ahead and double check that's true first. 
        if data.shape[0] == 4:
            data = data.mT
        else:
            assert data.shape[1] == 4, \
            f'One of the data dimensions for chromosome {key} must be 4, indicating ATCG.'

        fp = self.filepath
        file_existed = fp.exists()

        try:
            fp.parent.mkdir(exist_ok=True,parents=True)
            with h5py.File(fp, 'a') as hf:
    
                if key in hf.keys():
                    if overwrite:
                        del hf[key]
                    else:
                        raise Exception(
                            f'The file {fp} already has data for chromosome {key}. '
                            f'If you wish to overwrite the data, pass overwrite=True.'
                        )
    
                # 1: Convert to bool, which will ideally help with bitpacking, etc. 
                # 2: Ensure we're on the CPU so that we can convert to numpy.ndarray
                # 3: Convert to numpy.ndarray
                    # Note: h5py automatically converts torch.Tensor objects to numpy.ndarrays, 
                    # but I'm not positive that's true on all versions, so will do it explicitly
                    # to maximize compatibility. Doesn't exactly cost any extra computation anyway.
                data = data.bool().cpu().numpy()
    
                # Chunking is required if compression used, but otherwise we don't need it 
                # because we'll never change the size of this object. 
                hf.create_dataset(key, data=data, compression=compression)
        except:
            if not file_existed and fp.exists():
                fp.unlink()
            raise

        # Can ignore the executor since we're already inside the safe thread when using this method
        self.__set_chroms()

    def add_chrom(self,data,chromosome,compression='gzip',overwrite=False):
        assert isinstance(chromosome,(int,str)), \
        f'Expected chromosome name to be integer or string. Received {type(chromosome).__name__}.'
        chromosome = str(chromosome).replace('chr','')
        return self.__executor.submit(self.__add_chrom,data,chromosome,compression,overwrite).result()

    def __getitem(self,chrom,idx):
        with h5py.File(self.filepath, 'r') as hf:
            data = hf[chrom][idx,:]
        return data

    def __getitem__(self,chrom_idx):
        err_msg = 'Expected index to be either chromosome_name or (chromosome_name, genomic_index_to_take). '
        if isinstance(chrom_idx,tuple):
            assert len(chrom_idx)==2, f'However, received tuple index with {len(chrom_idx)} elements.'
            chromosome, idx = chrom_idx[0], chrom_idx[1]
        else:
            chromosome = chrom_idx
            idx = slice(None,None,None)

        assert isinstance(chromosome,(int,str)), \
        f'Expected chromosome name to be integer or string. Received {type(chromosome).__name__}.'
        chromosome = str(chromosome).replace('chr','')
        data = self.__executor.submit(self.__getitem,chromosome,idx).result()
        # To PyTorch, using specified device/dtype (if any)
        data = torch.from_numpy(data).to(dtype=self.dtype,device=self.device)
        # Place the ATCG index in position 0 and return
        if data.ndim == 1:
            # If indexed with an integer, returned value will be flat, so just unsqueeze
            data = data.unsqueeze(-1)
        else:
            data = data.mT
        return data

    def fetch(self,chromosome,start,end):
        if isinstance(start,float) and int(start)==start:
            start = int(start)
        assert isinstance(start,int), f'Expected start index to be an integer. Received {type(start).__name__}.'
        if isinstance(end,float) and int(end)==end:
            end = int(end)
        assert isinstance(end,int), f'Expected end index to be an integer. Received {type(end).__name__}.'
        if start==0 and end==-1:
            return self[chromosome,:] # Otherwise, we miss one
        return self[chromosome,start:end]