'''
Greg Schuette 2024
'''
from concurrent.futures import ThreadPoolExecutor
import pyBigWig
import torch
from pathlib import Path

class BigWigHandler:
    # For now, only supports local files
    def __init__(self,bigWig_filepath,dtype=None,device=None):
        self.filepath = bigWig_filepath
        self.dtype = dtype
        self.device = device

        # In case this class is called asynchronously, don't want to try to load from the
        # file in different threads simultaneously, as that significantly slows things down
        # in my experience. Circumvent that by forcing all loads through an executor with
        # one thread regardless of whether this object is being called by multiple processes
        # simultaneously. 
        # That said, I'll only bother doing that for fetch(), as nothing else is really supposed
        # to be called in multithreaded processes elsewhere in the ChromoGen package. 
        self.__executor = ThreadPoolExecutor(max_workers=1) 
    ############
    # Properties
    @property
    def filepath(self):
        return self.__filepath

    @filepath.setter
    def filepath(self,bigWig_filepath):
        if bigWig_filepath is None:
            self.__bigWig = None
        else:
            assert isinstance(bigWig_filepath,(str,Path)), \
            f'bigWig_filepath should be a string or pathlib.Path instance. Received {type(bigWig_filepath)}.'
            bigWig_filepath = Path(bigWig_filepath)
            assert bigWig_filepath.exists(), f'The provided bigWig_filepath, {bigWig_filepath}, does not exist.'
            assert not bigWig_filepath.is_dir(), f'The provided bigWig_filepath, {bigWig_filepath}, is a directory.'
            assert bigWig_filepath.suffix, 'The provided bigWig_filepath has no extension. Expected ".bw" or ".bigWig" extension.'
            assert bigWig_filepath.suffix in ['.bw','.bigWig'], \
            f'The provided bigWig_filepath has extension {bigWig_filepath.suffix}. Expected ".bw" or ".bigWig".'
            try:
                self.__bigWig = pyBigWig.open(str(bigWig_filepath))
            except Exception as e:
                raise Exception('The following error was reported while attempting to load the bigWig file:\n\t'+str(e))
        self.__bigWig_filepath = bigWig_filepath

    @property
    def bigWig(self):
        return self.__bigWig
    
    def exists(self):
        return self.bigWig is not None

    @property
    def chrom_names(self):
        return list(self.chroms().keys())

    ############
    # We'll artificially extend the bigWigFile class using the following methods & the
    # loop in __bigWig's setter
    def __assert_exists(self,call_fcn):
        assert self.exists(), (
            f'Attempting to access BigWigHandler.{call_fcn}, but this '
            'BigWigHandler instance has no associated bigWig file. Please add '
            'by running `bw_handler.filepath = <filepath>`.'
        )

    def __pseudo_extend(self,method_name):
        def f(*args,**kwargs):
            self.__assert_exists(method_name)
            return getattr(self.bigWig,method_name)(*args,**kwargs)
        return f

    @property
    def __bigWig(self):
        return self.__bigWigInstance

    @__bigWig.setter
    def __bigWig(self,bw):
        if bw is not None:
            # Simply verify that we have a bigWig & not bigBed file. Shouldn't be an issue 
            # unless someone misnames a file
            assert bw.isBigWig(), \
            f'Expected pyBigWig.bigWigFile instance. Received type {type(bw)}.'
            bw1 = bw
        else:
            bw1 = pyBigWig.pyBigWig

        # Add all the methods (note that pyBigWig.BigWigFile objects don't have properties, 
        # so I can be a bit lazy here... and that I'm adding properties in this janky way
        # since pyBigWig objects in general aren't easily extended.) 
        for method_name in dir(bw1):
            # Skip any hidden & magic methods
            if method_name and method_name[0].islower():
                setattr(self,method_name,self.__pseudo_extend(method_name))

        self.__bigWigInstance = bw

    ############
    # Fetch/format regions in a convenient way
    def fetch(self,chromosome,start,end,nan_to=None):
        # Verify/format inputs 
        chrom_names = self.chrom_names
        if chromosome not in chrom_names:
            if (c:=str(chromosome)) in chrom_names:
                chromosome = c
            elif (c:=f'chr{chromosome}') in chrom_names:
                chromosome = c
            else:
                raise Exception(f'Chromosome {chromosome} not found in bigWig file {self.filepath}.')
        if convert_nan:= (nan_to is not None):
            try:
                nan_to = float(nan_to)
            except:
                raise Exception(
                    'nan_to must be NoneType or convertible to a float. '
                    f'Provided value of type {type(nan_to).__name__} is not.'
                )
        if isinstance(start,float) and int(start)==start:
            start = int(start)
        assert isinstance(start,int), f'start index must be an integer. Received {type(start).__name__}.'
        if isinstance(end,float) and int(end)==end:
            end = int(end)
        assert isinstance(end,int), f'end index must be an integer. Received {type(end).__name__}.'

        # Fetch the data, place in a torch.Tensor, and replace NaN's (if desired). 
        values = self.__executor.submit(self.values,chromosome, start, end).result()
        values = torch.tensor(values, dtype=type(values[0])) # Going to realistically just be doubles, I think
        if convert_nan:
            values.nan_to_num_(nan_to)
        return values.to(dtype=self.dtype,device=self.device)
        