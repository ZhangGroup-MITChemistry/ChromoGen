'''
Greg Schuette 2024
'''
import torch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import warnings
from .BigWigHandler import BigWigHandler
from .H5GenomeFile import H5GenomeFile
from .FASTAHandler import FASTAHandler
from . import __filepath_utils as fpu

##########################################################################################
# Various support items

class _support:

    @staticmethod
    def get_h5(alignment_filepath,auto_parse_fasta,interactive,max_workers):
        # If the provided filepath is a FASTA file, check if the equivalent h5 file exists. If not, 
        # process it if need be (and user desires it), THEN get back to this initialization. \
        f = Path(alignment_filepath)
        fp_og = alignment_filepath
        alignment_filepath = fpu.format_h5_filepath(f)
        if not alignment_filepath.is_file() and not alignment_filepath.parent.is_dir():
            # If a URL was passed, then we'd want to drop everything except the filename since http://, etc., 
            # aren't local directory paths and hope we're in the correct directory already. 
            alignment_filepath = Path(alignment_filepath.name)
        if not alignment_filepath.is_file():
            if not fpu.is_fasta(f):
                if f.suffix == '.h5':
                    raise Exception(f'The provided .h5 genome filepath, {f}, does not exist.')
                else:
                    raise Exception(
                        f'The provided .h5 genome filepath, {f}, does not have an .h5 extension. '
                        f'Its inferred filepath, {alignment_filepath}, does not exist. '
                        'Please pass a path to a pre-formatted .h5 genome file OR a FASTA file '
                        'either in interactive mode or with auto_parse_fasta=True.'
                    )
            if not auto_parse_fasta and interactive:
                response = ''
                n=0
                while response not in ['Y','n']:
                    if n > 0:
                        print('Please choose a valid option (Y for yes or n for no).')
                    response = input(
                        'You seem to have passed a FASTA filepath for the genome file, and no equivalent HDF5 file '
                        'was found. Would you like me to process the FASTA file to create the HDF5-formatted file '
                        'required by ChromoGen.data_utils.SequenceLoader? [Y/n]: '
                    )
                    n+=1
                auto_parse_fasta = response == 'Y'
            if auto_parse_fasta:
                print('Will process the FASTA file to create the HDF5-formatted file required by ChromoGen.data_utils.SequenceLoader.',flush=True)
                if f.is_file():
                    FASTAHandler.from_file(
                        f,
                        save_dir=None,
                        chromosomes=None,
                        max_workers=max_workers,
                        filetype='h5',
                        compression='gzip',
                        overwrite=True
                    )
                else:
                    print((
                        f'The provided filepath, {alignment_filepath}, was not found on the local file system. '
                        'Attempting to download from the internet.'
                    ),flush=True)
                    FASTAHandler.from_url(
                        f,
                        save_dir='./',
                        chromosomes=None,
                        max_workers=max_workers,
                        max_download_time=600,
                        max_time_to_wait_for_connection=5,
                        save_fasta=False,
                        is_gzip=f.suffix=='.gz',
                        filetype='h5',
                        compression='gzip',
                        overwrite=True
                    )
    
        # Raise an exception if the file doesn't exist. 
        if not alignment_filepath.is_file():
            if f.exists():
                raise Exception(
                    f'The provided alignment_filepath ({fp_og}) exists, but it appears not to be in the '
                    f'proper format (.h5 extension). However, the inferred equivalent .h5 file, {alignment_filepath}, '
                    'does not exist.'
                )
            elif f==alignment_filepath:
                raise Exception(f'Thre provided alignment_filepath, {fp_og}, does not exist.')
            elif f.suffix == '.h5':
                raise Exception(
                    f'The provided alignment_filepath, {fp_og}, does not exist. I attempted to '
                    f'fix the issue but changing the filepath to {alignment_filepath}, but it also does not exist.'
                )
            else:
                raise Exception(
                    f'The provided alignment_filepath ({fp_og}) does not exist. I guessed that the HDF5 '
                    f"would be located at {alignment_filepath}, but that doesn't exist, either."
                )
        
        return alignment_filepath

    @staticmethod
    def format_chrom_name(chrom_name):
        # Convert chromosome name to the 'chr1', 'chr2', ... convention used 
        # in this class
        return f"chr{str(chrom_name).replace('chr','')}"

    @staticmethod
    def validate_chrom_name(chrom_name):
        assert isinstance(chrom_name,(str,int)), \
        f'Expected chromosome name to be a string or integer. Received {type(chrom_name).__name__}'
        return _support.format_chrom_name(chrom_name)

    @staticmethod
    def sort_chroms(chrom_dict):
        numeric = []
        alpha = []
        for chrom in chrom_dict:
            try:
                numeric.append(int(chrom.replace('chr','')))
            except:
                alpha.append(chrom.replace('chr',''))
        numeric.sort()
        alpha.sort()
        sorted_names = [_support.format_chrom_name(chrom) for chrom in numeric + alpha]
        return {c:chrom_dict[c] for c in sorted_names}

    @staticmethod
    def format_chrom_dicts(chroms,validate_type=False):
        # Choose function
        format_fcn = _support.validate_chrom_name if validate_type else _support.format_chrom_name
    
        # Keys correspond to what's used internally for both of these
        # Rosetta values are whatever convention is used in the file rather than
        # what's used internally in this class. 
        chroms_internal = {}
        rosetta = {}
        for chrom,chrom_len in chroms.items():
            icf = format_fcn(chrom)
            chroms_internal[icf] = chrom_len
            rosetta [icf] = chrom
    
        # Sort the chromosomes in the dict that can be accessed by users
        chroms_internal = _support.sort_chroms(chroms_internal)
    
        return chroms_internal, rosetta
        

##########################################################################################
# The class
class SequenceLoader:

    def __init__(
        self,
        alignment_filepath,
        bigWig_filepath,
        dtype=torch.float,
        device=None,
        max_workers=None, # Note: BigWigHandler and H5GenomeFile perform loads in a single-threaded executor to prevent
                          # their files being access multiple times simultaneously... 
        store_in_memory=False, # Whether to hold the data in (CPU!) memory rather than accessing the file every time. 
        bigWig_nan_to=0,  # Replace NaN's in the bigWig file with this value when loaded. Use None to leave them as NaN's
        chroms=None,
        stack_output=True,   # Whether or not to stack the data
        auto_parse_fasta=False,
        interactive=False,
        verify_assembly=True
    ):
        # Verify and format inputs.
        # Setters handle formatting/verification for the public attributes being set throughout __init__ 
        alignment_filepath = _support.get_h5(alignment_filepath,auto_parse_fasta,interactive,max_workers)
        self.__h5_genome_file = H5GenomeFile(alignment_filepath)
        self.__bigWig_handler = BigWigHandler(bigWig_filepath)
        self.dtype = dtype
        self.device = device
        self.bigWig_nan_to = bigWig_nan_to 
        self.__executor = ThreadPoolExecutor(max_workers=max_workers)
        self.stack_output = stack_output
        self.__data_in_memory = None

        # Get list of chromosomes from each file and verify that they aren't empty
        self.__h5_chroms, self.__h5_rosetta = _support.format_chrom_dicts(self.h5_genome_file.chroms())
        self.__bw_chroms, self.__bw_rosetta = _support.format_chrom_dicts(self.bigWig_handler.chroms())
        assert self.__h5_chroms, f'No chromosomes found in the HDF5 alignment file, {alignment_filepath}.'
        assert self.__bw_chroms, f'No chromosomes found in the BigWig alignment file, {bigWig_filepath}.'

        # Get the set of chromosomes present in both files. Ensure there is overlap
        self.__intersecting_chroms = list(set(self.__h5_chroms).intersection(set(self.__bw_chroms)))
        assert self.__intersecting_chroms, (
            'The genome and BigWig files appear to share no chromosomes in common.\n'
            f'{alignment_file} contains: {list(self.__h5_chroms)}''\n'
            f'{bigWig_filepath} contains: {list(self.__bw_chroms)}'
        )

        # Ensure that chromosome lengths are the same in each file (if requested). This is a way to verify 
        # that both files utilize the same genome assembly. 
        if verify_assembly:
            mismatches = []
            h5c = self.__h5_chroms
            bwc = self.__bw_chroms
            ic = self.__intersecting_chroms
            # Sort since these will populate the exception message
            ic = _support.sort_chroms({c:None for c in ic})
            mismatches = [(c,h5c[c],bwc[c]) for c in ic if h5c[c]!=bwc[c]]
            if mismatches:
                mismatches.insert(0,('Chromosome','Length (sequence)','Length (bigWig)'))
                longest_c = 0
                longest_h5c = 0
                longest_bwc = 0
                for k,(c,h5c,bwc) in enumerate(mismatches):
                    c = str(c)
                    try:
                        h5c = f'{int(h5c):,}'
                    except:
                        h5c = str(h5c)
                    try:
                        bwc = f'{int(bwc):,}'
                    except:
                        bwc = str(bwc)
                    longest_c = max(longest_c,len(c))
                    longest_h5c = max(longest_h5c,len(h5c))
                    longest_bwc = max(longest_bwc,len(bwc))
                    mismatches[k] = (c,h5c,bwc)
                    
                err_msg = 'Are you sure your data files correspond to the same genome assembly?\n'
                if len(mismatches) == len(self.__intersecting_chroms)+1:
                    err_msg+= 'None of the chromosomes have the same length in either file.'
                else:
                    err_msg+= f'{len(mismatches)-1} chromosomes have different lengths.'
                err_msg+= ' They are:\n'
                for c,h5c,bwc in mismatches:
                    err_msg+= f'{c.ljust(longest_c)}    {h5c.ljust(longest_h5c)}    {bwc.ljust(longest_bwc)}''\n'
                raise Exception(err_msg)

        # Set chromosomes of this specific instance. Passing None will result in using self.__intersecting_chroms
        self.selected_chroms = chroms

        # Load data, if desired
        if store_in_memory:
            self.load_into_memory()
            
    #############################
    # Properties
    @property
    def in_memory(self):
        return self.data_in_memory is not None

    @property
    def data_in_memory(self):
        if self.__data_in_memory is None:
            return None
        return self.__data_in_memory.copy()

    @property
    def bigWig_handler(self):
        return self.__bigWig_handler

    @property
    def h5_genome_file(self):
        return self.__h5_genome_file

    @property
    def bigWig_nan_to(self):
        return self.__bigWig_nan_to

    @bigWig_nan_to.setter
    def bigWig_nan_to(self,val):
        if val is not None:
            try:
                val = float(val)
            except:
                raise Exception(
                    'bigWig_nan_to must be NoneType, float, or able to be converted to a float. '
                    f'The provided {type(val).__name__} instance cannot be.'
                )
        self.__bigWig_nan_to = val

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self,device):
        d = device
        if device is not None:
            try:
                d = torch.device(d)
            except:
                raise Exception(f'Requested device {device} is improperly formatted.')
            try:
                d = torch.empty(0,device=d).device
            except:
                raise Exception(f'Requested device {device} does not exist.')
        self.__bigWig_handler.device = d
        self.__h5_genome_file.device = d
        self.__device = d

    @property
    def dtype(self):
        return self.__dtype

    @dtype.setter
    def dtype(self,dtype):
        d = dtype
        if dtype is not None:
            try:
                d = torch.empty(0,dtype=d).dtype
            except:
                raise Exception(f'Requested dtype {dtype} is not recognized by PyTorch.')

        self.__bigWig_handler.dtype = d
        self.__h5_genome_file.dtype = d
        self.__dtype = dtype

    @property
    def stack_output(self):
        return self.__stack_output

    @stack_output.setter
    def stack_output(self,stack_output):
        assert isinstance(stack_output,bool), \
        f'stack_output must be a bool. Received {type(stack_output).__name__}.'
        self.__stack_output = stack_output


    @property
    def bigWig_chroms(self):
        return self.__bw_chroms.copy()

    @property
    def genome_chroms(self):
        return self.__h5_chroms.copy()

    # Somewhat mimic behavior of pyBigWig chroms method
    def chroms(self,*selected_chroms):
        if not selected_chroms:
            return self.__chroms.copy()

        if len(selected_chroms) == 1:
            c = _support.validate_chrom_name(selected_chroms[0])
            self.__contains__(c,_validate_format=False,_assert=True,_chrom_name=selected_chroms[0])
            return self.__chroms[c]

        to_return = {}
        for chrom_name in selected_chroms:
            c = _support.validate_chrom_name(chrom_name)
            try:
                self.__contains__(c,_validate_format=False,_assert=True,_chrom_name=chrom_name)
                to_return[c] = self.__chroms[c]
            except Exception as e:
                warnings.warn(str(e))

        return to_return if to_return else None

    @property
    def selected_chroms(self):
        # Setter appears later because it's huuuge
        return list(self.__chroms)
    
    @selected_chroms.setter
    def selected_chroms(self,chroms):
        
        # Ease of notation
        bw_chroms = self.__bw_chroms
        h5_chroms = self.__h5_chroms
        all_chroms = self.__intersecting_chroms

        if chroms is None:
            self.__chroms = {chrom:bw_chroms[chrom] for chrom in self.__intersecting_chroms}
        else:
            # Allow us to iterate regardless of input
            if not isinstance(chroms,(list,tuple)):
                chroms = [chroms]

            # Check if each chrom name is valid and format them. 
            # Use chroms1 name so that the original input can be used in any necessary exceptions
            chroms1 = [_support.format_chrom_name(chrom) for chrom in chroms]

            # Ensure each chromosome is actually available in the attached files
            invalid_chroms = []
            for k,c in enumerate(chroms1):
                if c not in all_chroms + invalid_chroms:
                    invalid_chroms.append(chroms[k])
            if invalid_chroms:
                ic = self.__intersecting_chroms
                if len(ic) < 3:
                    s = ' and/or '.join([ic])
                else:
                    s = ', '.join(ic[:-1]) + f', and/or {ic[-1]}'
                if len(invalid_chroms) < 3:
                    s1 = ' and '.join([invalid_chroms])
                else:
                    s1 = ', '.join(invalid_chroms[:-1]) + f', and {invalid_chroms[-1]}'
                ss = '' if len(invalid_chroms)==1 else 's'
                raise Exception(
                    f'Requested chromosome{ss} {s1} are absent from one or more of the sequence '
                    f'data files. Available options are {s}.''\nYou may access these using the convention '
                    'shown, with integers (where possible), or with strings that exclude the preceding "chr".'
                )
    
            # Drop any duplicates, fetch the relevant chromosome lengths, and sort the output in case 
            # the user wants to access the values. 
            self.__chroms = {chrom:bw_chroms[chrom] for chrom in set(chroms1)}

        # Sort the chromosomes in case the user accesses these later
        self.__chroms = _support.sort_chroms(self.__chroms)

        # Update the data being held in memory, if relevant. 
        if self.data_in_memory is not None:
            self.load_into_memory()

    def __contains__(self,chrom_name,_validate_format=True,_assert=False,_chrom_name=None):
        if _validate_format:
            if hasattr(chrom_name,'__iter__') and not isinstance(chrom_name,str):
                return type(chrom_name)(c in self for c in chrom_name)
            try:
                chrom_name1 = _support.validate_chrom_name(chrom_name)
            except:
                return False
        else:
            chrom_name1 = chrom_name
        is_in = chrom_name1 in self.chroms()
        if is_in or not _assert:
            return is_in

        # Format exception messages to agree with the value specifically inserted by the user. 
        if _chrom_name is not None:
            chrom_name = _chrom_name
        if chrom_name1 in self.__intersecting_chroms:
            raise Exception(
                f'While chromosome {chrom_name} is available in both data files, it is absent from the current '
                'list of chromosomes to analyze. It would have been removed based on the chroms argument '
                'to `SequenceLoader.__init__` or later by setting `sequence_loader.selected_chroms = <chrom(s) to include>`. '
                f'You may add it back in using `sequence_loader.selected_chroms = sequence_loader.selected_chroms + [{chrom_name}]` or, '
                'if you want ALL available chromosomes added back in, `sequence_loader.selected_chroms = None`.'
            )

        if chrom_name1 in self.bigWig_chroms:
            raise Exception(f'The genome alignment file does not contain chromosome {chrom_name}.')

        if chrom_name1 in self.genome_chroms:
            raise Exception(f'The BigWig file does not contain chromosome {chrom_name}.')

        raise Exception(f'Neither data file contains data for chromosome {chrom_name}.')
        
    #############################
    # Load/store data in memory
    def load_into_memory(self,storage_device='cpu',storage_dtype=None):
        # If max_workers is 1, submitting fetch from here will 
        # cause infinite hangup. 
        # If max_workers is 2, this submission would prevent
        # data from being loaded out of the two files in parallel. 
        use_executor = self.__executor._max_workers > 2

        # Don't stack the data to simplify the process of dealing with any
        # changes the user may make to self.stack_output by just making it consistent.
        # Similarly, don't modify the dtypes or nan values.
        # Also, because GPU memory is somewhat precious, hold the data int the CPU's memory. 
        # At least, that is the default behavior. User can set storage_device and storage_dtype. 
        stack_output = self.stack_output
        self.stack_output = False
        dtype = self.dtype
        self.dtype = storage_dtype
        device = self.device
        self.device = storage_device
        bigWig_nan_to = self.bigWig_nan_to
        self.bigWig_nan_to = None

        # Load all the data.
        # Note that, if data's already loaded, __fetch will return that data rather
        # than re-loading it from the file. 
        loaded_data = {}
        for chrom in self.__chroms:
            if use_executor:
                # Otherwise, this submission will interfere with the loads
                loaded_data[chrom] = self.__executor.submit(self.__fetch,chrom,0,-1)
            else:
                loaded_data[chrom] = self.__fetch(chrom,0,-1)
        
        # Get the future results, if necessary 
        if use_executor:
            loaded_data = {chrom:future.result() for chrom,future in loaded_data.items()}

        # Restore values
        self.stack_output = stack_output
        self.device = device
        self.dtype = dtype
        self.bigWig_nan_to = bigWig_nan_to

        # Set the relevant data attribute
        self.__data_in_memory = loaded_data

    # Similarly, clear memory
    def clear(self):
        self.__data_in_memory.clear()
        del self.__data_in_memory
        self.__data_in_memory = None
        
    def __fetch(self,chromosome,start,end):
        dim = self.data_in_memory
        dim = {} if dim is None else dim
        if dim is not None and chromosome in dim:
            ad, bwd = dim[chromosome]

            # Index the data
            if not (start==0 and end==-1):
                ad = ad[...,start:end]
                bwd = bwd[...,start:end]

            # bwd didn't previously have its NaN's removed, 
            # per the standard logic for data stored in memory in case the 
            # user changes self.bigWig_nan_to, so update that now. 
            if self.bigWig_nan_to is not None:
                bwd = bwd.nan_to_num(self.bigWig_nan_to)

            # Data held in memory lives on some specific CPU's RAM. 
            # So, move the requested components to whichever device is now desired.
            # Similarly, the dtypes may disagree with what the user is now requesting.
            ad = ad.to(device=self.device,dtype=self.dtype)
            bwd = bwd.to(device=self.device,dtype=self.dtype)

            # If the 
        else:
            alignment_data_future = self.__executor.submit(
                self.h5_genome_file.fetch,
                self.__h5_rosetta[chromosome],
                start,
                end
            )
            bigWig_data_future = self.__executor.submit(
                self.bigWig_handler.fetch,
                self.__bw_rosetta[chromosome],
                start,
                end,
                nan_to=self.bigWig_nan_to
            )
    
            ad = alignment_data_future.result()
            bwd = bigWig_data_future.result()
                
        if self.stack_output:
            # In case dtype is None, ad and bwd will be bool and double, respectively. 
            # Give them the same dtype before concatenating
            ad = ad.to(bwd.dtype)
            # Note that bwd is returned without the ATCG-DNase-seq indexing dimension,
            # so we must create it. 
            return torch.cat([ad,bwd.unsqueeze(-2)],dim=-2)

        # Otherwise, return data separately. 
        return ad, bwd
    
    def fetch(self,chromosome,start,end):
        chromosome1 = _support.validate_chrom_name(chromosome)
        # Verify that this chromosome is currently available. 
        self.__contains__(chrom_name=chromosome1,_validate_format=False,_assert=True,_chrom_name=chromosome)
        chromosome = chromosome1
        if isinstance(start,float) and int(start) == start:
            start = int(start)
        assert isinstance(start,int), f'start index should be an integer. Received {type(start).__name__}.'
        if isinstance(end,float) and int(end) == end:
            end = int(end)
        assert isinstance(end,int), f'end index should be an integer. Received {type(end).__name__}.'

        # Ensure the indices are within bounds. Otherwise, errors 
        # can be difficult to catch. 
        # Note that bwd.shape[-1] is EXACTLY the length 
        # of the chromosome. 
        assert self.chroms(chromosome) >= end, (
            f'End index {end} is out of bounds for chromosome {chromosome}, which has '
            f'{int(self.chroms(chromosome)):,} base pairs.'
        )
        
        return self.__fetch(chromosome,start,end)
        

