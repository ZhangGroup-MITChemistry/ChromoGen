'''
Greg Schuette 2024
'''
import torch
from concurrent.futures import ThreadPoolExecutor
from .SequenceLoader import SequenceLoader, _support
import random
import numpy as np
import warnings

def _validate_int(x,arg_name,minimum=None,maximum=None):
    # Check type. Floats are fine so long as they're an int. 
    if isinstance(x,float):
        x = int(x) if int(x) == x else x
    assert isinstance(x,int), f'Expected {arg_name} to be an integer. Received {type(x).__name__}.'

    # Check bounds
    if has_min:= ( minimum is not None ):
        minimum = _validate_int(minimum,'minimum')
    if has_max:= ( maximum is not None ):
        maximum = _validate_int(maximum,'maximum')

    if has_min:
        if has_max:
            assert minimum <= maximum, f'minimum must be <= maximum. Received {minimum} > {maximum}.'
            assert minimum <= x <= maximum, \
            f'{arg_name} must lie in the range [{minimum},{maximum}]. Received {x}.'
        elif x < minimum:
            if minimum == 0:
                err_msg = f'{arg_name} must be positive-valued or 0. Received {x}.'
            elif minimum == 1:
                err_msg = f'{arg_name} must be positive-valued. Received {x}.'
            else:
                err_msg = f'{arg_name} must be >= {minimum}. Received {x}.'
            raise Exception(err_msg)
    elif has_max:
        if x > maximum:
            if maximum == 0:
                err_msg = f'{arg_name} must be negative-valued or 0. Received {x}.'
            elif maximum == -1:
                err_msg = f'{arg_name} must be negative-valued. Received {x}.'
            else:
                err_msg = f'{arg_name} must be <= {maximum}. Received {x}.'
            raise Exception(err_msg)
    
    return x


def _prep_lengths(resolution, num_bins, region_length, pad_size):
    resolution = _validate_int(resolution,'resolution',minimum=1)
    if num_bins is not None:
        num_bins = _validate_int(num_bins,'num_bins',minimum=1)
    if region_length is not None:
        region_length = _validate_int(region_length,'region_length',minimum=1)
    if pad_size is None:
        pad_size = 0
    else:
        pad_size = _validate_int(pad_size,'pad_size',minimum=0)

    assert pad_size <= resolution, f'pad_size {pad_size:,} must be <= resolution {resolution:,}.'

    if region_length is None:
        if num_bins is None:
            region_length = resolution
            num_bins = 1
            warnings.warn(
                'Neither region_length nor num_bins was specified, so I am assuming '
                f'num_bins={num_bins} and region_length=resolution={region_length:,}.'
            )
        else:
            region_length = resolution * num_bins
    else:
        assert region_length%resolution == 0, \
        f'The region_length ({region_length:,}) must be divisible by the resolution ({resolution:,}).'
        nbins1 = region_length//resolution
        if num_bins is None:
            num_bins = nbins1
        else:
            assert num_bins == nbins1, (
                'Incompatible inputs! '
                f'At resolution {resolution:,}, the specified num_bins should be {nbins1:,}. '
                f'However, the specified num_bins is {num_bins:,}.'
            )

    return resolution, num_bins, region_length, pad_size
    

class EPCOTInputLoader(SequenceLoader):

    def __init__(
        self,
        alignment_filepath,
        bigWig_filepath,
        #########################
        ### THE ONLY NEW ARGS ###
        # All others for SequenceLoader
        resolution: int,
        num_bins=None,      # None or positive int. Either this or region_length must be defined.
        region_length=None, # None or int. NONE or INT
        pad_size=None, # or integer. Default: 0
        start_indices=None, # Mostly for reverse-compatibility
        allow_bin_overlap=False,   # Can the bins defined by resolution overlap? 
        allow_region_overlap=True, # Can the larger regions (resolution * num_bins) overlap? Overrides allow_bin_overlap if False
        batch_size=1,              # How many 'regions' to return at once. 
        shuffle=False,             # Whether to shuffle the indices between epochs
        #########################
        dtype=torch.float,
        device=None,
        max_workers=None, # Note: BigWigHandler and H5GenomeFile perform loads in a single-threaded executor to prevent
                          # their files being access multiple times simultaneously... 
        store_in_memory=False, # Whether to hold the data in (CPU!) memory rather than accessing the file every time. 
        bigWig_nan_to=0,  # Replace NaN's in the bigWig file with this value when loaded. Use None to leave them as NaN's
        chroms=None,
        auto_parse_fasta=False,
        interactive=False,
        verify_assembly=True
    ):

        # We'll pass store_in_memory=False to super().__init__ so that we can more easily set the index of starting positions
        # before parsing the chromosomes actually being accessed, then ONLY load the relevant chromosomes into memory (if requested).
        assert isinstance(store_in_memory,bool), f'store_in_memory should be a bool. Received {type(store_in_memory).__name__}.'

        # Also use max_workers here
        self.__max_workers = max_workers if max_workers is None else int(max_workers) # Already validated above, but just make sure it's in int form

        # Take care of batch-specific values
        assert isinstance(allow_bin_overlap,bool), f'allow_bin_overlap must be a bool. Received {type(allow_bin_overlap).__name__}.'
        self.__allow_bin_overlap = allow_bin_overlap 
        
        assert isinstance(allow_region_overlap,bool), \
        f'allow_region_overlap must be a bool. Received {type(allow_region_overlap).__name__}.'
        self.__allow_region_overlap = allow_region_overlap
        
        (
            self.__resolution,
            self.__num_bins,
            self.__region_length,
            self.__pad_size
        )= _prep_lengths(resolution, num_bins, region_length, pad_size)

        '''

        if num_bins is not None:
            self.__num_bins = _validate_int(num_bins,'num_bins',minimum=1)
            if region_length is None:
                self.__region_length = _validate_int(num_bins,'num_bins',minimum=1)
                assert self.resolution*self.num_bins==self.region_length, ''
        
        assert (
            (resolution is not None and (num_bins is not None or region_length is not None)) or
            ()
        )
        
        if region_length is not None:
            self.__region_length = _validate_int(region_length,'region_length',minimum=1)
            assert self.region_length%self.resolution == 0, \
            f'region_length ({self.region_length:,}) must be divisible by the resolution ({self.resolution:,}).'
        if num_bins is not None:
            self.__num_bins = _validate_int(num_bins,'num_bins',minimum=1)
            
        if region_length is not None and num_bins is not None:
            assert self.resolution * self.num_bins == self.region_length, (
                f'When specifying both region_length and num_bins, they must agree. However, region_length := {self.region_length:,} '
                f'!= {self.resolution*self.num_bins:,} = {self.resolution:,} x {self.num_bins:,} =: resolution x num_bins.'
            )
        elif region_length is None:
            self.__region_length = self.resolution * self.num_bins
        elif num_bins is None:
            self.__num_bins = self.region_length // self.resolution
        else:
            raise Exception('Must define either region_length or num_bins.')        
      
        if pad_size is None:
            self.__pad_size = 0
        else:
            self.__pad_size = _validate_int(pad_size,'pad_size',minimum=0)
            assert self.pad_size < self.resolution, f'pad_size ({self.pad_size:,}) cannot exceed resolution ({self.resolution:,}).'
        '''
        assert isinstance(shuffle,bool), f'shuffle should be a bool. Received {type(shuffle).__name__}.'
        self.shuffle = shuffle

        self.__user_provided_start_indices = {}
        self.__default_start_indices = {}

        # Initialize this overall before using setters within this specific class
        super().__init__(
            alignment_filepath=alignment_filepath,
            bigWig_filepath=bigWig_filepath,
            dtype=dtype,
            device=device,
            max_workers=max_workers,
            store_in_memory=False,
            bigWig_nan_to=bigWig_nan_to,
            chroms=None, # Will be set after computing the default indices
            stack_output=True,
            auto_parse_fasta=auto_parse_fasta,
            interactive=interactive,
            verify_assembly=verify_assembly
        )

        
        self.start_indices = start_indices
        if start_indices is not None and chroms is not None:
            # Verify that there isn't a conflict between the provided chroms and the provided index (if relevant)
            # Otherwise, the user's chromosome selection will override the start_indices selection if relevant...
            # That could be fine, but it's likely unexpected behavior to the user, so simply throw an exception. 
            try:
                self.__contains__(chroms, _validate_format=True, _assert=True)
            except Exception as e:
                raise Exception(
                    'Some chromosomes listed in chroms do not appear in start_indices. '
                    'If this was intentional, please set `epcot_input_loader.selected_chroms = chroms` AFTER '
                    'the EPCOTInputLoader is initialized.\n\nIn case a deeper issue caused this Exception in the lower-level '
                    'methods, here is some additional traceback:\n'f'{e}'
                )

        self.selected_chroms = chroms

        if store_in_memory:
            self.load_into_memory()

    ##############################
    # Just a couple extra properties
  
    @property
    def resolution(self):
        return self.__resolution

    @property
    def num_bins(self):
        return self.__num_bins

    @property
    def region_length(self):
        return self.__region_length

    @property
    def pad_size(self):
        return self.__pad_size

    @property
    def allow_bin_overlap(self):
        return self.__allow_bin_overlap

    @property
    def allow_region_overlap(self):
        return self.__allow_region_overlap

    @property
    def start_indices(self):
        return self.__start_indices.copy()

    def start_indices_are_custom(self):
        si = self.start_indices
        dsi = self.__default_start_indices
        return si == {key:dsi[key] for key in si}

    __si_err_msg_preamble = (
        'Expected start_indices to be a dict whose keys are chromosome names and values are '
        'either NumPy ndarrays, torch Tensors, or lists populated with integers.'
    )
    @start_indices.setter
    def start_indices(self,start_indices):
        err_msg = (
            'Expected start_indices to be a dict whose keys are chromosome names and values are '
            'list, flat torch.Tensor, or flat numpy.ndarray containing integer start positions. '
        )
      
        if start_indices is not None:
            # Must be a non-empty dictionary
            assert isinstance(start_indices,dict), err_msg + f'Received {type(start_indices).__name__}.'
            assert start_indices, f'start_indices cannot be empty.'

            # Update the available chromosomes according to this dictionary. Raises exception if any are unavailable
            self.selected_chroms = list(start_indices)

            # Convert the chromosome names to those used internally and verify the indices provided. 
            si = self.__user_provided_start_indices
            for chrom_name, start_idx in start_indices.items():
                if isinstance(start_idx,np.ndarray):
                    try:
                        start_idx = torch.from_numpy(start_idx)
                    except:
                        raise Exception(
                            err_msg + f'However, the np.ndarray passed for chromosome {chrom_name} cannot be '
                            'converted to a torch.Tensor.'
                        )
                elif isinstance(start_idx,list):
                    try:
                        start_idx = torch.tensor(start_idx)
                    except:
                        raise Exception(
                            err_msg + f'However, the list passed for chromosome {chrom_name} cannot be '
                            'converted to a torch.Tensor.'
                        )
                assert isinstance(start_idx,torch.Tensor), \
                err_msg + f'Received {type(start_idx).__name__} for chromosome {chrom_name}.'

                # Allow a single value to count, as well... 
                assert start_idx.ndim<=1, \
                err_msg + f'However, the value for chromosome {chrom_name} has {start_idx.ndim} dimensions (not flat).'
                start_idx = start_idx.flatten() # ndim\in{0,1} -> ndim==1

                assert torch.equal(start_idx.int(),start_idx), \
                err_msg + f'However, the value for chromosome {chrom_name} contains non-integer values.'

                assert start_idx.isfinite().all(), \
                err_msg + f'However, the start index (value) for chromosome {chrom_name} contains non-finite values.'
              
                assert start_idx.min() >= 0, \
                f'All start indices must be >= 0, but the provided chromosome {chrom_name} index has minimum {start_idx.min()}.'

                if start_idx.max() + self.region_length > self.chroms(chrom_name):
                    cl = self.chroms(chrom_name)
                    n = ( start_idx + self.region_length > cl ).sum()
                    max_i = cl - self.region_length
                    raise Exception(
                        f'Chromosome {chrom_name} has length {self.chroms(chrom_name)}, and the regions to load have length '
                        f'{self.region_length} (resolution={self.resolution}, num_bins={self.num_bins}). As such, the maximum '
                        f'start index should not exceed {max_i}, but {n} of the provided values {"does" if n==1 else "do"}.'
                    )

                assert len(start_idx.unique()) == start_idx.numel(), f'Found repeated values in start indices for chromosome {chrom}.'

                # Finally verified!
                si[_support.format_chrom_name(chrom_name)] = start_idx.tolist()

            # Set the indices & store user's selections in case they swap around selected chromosomes later on. 
            # DON'T make the order pretty in case the user's very specifically providing a particular ordering. 
            self.__start_indices = si 
            self.__user_provided_start_indices.update(si)

        else:
            self.__start_indices = si = {}
            for chrom_name in self.selected_chroms:
                if not (si1:= self.__user_provided_start_indices.get(chrom_name)):
                    if not (si1:= self.__default_start_indices.get(chrom_name)):
                        stride = 1 if self.allow_bin_overlap else self.resolution
                        stride = stride if self.allow_region_overlap else self.region_length
                        si1 = torch.arange(0,self.chroms(chrom_name),stride)
                        self.__default_start_indices[chrom_name] = si1.tolist()
                si[chrom_name] = si1

        self.reset_index()

    @property
    def iteration_index(self):
        return self.__index.copy()

    @iteration_index.setter
    def iteration_index(self,new_idx):
        assert isinstance(new_idx,list), f'Index must be a list of tuples. Received {type(new_idx).__name__}.'
        assert new_idx, 'Index cannot be empty.'
        start_indices = {}
        for k,entry in enumerate(new_idx):
            assert isinstance(entry,tuple), \
            f'Index must be a list of tuples. Received {type(entry).__name__} in position {k}.'
            assert len(entry) == 2 and isinstance(entry[0],(str,int)), \
            'Each tuple must have exactly two elements: The chromosome name and the start position in bp.'
            chrom = _support.format_chrom_name(entry[0])
            try:
                start = _validate_int(entry[1], f'start index at position {k}', minimum = 0)
            except Exception as e:
                raise Exception(
                    'Each tuple must have exactly two elements: The chromosome name and the start position in bp.\n'
                    f'More specific exception: {e}'
                )
            start_indices[chrom] = start_indices.get(chrom,[]) + [start]
        self.start_indices = start_indices

    # Override selected_chroms setter so that the start indices are also updated
    @SequenceLoader.selected_chroms.setter
    def selected_chroms(self,chroms):
        SequenceLoader.selected_chroms.fset(self,chroms)
        self.start_indices = None # Will use user-provided indices where possible, default where necessary

    ##############################
    # Iteration support
    def reset_index(self):
        self.__index = []
        for chrom,start_idx in self.start_indices.items():
            self.__index.extend([
                (chrom,start) for start in start_idx
            ])
        if self.shuffle:
            random.shuffle(self.__index)
        self.step = 0

    def __iter__(self):
        # Use the executor to get stuff loading/formatting in parallel so that 
        # its ready-to-go for the user. 
        # Actually, going to move towards something else... but will leave that option commented in case I come back to it. 
        assert self.iteration_index, f'The iteration_index appears to be empty! This is likely an issue with the class.'
        if self.step >= len(self):
            self.reset_index()
        self.step = self.step
        while self.step < len(self):
            chrom,start = self.iteration_index[self.step]
            data = self.fetch(chrom,start,validate_input=False)
            self.step+=1
            yield chrom, start, data
        '''
        with ThreadPoolExecutor(max_workers=self.__max_workers) as executor:
            chrom,start = self.iteration_index[self.step]
            to_return = [(chrom,start,executor.submit(self.fetch,chrom,start,validate_input=False))]
            for k,(chrom,start) in enumerate(self.iteration_index[self.step+1:]):
                to_return.append((chrom,start,executor.submit(self.fetch,chrom,start,validate_input=False)))
                if len(to_return) < executor._max_workers + 1:
                    continue

                chrom, start, future = to_return.pop(0)
                self.step+=1
                yield chrom, start, future.result()

            while to_return:
                chrom, start, future = to_return.pop(0)
                self.step+=1
                yield chrom, start, future.result()
        '''
        
    def __len__(self):
        return len(self.iteration_index)
    
    ##############################
    # Retrieve data

    # Override the fetch function to reshape data as needed in EPCOT
    def fetch(
        self,
        chrom,
        start,
        end=None,
        *,
        num_bins=None,
        region_length=None,
        resolution=None,
        pad_size=None,
        validate_input=True # If False, end will be selected from the first of end, num_bins, region_length, or resolution to be defined
                            # If none of those, will use the internal resolution to infer end
    ):

        ##############################
        # Verify/format inputs (if requested... will risk harder-to-trace exceptions if not) 

        if validate_input:
            # Format the provided chromosome name and ensure that it's in the list of selected chromosomes
            # Don't redefine chrom yet in case it's needed for an assertion message. 
            chrom1 = _support.format_chrom_name(chrom)
            self.__contains__(chrom1, _validate_format=False, _assert=True, _chrom_name=chrom)
    
            # Resolution can be specified, or it can be set to the original value
            resolution = self.resolution if resolution is None else _validate_int(resolution,'resolution',1)
            start = _validate_int(start,'start index',minimum=0)
            end_indices = {}
            if end is not None:
                end = _validate_int(end,'end',minimum=1)
                end_indices['end'] = end
            if num_bins is not None:
                num_bins = _validate_int(num_bins,'num_bins',1)
                end_indices['num_bins'] = start + num_bins*resolution
            if region_length is not None:
                region_length = _validate_int(region_length,'region_length',1)
                end_indices['region_length'] = start + region_length
    
            if end_indices:
                ends = {rl for rl in end_indices.values()}
                if len(ends) > 1:
                    err_msg = ['Incompatible inputs!']
                    if e:= end_indices.get('end'):
                        err_msg.append(
                            'The provided start and end indices imply that region_length should be '
                            f'{e:,} - {start:,} = {e-start:,}.'
                        )
                    if e:= end_indices.get('num_bins'):
                        err_msg.append(
                            'The provided or pre-set resolution and provided num_bins imply that '
                            f'region_length should be {num_bins:,} x {resolution:,} = {e-start:,}.'
                        )
                    if e:= end_indices.get('region_length'):
                        err_msg.append(
                            'The provided region_length is {region_length:,}.'
                        )
                    err_msg[-1] = 'Meanwhile, t' + err_msg[-1][1:]
                    err_msg.append(
                        'Either correct the discrepancy or pass just one of these values so that the '
                        'others may simply be inferred.'
                    )
                    raise Exception('\n'.join(err_msg))
    
                end = ends.pop()
                assert end <= self.chroms(chrom), \
                f'The specified (or inferred) end index, {end:,}, exceeds the length of chromosome {chrom}, {self.chroms(chrom):,}.'
                assert (end-start)%resolution == 0, (
                    f'The specified (or inferred) region_length, {end-start:,}, is not divisible by the specified '
                    f'(or pre-initialized) resolution, {resolution:,}.'
                )
            else:
                end = start + resolution * self.num_bins # Keep same number of bins even if resolution has changed
    
            pad_size = self.pad_size if pad_size is None else _validate_int(pad_size,'pad_size',minimum=0)
            assert pad_size <= resolution, \
            f'pad_size must be <= resolution, but have pad_size={pad_size:,} and resolution={resolution:,}.'

            # Rename chrom now that we're done throwing exceptions
            chrom = chrom1
        else:

            # Whether or not a user passes values for them, we need resolution, pad_size, 
            # and end for the following computation. So, use the values provided at initialization
            # if needed. 
            resolution = self.resolution if resolution is None else resolution
            pad_size = self.pad_size if pad_size is None else pad_size

            if end is None:
                if region_length is None:
                    end = start + (self.num_bins if num_bins is None else num_bins) * resolution 
                else:
                    end = start + region_length

        ##############################
        # Retrieve and format the data

        # If padding is used and this region is close enough to the start and/or end of 
        # the chromosome that we can't pad the first/last bin exclusively with actual genomic data, 
        # then we'll pad with extra zeros. Decide how may to add at the start/end of the region here. 
        artificial_start_padding = 0
        artificial_end_padding = 0
        ps = pad_size
        if ps > 0:
            start1 = start - resolution
            if start1 < 0:
                start = 0
                artificial_start_padding = abs(start1)
            else:
                start = start1
            end1 = end + resolution
            if end1 > self.chroms(chrom):
                end = self.chroms(chrom)
                artificial_end_padding = end1-end
            else:
                end = end1

        # Load the data, including as much padding data as possible. 
        genome_data = self._SequenceLoader__fetch(chrom,start,end) # Should be fully validated at this point

        # Add any additional padding that may be needed
        s = genome_data.shape
        if artificial_start_padding or artificial_end_padding:
            genome_data = torch.cat(
                [
                    torch.zeros(*s[:-1],artificial_start_padding),
                    genome_data,
                    torch.zeros(*s[:-1],artificial_end_padding)
                ],
                dim=-1
            )

        # Reshape into a bunch of bins of the correct size
        genome_data = genome_data.reshape(*s[:-1],-1,resolution)
        if ps > 0:
            # If padding is being used, add it here
            genome_data = torch.cat(
                [
                    genome_data[...,:-2,-ps:],
                    genome_data[...,1:-1,:],
                    genome_data[...,2:,:ps]
                ],
              dim=-1
            )
        # No else needed because start/end isn't extended to include extra bins when ps == 0. 

        # To match the original EPCOT data dimensions, we must 
        # transpose the bin index and A-T-C-G-DNaseSeqValue dimensions
        genome_data = genome_data.transpose(-2,-3)

        return genome_data

