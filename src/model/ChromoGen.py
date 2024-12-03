'''
Greg Schuette 2024
'''
import torch
from torch import nn
from pathlib import Path
import warnings
import time
import copy
from tqdm.auto import tqdm 
from .Diffuser.ChromoGenDiffuser import ChromoGenDiffuser
from .Embedder.EPCOT import EPCOT
from ..data_utils.EPCOTInputLoader import EPCOTInputLoader
from ..Conformations._Distances import Distances
from ._ResourceManager import ResourceManager

########################################################################################################################
# Support functions

#######
# These are repeatedly used in the class
def pprint(message,flush=True,silent=False):
    if not silent:
        print(message,flush=flush)
    
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

def _validate_float(x,arg_name,arg_name1=None,minimum=None,maximum=None):
    try:
        return float(x)
    except:
        arg_name1 = arg_name if arg_name1 is None else arg_name1
        raise Exception(
            f'{arg_name} should be a float or convertible to floats. '
            f'{arg_name1} of type {type(xx).__name__} cannot be converted to a float.'
        )

def _validate_list_of_floats(x,arg_name,minimum=None,maximum=None,empty_ok=False):

    if minimum is not None:
        minimum = _validate_float(minimum,'minimum')
    if maximum is not None:
        maximum = _validate_float(maximum,'maximum')
    
    if not isinstance(x,(list,tuple)):
        x = [x]
        
    if not empty_ok:
        assert x, f'{arg_name} cannot be empty'
    
    xs = [
        _validate_float(xx,arg_name,f'Entry {k}') for k,xx in enumerate(x)
    ]

    if minimum is not None:
        assert all(x>=minimum for x in xs), \
        f'Minimum value permitted for {arg_name} entries is {minimum}. Received {min(xs)}.'
    if maximum is not None:
        assert all(x<=maximum for x in xs), \
        f'Maximum value permitted for {arg_name} entries is {maximum}. Received {max(xs)}.'

    return xs

#######
# The following are only used once (in ChromoGen.sample()), but I'm moving them here since 
# they're LONG and make the code harder to follow if not separated. 
def _validate_sample_parameters(
    x,
    samples_per_region,
    return_coords,
    correct_distmap,
    cond_scales,
    rescaled_phis,
    proportion_from_each_scale,
    force_eval_mode,
    distribute
):
    
    if not isinstance(x,torch.Tensor):
        try:
            x = _validate_int(x,'x',minimum=1)
            samples_per_region = x
        except:
            raise Exception(
                'x should be an integer (number of unguided samples to produce) or a torch.Tensor '
                f'(sequencing input) instance. Received {type(x).__name__}.'
            )
    else:
        while x.ndim < 3:
            x = x.unsqueeze(0)

    assert isinstance(distribute,bool), \
    f'distribute should be a bool. Received {type(distribute).__name__}.'
    
    samples_per_region = _validate_int(samples_per_region,'samples_per_region',minimum=1)
    num_samples = _validate_int(samples_per_region,'samples_per_region',minimum=1)
    assert isinstance(return_coords,bool), f'return_coords should be a bool. Received {type(return_coords).__name__}.'
    assert isinstance(correct_distmap,bool), f'correct_distmap should be a bool. Received {type(correct_distmap).__name__}.'
    assert isinstance(force_eval_mode,bool), f'force_eval_mode should be a bool. Received {type(force_eval_mode).__name__}.'
    
    cond_scales = _validate_list_of_floats(cond_scales,'cond_scales')
    rescaled_phis = _validate_list_of_floats(rescaled_phis,'rescaled_phis')

    if len(rescaled_phis) != len(cond_scales):
        if len(rescaled_phis) == 1:
            rescaled_phis*= len(cond_scales)
        elif len(cond_scales) == 1:
            cond_scales*= len(rescaled_phis)
        else:
            raise Exception(
                'You must provide the same number of values in cond_scales and rescaled_phis. Alternatively, you can '
                'provide ONE value of one of them to apply it to all variations of the other. '
                f'Received {len(cond_scales)} cond_scale values and {len(rescaled_phis)} rescaled_phi values.'
            )

    if proportion_from_each_scale is None:
        proportion_from_each_scale = [1]
    else:
        proportion_from_each_scale = _validate_list_of_floats(proportion_from_each_scale,'proportion_from_each_scale',minimum=0)
    if len(proportion_from_each_scale) == 1:
        proportion_from_each_scale*= len(rescaled_phis)
    else:
        assert len(proportion_from_each_scale) == len(cond_scale), (
            'When proportion_from_each_scale is defined, it must have the same number of entries as cond_scales '
            'and rescaled_phis OR just one entry (equal weight everywhere, equivalent to passing None). '
            f'However, received {len(cond_scales)} cond_scale/rescaled_phi combinations and '
            f'{len(proportion_from_each_scale)} entries in proportion_from_each_scale.'
        )
    assert (s:= sum(proportion_from_each_scale)) > 0, \
    f'The sum of entries in proportion_from_each_scale must be positive-valued. Received {s}.'

    samples_per_scale = {}
    for k,cs in enumerate(cond_scales):
        key = (cs,rescaled_phis[k])
        p = proportion_from_each_scale[k]
        if key in samples_per_scale:
            samples_per_scale[key]+= p
            warnings.warn(
                f'cond_scale={cs}, rescaled_phi={key[-1]} combination was repeated. '
                'Combining the weights associated with each appearance to determine the '
                'overall weight to apply to this weighting/scaling combination. Please '
                'ensure this is the desired behavior.'
            )
        else:
            samples_per_scale[key] = p

    # convert proportion to actual number of samples
    for key,weight in samples_per_scale.copy().items():
        if weight == 0:
            warnings.warn(
                f'The relative weight applied to cond_scale={key[0]} and rescaled_phi='
                f'{key[1]} is 0. Removing from the list of combinations to generate.'
            )
            samples_per_scale.pop(key)
            continue
        n = round(samples_per_region * weight / s)
        if n == 0:
            warnings.warn(
                f'The proportion of samples to generate with cond_scale={key[0]} and rescaled_phi='
                f'{key[1]}, {weight/s:.4e}, rounds to 0 samples with total sample equal to '
                f'{"x" if isinstance(x,int) else "samples_per_region"}={samples_per_region}. '
                'Artificially increasing this to 1.'
            )
            n = 1
        samples_per_scale[key] = n
    n = sum(v for v in samples_per_scale.values())
    if n != samples_per_region:
        item_list = [('cond_scale','rescaled_phi','samples_to_generate')]
        m = len('samples_to_generate')
        for (cs,rp),n in samples_per_scale.items():
            cs = str(cs)
            rp = str(rp)
            n = str(n)
            m = max(m,len(cs),len(rp),len(n))
            item_list.append((cs,rp,n))
        item_list = '\n'.join([f'{cs.ljust(m)}    {rp.ljust(m)}    {n.ljust(m)}' for cs,rp,n in item_list])
        warnings.warn(
            f'The combination of proportion_from_each_scale and {"x" if isinstance(x,int) else "samples_per_region"} '
            'values yields the following breakdown of samples per cond_scale/rescaled_phi combination:\n' + item_list
        )

    return x, samples_per_scale

def _organize_samples(conformations, generative_steps, x_shape_original):

    # Convert from the generative_steps indexing scheme to the (flattened) order of 
    # of each region passed by the user. 
    results = {}
    for k,(cond_scale,rescaled_phi,n_samples,i,j) in enumerate(generative_steps):
        for jj,ii in enumerate(range(i,min(j,conformations[k].shape[0]))):
            if ii not in results:
                results[ii] = {}
            if cond_scale not in results[ii]:
                results[ii][cond_scale] = {}
            if rescaled_phi not in results[ii][cond_scale]:
                results[ii][cond_scale][rescaled_phi] = []
            results[ii][cond_scale][rescaled_phi].append(conformations[k][ii])
    #return results
    # Always organize in the same order used with the inputted embeddings
    conformations = []
    region_ids = list(results)
    region_ids.sort()
    for rid in region_ids:
        conformations1 = []
        # Within each region, always organize conformations by increasing cond_scale 
        # (in case someone wants to extract that information later)
        region_results = results[rid]
        cond_scales = list(region_results)
        cond_scales.sort()
        for cs in cond_scales:
            # Within each cond_scale, always organize conformations by increasing rescaled_phi 
            # (in case someone wants to extract that information later)
            cs_results = region_results[cs]
            rescaled_phis = list(cs_results)
            rescaled_phis.sort()
            for rp in rescaled_phis:
                # Note that this indexing removed dim 0, BUT because it corresponds to a sub-index
                # of the region IDs, we still want to concatenate rather than stack below.
                conformations1.extend(cs_results[rp])

        # Combine the sorted conformations for this particular region together
        conformations.append(conformations1[0].cat(conformations1[1:],dim=0))

    # Make dim 0 the region_id index, reshape to match the original data passed here, and return
    # Note that one additional dimension was added to index the generated samples, so we must 
    # account for that when calling reshape. 
    conformations = conformations[0].stack(conformations[1:])
    
    # Have yet to build reshape into the relevant classes, so... use the variable that users aren't
    # _supposed_ to use so that we can reshape using torch
    if x_shape_original is not None:
        conformations._values = conformations.values.reshape(*x_shape_original[:-3],*conformations.shape[1:])
    return conformations

########################################################################################################################
# The class
class ChromoGen(nn.Module):

    def __init__(
        self,
        front_end = None, 
        diffuser = None,
        data_loader = None,
        gpus_to_use = 'detect', # To distribute sampling across multiple GPUs. 
                                # WON'T use torch.distributed since that's forbidden on our cluster for some reason
        maximum_samples_per_gpu=1000, # If distributing across GPUs, don't create enough replicas such 
                                      # that the batch size at each replica is less than this.
        maximum_regions_embedded_per_GPU=6, # This is WAY more memory intensive than the diffusion model
        max_gpus_to_use = None,
        home_cpu='cpu:0',
        maximum_conformations_per_call=1_000_000,
        requires_grad=False,
        training_mode=False,
        _internal=False
    ):
        super().__init__()

        # These are easy to validate and aren't used until all the hard stuff is done, so do now
        assert isinstance(requires_grad, bool), \
        f'requires_grad should be a bool. Received {type(requires_grad).__name__}'
        assert isinstance(training_mode, bool), \
        f'training_mode should be a bool. Received {type(training_mode).__name__}'
        
        # Will broadcast across GPUs when possible to accelerate generation.
        # These support that effort
        self.__set_available_gpus()
        self.max_gpus_to_use = max_gpus_to_use
        self.gpus_to_use = gpus_to_use
        self.__replica_id = 0 

        # After generating in batches, all conformations are collected on the same CPU's RAM so 
        # that they can be concatenated. This determines which one. 
        self.home_cpu = home_cpu

        # In case the user makes a mistake that would cause an obscene amount of generation, 
        # this will limit the number of conformations generated. 
        self.maximum_conformations_per_call = maximum_conformations_per_call

        # Primarily due to RAM limitations, may want to limit the number of regions being
        # embedded via EPCOT's front end or number of conformations diffused at one time
        # on each GPU. 
        self.maximum_samples_per_gpu = maximum_samples_per_gpu
        self.maximum_regions_embedded_per_GPU = maximum_regions_embedded_per_GPU
        
        # Uses default values if the models are passed as None. 
        if front_end is None:
            self.front_end = EPCOT()
        elif isinstance(front_end,EPCOT):
            self.front_end = front_end
        else:
            raise Exception(
                f'Received {type(front_end).__name__} instance for front_end argument. Should either be NoneType or '
                f'an instance of {EPCOT.__name__}.'
            )
        self.front_end.as_sequence_embedder()

        if diffuser is None:
            self.diffuser = ChromoGenDiffuser()
        elif isinstance(diffuser,ChromoGenDiffuser):
            self.diffuser = diffuser
        else:
            raise Exception(
                f'Received {type(diffuser).__name__} instance for diffuser argument. Should either be NoneType or '
                f'an instance of {ChromoGenDiffuser.__name__}.'
            )

        # Ensure everything's on the same device 
        self.diffuser.to(dtype=self.front_end.dtype, device=self.front_end.device) 

        # If the data_loader is passed as None, go ahead and ignore. 
        # That's handled by the setter
        self.data_loader = data_loader

        ####
        # Track/don't track gradients,  the model with grad, eval/training mode
        self.requires_grad_(requires_grad)

        # In case this was called from_file() or from_files(), if no grad required, delete any existing
        # gradients that may have been saved in the file to free up some RAM.
        if not requires_grad and _internal:
            self.zero_grad()

        if training_mode:
            self.train()
        else:
            self.eval()

    ##############################################################################################################################
    # Basic properties
    @property
    def dtype(self):
        for p in self.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.parameters():
            return p.device

    @property
    def requires_grad(self):
        return any(p.requires_grad for p in self.parameters())

    @property
    def max_gpus_to_use(self):
        return self.__max_gpus_to_use

    @max_gpus_to_use.setter
    def max_gpus_to_use(self,max_gpus_to_use):
        if max_gpus_to_use is None:
            self.__max_gpus_to_use = None
        else:
            self.__max_gpus_to_use = _validate_int(max_gpus_to_use,'max_gpus_to_use',minimum=0)

    @property
    def maximum_conformations_per_call(self):
        return self.__maximum_conformations_per_call

    @maximum_conformations_per_call.setter
    def maximum_conformations_per_call(self,new_val):
        if new_val is None:
            self.__maximum_conformations_per_call = new_val
        else:
            self.__maximum_conformations_per_call = _validate_int(
                new_val,'maximum_conformations_per_call',minimum=1
            )

    @property
    def available_gpus(self):
        return self.__available_gpus.copy()

    @property
    def num_available_gpus(self):
        if (ag:= self.available_gpus) is None:
            return 0
        return len(ag)

    @property
    def gpus_to_use(self):
        if self.__gpus_to_use is not None:
            return self.__gpus_to_use.copy()
        else:
            return None

    @gpus_to_use.setter
    def gpus_to_use(self,gpus):
        if gpus is None:
            self.__gpus_to_use = None
            return
        elif isinstance(gpus,str) and gpus == 'detect':
            self.__gpus_to_use = self.available_gpus
            return
        if not isinstance(gpus,(list,tuple)):
            gpus = [gpus]

        assert gpus, (
            'The list of GPUs cannot be empty.\nIf you wish to force this instance to use the CPU, '
            'then set `chromogen.gpus_to_use = None` or `chromogen.max_gpus_to_use = 0`.\nAlternatively, '
            'simply setting `chromogen.cpu()` will prevent computations from automatically running on the GPU.'
        )

        gpus1 = []
        for k,gpu in enumerate(gpus):
            try:
                gpu = torch.device(gpu) # returns self if already a torch.device
            except Exception as e:
                raise Exception(
                    'Each listed GPU must either be a torch.device instance '
                    'or something that can initialize a torch.device instance. '
                    f'Input {k} of type {type(gpu).__name__} does not satisfy this requirement.'
                )
                
            if gpu in gpus1:
                warnings.warn(f'Device {gpu} was requested more than once. Removing duplicate.')
                continue
            assert gpu.type == 'cuda', \
            f'GPUs should be CUDA devices. However, input "{gpus[k]}" has type {gpu.type}.'
            try:
                torch.empty(0,device=gpu)
            except:
                warnings.warn(f'CUDA device {gpus[k]} is inaccessible. Skipping.')
                continue
            gpus1.append(gpu)

        self.__gpus_to_use = gpus1

        if self.max_gpus_to_use < (num_permitted_gpus:=len(self.gpus_to_use)):
            warnings.warn(
                f'I have been told to use at most {self.max_gpus_to_use} GPUs, but there appear to be '
                f'{num_permitted_gpus} available to me in chromogen.gpus_to_use. By default, I will use CUDA devices '
                f'{self.gpus_to_use[:self.max_gpus_to_use]} and ignore CUDA devices '
                f'{self.gpus_to_use[self.max_gpus_to_use:]}.''\n\nIf you wish to use all GPUs, set '
                f'`chromogen.max_gpus_to_use = {self.num_permitted_gpus}` (or None)''\n\nIf you wish to use a different '
                'selection of GPUs, set `chromogen.gpus_to_use = [<desired CUDA device 1>,<desired CUDA device 2>,...]`.'
            )

    @property
    def home_cpu(self):
        return self.__home_cpu

    @home_cpu.setter
    def home_cpu(self,cpu):
        try:
            cpu1 = torch.device(cpu)
        except:
            raise Exception(
                'The cpu listed as home_cpu must either be a torch.device instance '
                'or something that can initialize a torch.device instance. '
                f'Input {cpu} of type {type(cpu).__name__} does not satisfy this requirement.'
            )
        assert cpu1.type=='cpu', f'torch.device(home_cpu).type should be "cpu". Received "{cpu1.type}". '
        try:
            torch.empty(0,device=cpu1)
        except:
            raise Exception(
                f'The CPU device indicated by home_cpu = {cpu} seems to be inaccessible.'
            )
        
        if cpu1.index is None:
            if (cpu2:=torch.device('cpu:0')).index is not None:
                warnings.warn(
                    'When multiple CPU devices are available, '
                    'you should specify an index for home_cpu. '
                    'Defaulting to index 0.'
                )
                cpu1 = cpu2
        self.__home_cpu = cpu1

    @property
    def replica_id(self):
        return self.__replica_id

    @property
    def maximum_samples_per_gpu(self):
        return self.__maximum_samples_per_gpu

    @maximum_samples_per_gpu.setter
    def maximum_samples_per_gpu(self,maximum_samples_per_gpu):
        self.__maximum_samples_per_gpu = _validate_int(
            maximum_samples_per_gpu,
            'maximum_samples_per_gpu',
            minimum=1
        )

    @property
    def maximum_regions_embedded_per_GPU(self):
        return self.__maximum_regions_embedded_per_GPU

    @maximum_regions_embedded_per_GPU.setter
    def maximum_regions_embedded_per_GPU(self,new_val):
        self.__maximum_regions_embedded_per_GPU = _validate_int(
            new_val,
            'maximum_regions_embedded_per_GPU',
            minimum=1
        )

    ##############################################################################################################################
    # Some specialized functinonality
    def __set_available_gpus(self):
        if torch.cuda.is_available():
            self.__available_gpus = [
                torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())
            ]
        else:
            # No GPUs available!
            self.__available_gpus = []

    __valid_arg_format = '\n'.join([
        'Positional arguments to the forward and load_input methods MUST be one of the following:',
        '    1. NO positional arguments, meaning "load all regions in this_instance.data_loader.start_indices".',
        '    2. ONE positional argument: Pre-formatted (non-empty) start positions.',
        '        a. dict: Same formatting as this_instance.data_loader.start_indices, i.e.,',
        '            {',
        '                chrom_name_1:[start_position_1, ...],',
        '                chrom_name_2:[start_position_1,...],',
        '                ...',
        '            }',
        '        b. list: Same formatting as this_instance.data_loader.iteration_index, i.e.,',
        '            [',
        '                (chrom_name1,start_position1),',
        '                (chrom_name2,start_position2),',
        '                ...',
        '            ]',
        '    3. TWO OR MORE positional arguments: chrom_name1, start_position1, chrom_name2, start_position2, ...',
        '',
        'In all cases, start positions must be a genomic coordinate in base pairs, and chrom_name may follow any',
        'of the following conventions:',
        '    - 1, 2, 3, 4, ...',
        '    - "1", "2", "3", ...',
        '    - "chr1", "chr2", "chr3", ...'
    ])
    @property
    def arg_format(self):
        return self.__valid_arg_format

    ##############################################################################################################################
    # Some basic functionality
    def cuda(self,device=None):
        if device is None:
            if self.gpus_to_use:
                device = self.gpus_to_use[self.__replica_id]
            elif self.available_gpus:
                err_msg = 'No GPUs are associated with ChromoGen instance!\n'
                err_msg+= f'That said, the following CUDA devices available: {self.available_gpus}''\n\n'
                if self.max_gpus_to_use == 0:
                    err_msg+= 'However, chromogen.max_gpus_to_use == 0, which you, the user, would have set.\n'
                    err_msg+= 'Increase this number to allow this instance to run on a CUDA device.'
                    if not self.gpus_to_use:
                        err_msg+= '\n\nIt seems you also set chromogen.gpus_to_use to None, so you will also have to fix that.\n'
                else:
                    err_msg+= 'It seems chromogen.gpus_to_use was set to None at some point.\n'
                if not self.gpus_to_use:
                    err_msg+= 'Try setting `chromogen.gpus_to_use = "detect"` to provide access to all CUDA devices or '
                    err_msg+= '`chromogen.gpus_to_use = [<chosen CUDA device 1>, <chosen CUDA device 2>, ...]`.\n'
                    err_msg+= 'Alternatively, specify the cuda device when calling cuda, i.e., `chromogen.cuda(device)`.'
                raise Exception(err_msg)
            else:
                raise Exception('No CUDA devices detected.')
        return super().cuda(device)

    ##############################################################################################################################
    # Initialize the EPCOTInputHandler
    
    @property
    def data_loader(self):
        return self.__data_loader

    @data_loader.setter
    def data_loader(self,loader):
        if loader is not None:
            assert isinstance(loader,EPCOTInputLoader), \
            f'Expected data_loader to be an EPCOTInputLoader instance. Received {type(data_loader).__name__}.'
        self.__data_loader = loader

    ###
    # EPCOTInputLoader
    def attach_data(
        self,
        alignment_filepath, # The genome (in .h5 format) filepath
        bigWig_filepath,    # DNase-seq data filepath
        resolution = 1000,  # Resolution for the bins in the first step of EPCOT
        num_bins=None,      # None or positive int. Either this or region_length must be defined.
        region_length=1_280_000, # None or int. NONE or INT
        pad_size=300, # or integer. Default: 300
        start_indices=None, # Mostly for reverse-compatibility
        allow_bin_overlap=False,   # Can the bins defined by resolution overlap? 
        allow_region_overlap=True, # Can the larger regions (resolution * num_bins) overlap? Overrides allow_bin_overlap if False
        batch_size=1,              # How many 'regions' to return at once. 
        shuffle=False,             # Whether to shuffle the indices between epochs
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

        self.data_loader = EPCOTInputLoader(
            alignment_filepath=alignment_filepath,
            bigWig_filepath=bigWig_filepath,
            resolution=resolution,
            num_bins=num_bins,
            region_length=region_length,
            pad_size=pad_size,
            start_indices=start_indices,
            allow_bin_overlap=allow_bin_overlap,
            allow_region_overlap=allow_region_overlap,
            batch_size=batch_size,
            shuffle=shuffle,
            dtype=dtype,
            device=device,
            max_workers=max_workers,
            store_in_memory=store_in_memory,
            bigWig_nan_to=bigWig_nan_to,
            chroms=chroms,
            auto_parse_fasta=auto_parse_fasta,
            interactive=interactive,
            verify_assembly=verify_assembly
        )
    
    ##############################################################################################################################
    # Load from file, etc. 
    @staticmethod
    def from_file(filepath,**init_kwargs):
        assert isinstance(filepath,(str,Path)), f'Expected filepath to be a string or pathlib.Path object. Received {type(filepath).__name__}.'
        data = torch.load(filepath,map_location='cpu')
        assert (
            isinstance(data,dict) and 
            'model' in data and 
            'unet_config' in data and 
            'diffusion_config' in data and 
            'cnn_config' in data and 
            'transformer_config' in data and 
            'epcot_hic_config' in data and 
            'state_dict' in data
        ), 'This doesn\'t appear to be a ChromoGen save file.'
        front_end = EPCOT.from_dict(data)
        diffuser = ChromoGenDiffuser.from_dict(data)
        return ChromoGen(front_end=front_end,diffuser=diffuser,_internal=True,**init_kwargs)

    @staticmethod
    def from_files(epcot_filepath,diffuser_filepath,**init_kwargs):
        # Load model from separate EPCOT and Diffuser filepaths
        front_end = EPCOT.from_file(epcot_filepath)
        diffuser = ChromoGenDiffuser.from_file(diffuser_filepath)
        return ChromoGen(front_end=front_end,diffuser=diffuser,_internal=True,**init_kwargs)
    
    def save(self,filepath):
        f = Path(filepath)
        f.parent.mkdir(exist_ok=True,parents=True)
        f = f.with_suffix('.pt')
        data = {
            'cnn_config':self.front_end.front_end.cnn_config,
            'transformer_config':self.front_end.front_end.transformer_config,
            'epcot_hic_config':self.front_end.config,
            'state_dict':self.front_end.state_dict(),
            'model': self.diffuser.state_dict(),
            'unet_config':self.diffuser.model.config,
            'diffusion_config':self.diffuser.config
        }
        torch.save(data,f)
    
    ##############################################################################################################################
    # SAMPLE (and support for sampling)

    # So that multiple GPUs can be utilized during generation
    def __get_replicas(self,max_needed):

        # Don't broadcast to GPUs unless we're already on one
        if max_needed <= 1 or self.device.type == 'cpu' or self.gpus_to_use is None or self.max_gpus_to_use == 0:
            return [self]

        # Determine the number of replicas actually needed
        n = min(max_needed, len(self.gpus_to_use))
        if self.max_gpus_to_use:
            n = min(n,self.max_gpus_to_use)

        # Get all replicas
        replicas = [self.cuda()] # Automatically sends it to the pre-designated CUDA device. 
                                 # Will only be on another device if set manually by user, but... override them now that we're replicating
        # Because it uses multithreading, running copy.deepcopy() with the data_loader attached will cause errors. 
        # So, detach before copying and replace afterwards. We don't really need to copy that anyway, as the whole point
        # is to avoid sending multiple requests to the same file simultaneously and causing IO issues
        dl = self.data_loader
        self.data_loader = None
        for k in range(1,n):
            replica = copy.deepcopy(self)
            replica.data_loader = dl
            replica.__replica_id = k
            replicas.append(replica.cuda()) 
        self.data_loader = dl

        return replicas

    def get_replicas(self,max_needed):
        return self.__get_replicas(max_needed)

    ######
    # Sample for REAL... placing in private methods so that they can easily be called from an executor. 
    # Separate the diffusion & embedding steps since the embeddings are deterministic and don't need to
    # be recreated when combining conformations from multiple cond_scale/rescaled_phi combinations.
    @torch.inference_mode()
    def _embed(
        self,
        genomic_data
    ):
        
        return self.front_end.embed_sequences(genomic_data.to(dtype=self.front_end.dtype,device=self.front_end.device))

    # Can't use inference mode here because the coordinates may be optimized... 
    def _diffuse(
        self,
        embeddings,
        cond_scale,
        rescaled_phi,
        return_coords,
        correct_distmap
    ):

        if isinstance(embeddings,torch.Tensor):
            embeddings=embeddings.to(dtype=self.diffuser.dtype,device=self.diffuser.device)
        # The Diffuser returns either uncorrected distances (as Distances instance)
        # or optimized coordinates (as Coordinates instance)
        conformations = self.diffuser.sample(
            embeddings,
            cond_scale = cond_scale,
            rescaled_phi = rescaled_phi,
            coordinates=return_coords or correct_distmap
        )

        if correct_distmap and not return_coords:
            # Already retrieved the optimized coordinates, but want
            # the associated distance maps.
            conformations = conformations.distances

        # Place in CPU memory and return
        return conformations.to(self.home_cpu)
        
    def sample(
        self,
        x,                       # This is either a torch.Tensor containing sequencing data or the batch size for unguided generation
        samples_per_region = 1,  # Samples per provided sequence
        return_coords=True,      # Whether to convert the generated distance maps into optimized coordinates prior to returning them
        correct_distmap=False,   # Ignored if return_coords is True. Whether to ensure the physical validity of the distance maps
                                 # prior to returning them
        cond_scales = [1.,5.],   # Conditioning to use. Generally corresponds to to guidance_strength+1
        rescaled_phis = [0.,8.], # Rescaling coefficient to help minimize artifacts introduced by larger guidanance strengths
        proportion_from_each_scale = None, # Relative weightings of each scale/rescaled_phi combination. None splits samples equally. 
        force_eval_mode=True,
        distribute=True,
        silent=False
    ):
        pprint('Preparing for generation',silent=silent)
        ########################################################################
        # Validate/format inputs
        # samples_per_scale is a dict that replaces samples_per_region, cond_scales, rescaled_phis, and proportions_from_each_scale,
        # as `samples_per_scale[(cond_scale, rescaled_phi)] == <int: number of samples to generate with this combination>.`
        # Other values are simply validated rather than reformatted, so they don't need to be returned.

        if isinstance(x,int):
            # If x is an integer, we can very easily construct this dictionary and avoid the more complicated function
            assert x > 0, f'When x is passed as an integer, it must be positive-valued. Received {x}'
            samples_per_scale = {(0.,0.):x}
            x_shape_original = None
        else:
            # Ok, go through and verify everything
            x, samples_per_scale = _validate_sample_parameters(
                x,
                samples_per_region,
                return_coords,
                correct_distmap,
                cond_scales,
                rescaled_phis,
                proportion_from_each_scale,
                force_eval_mode,
                distribute
            )
            x_shape_original = x.shape

            ###
            # Some minor prep of the inputs 
            
            # Place all batch dimensions in the first dim for now. 
            x = x.flatten(0,-4)

            # SHOULD check for repeats in x, but won't implement for now
        
        ########################################################################
        # Determine how to split up the generative steps to satisfy
        # the maximum samples allowed on each GPU

        # Ensure we don't put more samples on a single GPU than requested by the user
        n = self.maximum_samples_per_gpu
        generative_steps = []
        for (cond_scale,rescaled_phi),n_samples in samples_per_scale.items():
            while n_samples:
                if n_samples > n:
                    nn = n_samples//n
                    if n*nn != n_samples:
                        nn+=1
                    this_step = round(n_samples/nn)
                else:
                    this_step = n_samples
                generative_steps.append((cond_scale,rescaled_phi,this_step))
                n_samples-= this_step

        # That said, we CAN combine generation in different regions so long as 
        # they have the same cond_scale, rescaled_phi
        if isinstance(x,torch.Tensor):
            n_regions = x.shape[0]
            generative_steps2 = []
            for cond_scale,rescaled_phi,n_samples in generative_steps:
                max_can_merge = n // n_samples
                i = 0
                while i < n_regions:
                    j = i+max_can_merge
                    generative_steps2.append((cond_scale,rescaled_phi,n_samples,i,j))
                    i = j
        else:
            generative_steps2 = [
                (cond_scale,rescaled_phi,n_samples,None,None) 
                for cond_scale,rescaled_phi,n_samples in generative_steps
            ]
        generative_steps = generative_steps2

        # We have a separate limit on the number of embeddings to generate per GPU
        # At the same time, embeddings are deterministic, so we won't recreate them 
        # for every cond_scale/rescaled_phi combination. 
        # Split them here. 
        sequence_batches = []
        if isinstance(x,torch.Tensor):
            # otherwise, it's unguided
            i = 0
            while i < x.shape[0]:
                j = i + self.maximum_regions_embedded_per_GPU
                sequence_batches.append(x[i:j,...])
                i = j
        
        # Prepare to distribute this process across GPUs
        if distribute:
            max_EPCOT_replicas_helpful = len(sequence_batches)
            max_diffusion_replicas_helpful = len(generative_steps)
        else:
            max_diffusion_replicas_helpful = 1
            max_EPCOT_replicas_helpful = 1
        max_replicas = max(max_diffusion_replicas_helpful,max_EPCOT_replicas_helpful)
        

        # For simplicity, place in eval mode BEFORE duplicating samples
        was_training = self.training
        required_grad = any(p.requires_grad for p in self.parameters())
        if force_eval_mode:
            self.requires_grad_(False)
            self.eval()

        # Do the rest in try-except-finally so that we can ensure the model goes back to 
        # training mode (if it was in training mode before)
        try:

            # Create the replicas
            replicas = self.__get_replicas(max_replicas)
            
            ########################################################################
            # Generate the samples
            pprint('Generating conformations',silent=silent)

            
            ###
            # Get the resource manager. Will distribute jobs across GPUs as they become available.
            #resource_manager = ResourceManager(replicas)
            '''
            ####
            # Start by creating the embeddings. They're deterministic, so... don't
            # recreate them for each guidance strength.
            if has_embeddings := (len(sequence_batches) > 0):
                for batch in sequence_batches:
                    resource_manager.submit('_embed',batch)
                embeddings = torch.cat(resource_manager.results(verbose=not silent, desc='Embedding tasks completed'))
                emb_shape = embeddings.shape[1:]
                resource_manager.clear()

            ###
            # Generate all the distance maps/coordinates
            for cond_scale,rescaled_phi,n_samples,i,j in generative_steps:
                if has_embeddings:
                    # Keep the different embeddings organized, but this object MUST have three dimensions
                    # when passed to the diffuser
                    emb = embeddings[i:j,...].unsqueeze(1).expand(-1,n_samples,-1,-1).flatten(0,1)
                else:
                    emb = n_samples
                resource_manager.submit('_diffuse',emb,cond_scale,rescaled_phi,return_coords,correct_distmap)
            """
            return resource_manager
            """
            conformations = resource_manager.results(verbose=not silent, desc='Diffusion tasks completed')
            # Undo the effect of .flatten(0,1) above
            # Have to be janky with it because reshape isn't a method in the ConformationsABC subclasses... yet...
            for c in conformations:
                c._values = c.values.reshape(-1,n_samples,*c.shape[-2:])
            '''
            ################################
            #'''
            # TEMPORARY, FOR DEBUGGING
            if has_embeddings := (len(sequence_batches) > 0):
                embeddings = []
                for batch in sequence_batches:
                    #resource_manager.submit('_embed',batch)
                    embeddings.append(self._embed(batch))
                #embeddings = torch.cat(resource_manager.results(verbose=not silent, desc='Embedding tasks completed'))
                embeddings = torch.cat(embeddings)
                emb_shape = embeddings.shape[1:]
                #resource_manager.clear()
                
            ###
            # Generate all the distance maps/coordinates
            conformations = []
            for cond_scale,rescaled_phi,n_samples,i,j in generative_steps:
                if has_embeddings:
                    # Keep the different embeddings organized, but this object MUST have three dimensions
                    # when passed to the diffuser
                    emb = embeddings[i:j,...].unsqueeze(1).expand(-1,n_samples,-1,-1).flatten(0,1)
                else:
                    emb = n_samples
                
                conformations.append(self._diffuse(emb,cond_scale,rescaled_phi,return_coords,correct_distmap))
                if has_embeddings:
                    # Return embedding index to the first dimension.
                    # Reshape doesn't work on the ConformationsABC classes... yet...
                    c = conformations[-1]
                    c._values = c.values.reshape(-1,n_samples,*c.shape[-2:])
                #resource_manager.submit('_diffuse',emb,cond_scale,rescaled_phi,return_coords,correct_distmap)
            #conformations = resource_manager.results(verbose=not silent, desc='Diffusion tasks completed')
            #'''
            ################################
            ###
            # Organize output
            pprint('Organizing output',silent=silent)
            conformations = _organize_samples(conformations, generative_steps, x_shape_original)

            # DONE!
            return conformations
        except:
            raise
        finally:

            # Delete the resource manager if it exists. Otherwise, 
            # the next step won't be able to ACTUALLY delete the replicas
            if 'resource_manager' in locals():
                del resource_manager
            
            # Delete all replicas and clear CUDA cache
            if 'replicas' in locals():
                while len(replicas) > 1:
                    replica = replicas.pop(-1)
                    replica.cpu()
                    del replica
            

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                    
            # Return to training mode if that's what this was in before
            if was_training:
                self.train()
            if required_grad:
                self.requires_grad_(True)

    
    def load_input(self,*args,_internal=False,silent=False,**kwargs):
        dl = self.data_loader
        if dl is None and not silent:
            raise Exception(
                'Cannot load input data because no data is attached to this instance. '
                'You can attach data with\n`this_instance.attach_data(alignment_filepath,bigWig_filepath,**kwargs)`, '
                'where alignment_filepath is a preprocessed HDF5 file, or\n'
                '`this_instance.attach_data(alignment_filepath,bigWig_filepath,auto_parse_fasta=True,**kwargs)` if '
                'you only have a FASTA file.\nIn both cases, the bigWig file should contain DNase-seq '
                'data for the cell type of interest.'
            )

        if (samples_per_region:= kwargs.get('samples_per_region')) is None:
            if _internal and not silent:
                warnings.warn(
                    'It is HIGHLY recommended that you define samples_per_region when calling ChromoGen '
                    'with chromosome positions rather than torch.Tensor input.'
                )
            samples_per_region = 1
        
        # Saves some code later
        # Should be (chrom_name, start_position)
        if len(args) == 2:
            args = tuple(args)
        
        if not args:
            start_indices = dl.iteration_index
            n_regions = len(iteration_index)
            if not silent and (not dl.start_indices_are_custom() or n_regions > 100):
                warnings.warn(
                    'No chromosome/genomic index input arguments passed. Assuming you want to load ALL '
                    'conformations listed in `this_instance.data_loader.start_indices`. '
                    'If you didn\'t set the start indices, this is likely to generate conformations '
                    f'for all regions obtained by iteratively sliding {dl.resolution} '
                    f'iteratively sliding {dl.resolution} base pairs across the whole genome. '
                    'You should kill this process if that wasn\'t intended.'
                )
        elif len(args) == 1:
            match (start_indices:= args[0]):
                case dict():
                    assert start_indices, self.arg_format
                    si = []
                    for key,value in start_indices:
                        assert (
                            isinstance(key,(int,str)) and
                            isinstance(value,list)    and
                            value
                        ), self.arg_format
                        si.extend([
                            (key,v) for v in value
                        ])
                    start_indices = si
                case tuple():
                    assert len(tuple) == 2, self.arg_format
                    start_indices = [start_indices]
                case list():
                    pass
                case _:
                    raise Exception(self.arg_format)
        else:
            assert len(args)%2 == 0, self.arg_format
            start_indices = [
                (args[2*k],args[2*k+1]) for k in range(len(args)//2)
            ]

        no_duplicates = True
        si = []
        for chrom_name in (chroms:={chrom_name for chrom_name,_ in start_indices}):
            dl.__contains__(chrom_name,_validate_format=True,_assert=True)
            l = dl.chroms(chrom_name)
            for start in [start for chrom,start in start_indices if chrom==chrom_name]:
                start = _validate_int(start, 'start indices',minimum=0)
                assert start <= l, f'Start index {start:,} is out of bounds in chromosome {chrom_name}.'
                cns = (chrom_name,start)
                if cns in si:
                    if no_duplicates and not silent:
                        warnings.warn('Duplicate entries found. Removing.')
                    no_duplicates = False
                else:
                    si.append(cns)

        if _internal:
            total_samples = samples_per_region * len(start_indices)
            m = self.maximum_conformations_per_call
            if m < total_samples:
                raise Exception(
                    f'This ChromoGen instance was set to generate a maximum of {m:,} samples '
                    'in a single call. However, your selected start indices and samples_per_region '
                    f'settings would result in {total_samples:,} samples being generated.'
                )

        pprint('Loading sequence data from file.',silent=silent)
        sequence_data = []
        for chrom_name, start_idx in tqdm(start_indices,desc='Regions loaded',disable=not silent):
            sequence_data.append(dl.fetch(chrom_name,start_idx))
        
        return torch.stack(sequence_data,dim=0)
    
    def forward(self,*args,silent=False,**kwargs):
        if 'x' in kwargs:
            assert not args, (
                'It seems you passed x as a keyword argument. This is fine, but now I don\'t know '
                'what to do with the positional arguments, which should either be x or inputs '
                'to this_instance.load_input() so that we could load inputs from the sequence files. '
                'All other arguments you intend to have passed to sample should be inserted as keyword '
                'arguments.'
            )
            x = kwargs['x']
        
        elif len(args) == 1 and isinstance(args[0],(int,torch.Tensor)):
            x = args[0]
            if isinstance(x,int):
                pprint(f'Integer passed. Will generate {x} unguided conformations.',silent=silent)
        else:
            pprint('Attempting to load input from file.',silent=silent)
            x = self.load_input(*args,_internal=True,silent=silent,**kwargs)

        return self.sample(x,silent=silent,**kwargs)
        




