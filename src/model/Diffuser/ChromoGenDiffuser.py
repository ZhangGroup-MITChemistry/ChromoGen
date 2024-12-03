'''
Greg Schuette 2024
'''
import torch
from pathlib import Path
from .GaussianDiffusion import GaussianDiffusion
from .Unet import Unet
from .Trainer import Trainer
from .training_utils.DataLoader import DataLoader
from .training_utils.EmbeddedRegions import EmbeddedRegions
from .training_utils.ConfigDataset import ConfigDataset
from ...Conformations._Distances import Distances
from collections import OrderedDict

def pprint(message,flush=True,silent=False):
    if silent:
        return
    print(message,flush=flush)

###########################################
# Sample-generating diffusion model
class ChromoGenDiffuser(GaussianDiffusion):

    def __init__(
        self,
        filepath=None,
        silent=False,
        *,
        dim=64,
        cond_drop_prob = 0.5,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 2,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        attn_dim_head = 32,
        attn_heads = 4,
        embedding_dimensions=(1, 256, 256),
        null_parameter=None,
        # GaussianDiffusion parameters
        image_size=32,
        timesteps = 1000,
        sampling_timesteps = None,
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        ddim_sampling_eta = 1.,
        offset_noise_strength = 0.,
        min_snr_loss_weight = False,
        min_snr_gamma = 5,
        # Data parameters
        dtype=torch.float,
        device='cpu',
        model=None
    ):

        self.silent=silent
        
        # U-Net parameters
        if model is None:
            unet = Unet(
                dim=dim,
                cond_drop_prob = cond_drop_prob,
                init_dim = init_dim,
                out_dim = out_dim,
                dim_mults=dim_mults,
                channels = channels,
                resnet_block_groups = resnet_block_groups,
                learned_variance = learned_variance,
                learned_sinusoidal_cond = learned_sinusoidal_cond,
                random_fourier_features = random_fourier_features,
                learned_sinusoidal_dim = learned_sinusoidal_dim,
                attn_dim_head = attn_dim_head,
                attn_heads = attn_heads,
                embedding_dimensions=embedding_dimensions,
                null_parameter=null_parameter
            )
        else:
            unet = model

        
        # GaussianDiffusion parameters
        super().__init__(
            model=unet,
            image_size=image_size,
            timesteps = timesteps,
            sampling_timesteps = sampling_timesteps,
            objective = objective,
            beta_schedule = beta_schedule,
            ddim_sampling_eta = ddim_sampling_eta,
            offset_noise_strength = offset_noise_strength,
            min_snr_loss_weight = min_snr_loss_weight,
            min_snr_gamma = min_snr_gamma
        )

        if filepath is not None:
            self.load(filepath)

    @property
    def dtype(self):
        for p in self.parameters():
            return p.dtype

    @property
    def device(self):
        for p in self.parameters():
            return p.device
    
    def load(self,filepath_or_ordered_dict):
        if isinstance(filepath_or_ordered_dict,(str,Path)):
            filepath_or_ordered_dict = torch.load(filepath_or_ordered_dict,map_location=self.device)
            if isinstance(filepath_or_ordered_dict,dict):
                assert 'model' in filepath_or_ordered_dict, 'The file appears to be improperly formatted for a ChromoGenDiffuser.'
                filepath_or_ordered_dict = filepath_or_ordered_dict['model']
        assert isinstance(filepath_or_ordered_dict,OrderedDict), f'Expected path to file or OrderedDict. Received {type(OrderedDict).__name__}.'
        self.load_state_dict(filepath_or_ordered_dict)

    @staticmethod
    def from_dict(data):
        if isinstance(data,OrderedDict):
            # Use default values. Likely from before configurations saved. 
            chromogen_diffuser = ChromoGenDiffuser()
        else:
            assert isinstance(data,dict), f'Expected data to be a dict or OrderedDict. Received {type(data).__name__}.'
            assert 'model' in data, 'Expected data dictionary to contain "model" key.'
            assert isinstance(data['model'],OrderedDict), (
                'The "model" key in the passed dictionary should contain an OrderedDict. '
                f'It is actually a(n) {type(data["model"]).__name__}.'
            )
            if 'unet_config' in data or 'diffusion_config' in data:
                # Loading from file that was created by the refined class
                not_present = ''
                not_present+= '' if 'unet_config' in data else 'unet_config'
                not_present+= '' if 'diffusion_config' in data else 'diffusion_config'
                if not_present:
                    present = 'unet_config' if not_present == 'diffusion_config' else 'diffusion_config'
                    raise Exception(
                        'Regardless, a passed dictionary should have a "model" key containing the OrderedDict '
                        'with the diffuser\'s model parameters (which appears to be satisfied). '
                        'Otherwise, expected the dictionary to contain either BOTH "unet_config" and '
                        '"diffusion_config" keys (corresponding to files created by this class) or '
                        'NEITHER of these keys (corresponding to files created by legacy code). However, the '
                        f'provided dict contains "{present}" but not "{not_present}".'
                    )
                
                chromogen_diffuser = ChromoGenDiffuser(
                    **data['unet_config'],
                    **data['diffusion_config']
                )
                data = data['model']
            else:
                return ChromoGenDiffuser.from_dict(data['model'])
        chromogen_diffuser.load_state_dict(data)
        return chromogen_diffuser
    
    @staticmethod
    def from_file(filepath,dtype=torch.float,device='cpu'):
        data = torch.load(filepath, map_location='cpu')
        return ChromoGenDiffuser.from_dict(data)

    def get_dataloader(self,embeddings_directory,configuration_filepath,**kwargs):
        return ChromoGenDiffuserDataLoader(
            embeddings_directory=embeddings_directory,
            configuration_filepath=configuration_filepath,
            **kwargs
        )

    def sample(self, embeddings, cond_scale = 6., rescaled_phi = 0.7, coordinates=True):
        samples = Distances(super().sample(embeddings, cond_scale = cond_scale, rescaled_phi = rescaled_phi)) # Runs with torch.no_grad
        samples.unfold_().unnormalize_()
        # If coordinates are desired, return coordinates. Otherwise, return the raw (uncorrected) distance maps
        if coordinates:
            return samples.coordinates # Runs with gradients active
        return samples
    
    def get_trainer(
        self,
        data_loader=None,
        save_folder=None,
        results_folder=None,
        *,
        embeddings_directory=None,
        configuration_filepath=None,
        **kwargs
    ):
        # Silent should really only need to be passed when THIS object is initialized, 
        # but consider the case in which it was specified separately. 
        if 'silent' not in kwargs:
            kwargs['silent'] = self.silent

        # Get the results folder
        assert (results_folder is None) ^ (save_folder is None), (
            'One of save_folder or results_folder MUST be defined. However, they are '
            'synonyms, so ONLY ONE of them may be defined. '
            f'Received {type(results_folder)} and {type(save_folder)} for results_folder and save_folder, respectively.'
        )
        results_folder = results_folder if save_folder is None else save_folder
        assert isinstance(results_folder,(str,Path)), (
            'results_folder (synonym: save_folder) must be a string or pathlib.Path instance. '
            f'Received {type(results_folder)}'
        )
        
        # Handle the DataLoader if need be. 
        if data_loader is None:
            assert isinstance(embeddings_directory,(str,Path)) and isinstance(configuration_filepath,(str,Path)), (
                'If no DataLoader instance is passed via the data_loader argument, then both '
                'embeddings_directory and configuration_filepath must be passed as either pathlib.Path '
                'or str instances, indicating the location of each relevant training data. However, '
                f'embeddings_directory and configuration_filepath are {type(embeddings_directory)} '
                f'and {type(configuration_filepath)} instances, respectively.'
            )
            data_loader = self.get_dataloader(
                embeddings_directory=embeddings_directory,
                configuration_filepath=configuration_filepath,
                **kwargs
            )

        # Initialize the Trainer. 
        return ChromoGenDiffuserTrainer(
            chromogen_diffuser=self,
            data_loader=data_loader,
            results_folder=str(results_folder),
            **kwargs
        )

###########################################
# Dataloader
class ChromoGenDiffuserDataLoader(DataLoader):

    def __init__(
        self,
        embeddings_directory,
        configuration_filepath,
        silent=False,
        *,
        # EmbeddedRegions AND ConfigDataset arguments 
        chromosomes=[str(k) for k in range(1,23)],
        # ConfigDataset arguments
        segment_length=64,
        remove_diagonal=False,
        normalize_distances=True,
        geos=None,
        organisms=None,
        cell_types=None,
        cell_numbers=None,
        replicates=None,
        allow_overlap=True,
        two_channels=False,
        try_GPU=True,
        mean_dist_fp=str(Path(__file__).parent/'support_data/mean_distance_by_separation.pt'),
        mean_sq_dist_fp=str(Path(__file__).parent/'support_data/mean_square_distance_by_separation.pt'),
        # DataLoader arguments
        drop_unmatched_pairs=True, 
        shuffle = True, 
        batch_size=128,
        n_batches_to_queue = 25,
        max_workers = 4, 
        **kwargs
    ):
        
        pprint('Loading Configuration Dataset',silent=silent)
        config_ds = ConfigDataset(
            configuration_filepath,
            segment_length=segment_length,
            remove_diagonal=remove_diagonal,
            batch_size=0, # Not used as the dataloader, so just ignore this or it'll unnecessarily take up extra memory. 
            normalize_distances=normalize_distances,
            geos=geos,
            organisms=organisms,
            cell_types=cell_types,
            cell_numbers=cell_numbers,
            chroms=chromosomes,
            replicates=replicates,
            shuffle=False, # Irrelevant since not used as the dataloader
            allow_overlap=allow_overlap,
            two_channels=two_channels,
            try_GPU=try_GPU,
            mean_dist_fp=mean_dist_fp,
            mean_sq_dist_fp=mean_sq_dist_fp
        )
        
        pprint('Loading Embeddings',silent=silent)
        er = EmbeddedRegions(
            directory=embeddings_directory,
            chroms=chromosomes
        )
        
        pprint('Constructing DataLoader',silent=silent)
        super().__init__(
            config_ds,
            er,
            drop_unmatched_pairs=drop_unmatched_pairs, 
            shuffle = shuffle,
            batch_size=batch_size,
            n_batches_to_queue = 25,
            max_workers = 4
        )

        pprint('DataLoader fully initialized',silent=silent)

# Sample-generating diffusion model's Trainer
class ChromoGenDiffuserTrainer(Trainer):
    
    def __init__(
        self,
        chromogen_diffuser,
        data_loader, 
        results_folder,
        *,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 600_000,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 5000,
        save_folder = None,
        amp = False, # using false here turns off mixed precision, making next value irrelevant 
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.,
        save_latest_only = False,
        load_and_save_in_parallel = False,
        **kwargs
    ):
        
        super().__init__(
            chromogen_diffuser,
            data_loader,
            gradient_accumulate_every = gradient_accumulate_every,
            train_lr = train_lr,
            train_num_steps = train_num_steps,
            adam_betas = adam_betas,
            save_and_sample_every = save_and_sample_every,
            results_folder = results_folder,
            amp = amp, 
            mixed_precision_type = mixed_precision_type,
            split_batches = split_batches,
            max_grad_norm = max_grad_norm,
            save_latest_only = save_latest_only,
            load_and_save_in_parallel=load_and_save_in_parallel
        )


        

        
        