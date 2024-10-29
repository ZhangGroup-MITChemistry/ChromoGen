import torch
import os
import sys
sys.path.insert(0,'../diffusion_compartmental/model_files')
from Unet_no_embed_reduction import Unet
from GaussianDiffusion import GaussianDiffusion

sys.path.insert(1,'./')
from support_functions import *

sys.path.insert(1,'../data_utils/SampleClass/')
from Distances import Distances, Normalizer
from OrigamiTransform import OrigamiTransform

def load_model(
    *,
    model_dir = '../../data/models/diffusion_origami_64_no_embed_reduction/',
    # U-Net options
    dim=64,
    cond_drop_prob = 0.,
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
    embedding_dimensions=(1,256,256),
    # Diffusion model options
    image_size=32,
    timesteps = 1000,
    sampling_timesteps = None,
    objective = 'pred_noise',
    beta_schedule = 'cosine',
    ddim_sampling_eta = 1.,
    offset_noise_strength = 0.,
    min_snr_loss_weight = False,
    min_snr_gamma = 5,
    milestone=None
):

    ##########################
    # Initialize the unet
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
        embedding_dimensions=embedding_dimensions
    )

    ##########################
    # Initialize the diffusion model
    diffusion = GaussianDiffusion(
        unet,
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
    
    ##########################
    # Load the model parameters

    milestone = most_recent_milestone(model_dir) if milestone is None else milestone
    model_filepath = get_model_filepath(model_dir,milestone)#most_recent_milestone(model_dir))
    diffusion.load_state_dict(torch.load(model_filepath,map_location=diffusion.device)['model'])

    ##########################
    # Set the model in evaluation mode & reduce overhead by not requiring gradients
    diffusion.eval()
    diffusion.requires_grad_(False)

    ##########################
    # Move to the GPU, if available
    if torch.cuda.is_available():
        diffusion = diffusion.to('cuda')

    return diffusion
    
    

class Sampler:
    
    def __init__(
        self,
        model_dir,
        *,
        save_file=None,
        samples_device=torch.device('cpu'), # Can also be None
        cond_scale=1.5,
        rescaled_phi=.5,
        normalizer=None,
        mean_dist_fp='../../data/mean_dists.pt',
        mean_sq_dist_fp = '../../data/squares.pt',
        **kwargs # for the load function
    ):

        # Loading the model is slow, so start by preparing the other objects that may well fail.
        if normalizer is None:
            self.normalizer = Normalizer(
                mean_dist_fp=mean_dist_fp,
                mean_sq_dist_fp=mean_sq_dist_fp
            )
        elif type(normalizer) != Normalizer:
            raise Exception(f'normalizer input should be of type Normalizer, but receieved {type(normalizer)}')
        else:
            self.normalizer = normalizer

        # Load model
        self.model = load_model(model_dir=model_dir,**kwargs)
        self.__samples = None
        self.__data_is_saved = True
        self.__save_file = None
        self.samples_device = samples_device
        self.cond_scale = cond_scale
        self.rescaled_phi = rescaled_phi
        self.save_location = save_file

        # Prepare the relevant origami transforms
        model = self.model
        if model.model.channels == 1:
            self.origami_transform = OrigamiTransform(
                num_reduction_steps=0
            )
            self.image_size = model.image_size
        elif model.model.channels == 2:
            self.origami_transform = OrigamiTransform(
                num_reduction_steps=1,
                drop_lower_triangle=True,
                preserve_diagonal=False
            )
            self.image_size = 2 * model.image_size
        elif model.model.channels == 8:
            self.origami_transform = OrigamiTransform(
                num_reduction_steps=2,
                drop_lower_triangle=True,
                preserve_diagonal=False
            )
            self.image_size = 4 * model.image_size - 2
            
        

    def __len__(self):
        if self.samples is None:
            return 0
        return self.samples.shape[0]
    
    @property
    def device(self):
        return self.model.device

    @property
    def samples(self):
        return self.__samples
    
    @property
    def save_location(self):
        return self.__save_file

    @save_location.setter
    def save_location(self,f):
        # Save the currently-held data, if applicable
        if self.__save_file is not None and self.samples is not None and not self.__data_is_saved:
            self.dump()

        self.purge()
        self.__save_file = f
        if f is not None:
            self.load_sample(f)

    def load_sample(self,f):
        if not os.path.exists(f):
            return

        map_location = self.device if self.samples_device is None else self.samples_device
        self.__samples = torch.load(f,map_location=map_location)

    def to(self,*args,**kwargs):
        self.model.to(*args,**kwargs)
        if self.samples is not None and self.samples_device is None:
            self.__samples = self.__samples.to(*args,**kwargs)
        return self
         
    def dump(self,save_file=None):
        sl = self.save_location if save_file is None else save_file
        assert sl is not None, 'No filepath specified!'

        s = self.samples
        assert s is not None, 'The Sampler contains 0 samples!'

        # This miiiiight cause issues
        if save_file is not None and os.path.exists(save_file) and save_file != self.save_location:
            s = torch.cat(
                [
                    torch.load(save_file,map_location=s.device).to(s.dtype)
                ]
                ,
                dim=0
            )
            
        
        torch.save(s.cpu(),sl)
        self.__data_is_saved = True

    def purge(self):
        self.__samples = None
        self.__save_file = None
        self.__data_is_saved = True

    def __append(self,samples):
        s = self.samples
        if s is None:
            self.__samples = samples
        elif type(samples) == torch.Tensor:
            self.__samples = torch.cat([s,samples.to(device=s.device,dtype=s.dtype)],dim=0)
        else:
            self.__samples = torch.cat([s,*[sam.to(device=s.device,dtype=s.dtype) for sam in sample]],dim=0)
            
    
    def sample(
        self,
        embedding,
        *,
        cond_scale=None,
        rescaled_phi=None,
        num_samples=1000,
        batch_size=3000,
        save_results=True,
        save_every_step=True,
        save_location=None, # Resets the filepath! 
        coordinates = True # Otherwise, return uncorrected distance maps
    ):
        assert embedding.ndim > 2

        cond_scale = self.cond_scale if cond_scale is None else cond_scale
        rescaled_phi = self.rescaled_phi if rescaled_phi is None else rescaled_phi

        if save_location is not None:
            self.save_location = save_location
        elif save_results:
            assert self.save_location is not None, 'You\'ve requested to save the data during sampling, '+\
            'but no save_location specified is specified in the Sampler object!'

        # Number of samples to generate
        N = num_samples - len(self)
        if N <= 0:
            return self

        # just doing this for now... while all embeddings are size 1xNxN
        # So, if accidentally batched in 0th dimension of 3-dimensional tensor, 
        # this will place the batch in the correct location
        if embedding.ndim == 3:
            embedding = embedding.unsqueeze(1)
            
        # Place the embeddings on the cpu; with repeat, the object can become quite large, 
        # especially if multiple embeddings are provided
        #embedding = embedding.cpu().repeat_interleave(num_samples,0)

        # Simpler approach for now
        assert embedding.shape[0] == 1, 'Currently only support generation on one embedding, '+\
        f'but received {embedding.shape[0]} embeddings stacked together!'
        
        embedding = embedding.to(self.device)

        # Sample (in batches, if necessary)
        unfold = lambda input: self.origami_transform.inverse(input,final_imsize=self.image_size)
        fold = OrigamiTransform() # Most efficient way to store
        normalizer = self.normalizer
        model = self.model
        
        i = 0
        samples = []
        while i < N:
            n = min(N-i,batch_size)

            with torch.inference_mode():
                samples.append(
                    Distances(
                        unfold(
                            model.sample(
                                embedding.repeat(n,1,1,1),
                                cond_scale=cond_scale,
                                rescaled_phi=rescaled_phi
                            )
                        )
                    ).unnormalize_(
                        normalizer
                    )
                )
            if coordinates:
                samples[-1] = samples[-1].coordinates.values.squeeze()
            else:
                samples[-1] = fold(samples[-1].values)
            '''
            if coordinates:
                samples.append(
                    Distances(
                        unfold(
                            model.sample(
                                embedding.repeat(n,1,1,1),
                                cond_scale=cond_scale,
                                rescaled_phi=rescaled_phi
                            )
                        ).unnormalize_(
                            normalizer
                        ).coordinates.values
                    )
                )
            else:
                samples.append(
                    Distances(
                        unfold(
                            model.sample(
                                embedding.repeat(n,1,1,1),
                                cond_scale=cond_scale,
                                rescaled_phi=rescaled_phi
                            )
                        ).unnormalize_(
                            normalizer
                        ).coordinates.values
                    )
                )
            '''
            '''
            samples.append(
                model.sample(
                    embedding.repeat(n,1,1,1),
                    cond_scale=cond_scale,
                    rescaled_phi=rescaled_phi
                )
            )
            '''
            
            if save_results and save_every_step:
                self.__append(samples.pop())
                self.dump()
            i+= n

        if len(samples) > 0:
            self.__append(samples)
            if save_results:
                # In case save_every_step was False
                self.dump()
            else:
                self.__data_is_saved = False

        return self
        

        
