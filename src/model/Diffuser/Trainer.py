'''
Adapted from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
Greg Schuette 2023-2024
'''
import math
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
import warnings
import torch
from torch.optim import Adam
from tqdm.auto import tqdm
from accelerate import Accelerator
import sys
sys.path.insert(0,'./')
from .helper_functions import *
sys.path.insert(1,'../data_utils')
from ...Conformations._OrigamiTransform import OrigamiTransform
origami_transform = OrigamiTransform()

###
def divisible_by(numer, denom):
    return (numer % denom) == 0

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataloader, 
        *,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.,
        save_latest_only = True,
        load_and_save_in_parallel = False
    ):
        super().__init__()

        # Whether to save/load data in parallel. Helpful/saves time if on a distributed system where these files are likely on different drives. 
        self.load_and_save_in_parallel = load_and_save_in_parallel

        # accelerator, to help distribute the model across resources/GPUs (didn't really do anything in the ChromoGen work due to our cluster's setup)
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # sampling and training hyperparameters
        self.save_and_sample_every = save_and_sample_every

        self.gradient_accumulate_every = gradient_accumulate_every
        assert (dataloader.batch_size * gradient_accumulate_every) >= 16, (
            'Your effective batch size (dataloader.batch_size x gradient_accumulate_every) should be at least 16, but the provided parameters '
            f'yield ({dataloader.batch_size}x{gradient_accumulate_every}) = {dataloader.batch_size*gradient_accumulate_every}.'
        )

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset/dataloader (all under one hood in the ChromoGen repo)
        self.dl = dataloader

        # optimizer
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # Save directory
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True, parents=True)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt, self.dl = self.accelerator.prepare(self.model, self.opt, self.dl)

        # Whether to preserve all versions of the model (each milestone) or to overwrite them. 
        self.save_latest_only = save_latest_only

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone,parallel_save=None):
        # If files are on different drives, it can save time to save the model and trainer in parallel. 
        parallel_save = self.load_and_save_in_parallel if parallel_save is None else parallel_save
        
        if not self.accelerator.is_local_main_process:
            return

        model_data = {
            'model': self.accelerator.get_state_dict(self.model),
            'unet_config':self.model.model.config,
            'diffusion_config':self.model.config
        }
        trainer_data = {
            'step': self.step,
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }
        with ThreadPoolExecutor(max_workers=int(parallel_save) + 1) as executor:
            # Submit the save jobs. 
            future1 = executor.submit(torch.save,model_data, str(self.results_folder / f'model-{milestone}.pt'))
            future2 = executor.submit(torch.save,trainer_data, str(self.results_folder / f'trainer-{milestone}.pt'))

            # Raise exception if one appeared in either of the processes
            future1.result()
            future2.result()
            
        # If desired, delete the prior version of the model now that the new version has successfully saved
        if self.save_latest_only and milestone > 1:
            if (prior_save:= Path(self.results_folder  / f'model-{milestone-1}.pt')).exists():
                prior_save.unlink()
            if (prior_save:= Path(self.results_folder  / f'trainer-{milestone-1}.pt')).exists():
                prior_save.unlink()
    
    def load(self, milestone=-1, parallel_load=None):
        # If files are on different drives, it can save time to load the model and trainer files in parallel. 
        parallel_load = self.load_and_save_in_parallel if parallel_load is None else parallel_load
        
        accelerator = self.accelerator
        device = accelerator.device

        if milestone == -1:
            milestones = {int(fn[6:-3]) for fn1 in self.results_folder.glob('*.pt') if (fn:=fn1.name)[:6] == 'model-'}
            milestones = sorted(milestones.intersection({
                int(fn[8:-3]) for fn1 in self.results_folder.glob('*.pt') if (fn:=fn1.name)[:8] == 'trainer-'
            }))
            if milestones:
                milestone = milestones[-1]
            else:
                warnings.warn(f'No model/trainer file pairs located in save directory {self.results_folder}. Skipping load.')
                return

        f = self.results_folder / f'model-{milestone}.pt'
        f1 = self.results_folder / f'trainer-{milestone}.pt'
        if not f.is_file():
            if not f1.is_file():
                warnings.warn(f'Specified milestone, {milestone}, has neither model file {f.name} nor trainer file {f1.name} inside save directory {f.parent}. Skipping load.')
            else:
                warnings.warn(f'Specified milestone, {milestone}, does not have associated model file {f.name} inside save directory {f.parent}. Skipping load.')
            return
        if not f1.is_file():
            warnings.warn(f'Specified milestone, {milestone}, does not have associated trainer file {f1.name} inside save directory {f.parent}. Skipping load.')
            return
        
        print(f'Loading model and trainer files with milestone {milestone} from save directory {f.parent}.',flush=True)
        with ThreadPoolExecutor(max_workers=int(parallel_load)+1) as executor:

            # Load model data. This is an in-place operation, so we can largely ignore the future object.
            # However, we'll track it so that we can raise the relevant exception if something goes wrong. 
            future1 = executor.submit(self.model.load, str(f))

            # Load trainer data 
            future2 = executor.submit(torch.load, str(f1), map_location=device)

            # Fetch the fully loaded trainer data once the future is complete. 
            data = future2.result()

            # Place trainer data into the relevant objects
            self.step = data['step']
            self.opt.load_state_dict(data['opt'])
            if exists(self.accelerator.scaler) and exists(data.get('scaler')):
                self.accelerator.scaler.load_state_dict(data['scaler'])

            # If a version is specified (not relevant in version 1), then tell the user which version was loaded. 
            if 'version' in data:
                print(f"loaded from version {data['version']}",flush=True)

            # Raises exception if something went wrong in that process. 
            future1.result()
                
        print('Model and trainer successfully loaded',flush=True)
        

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process, leave=None) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    # Get next batch
                    dist_maps, embeddings = next(self.dl) 

                    # Perform the origami transform 
                    dist_maps = origami_transform(dist_maps)

                    with self.accelerator.autocast():
                        loss = self.model(dist_maps,embeddings)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        with torch.inference_mode():
                            milestone = self.step // self.save_and_sample_every
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
