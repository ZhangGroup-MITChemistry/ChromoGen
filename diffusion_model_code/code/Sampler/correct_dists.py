#!/state/partition1/llgrid/pkg/anaconda/python-LLM-2023b/bin/python

#SBATCH --job-name=corect_dists
##SBATCH --partition=debug-gpu
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:volta:1  
#SBATCH --cpus-per-task=20  
#SBATCH --array=0-1
#SBATCH --output=./log_files/corect_dists.log

import torch
import os
import sys
sys.path.insert(0,'../data_utils/SampleClass/')
from OrigamiTransform import OrigamiTransform
origami_transform = OrigamiTransform()
from Distances import Distances, Normalizer

normalizer = Normalizer()


'''
NVM I didn't end up using this; the GPU can handle this, but another process was taking like 15 GB
of memory for some reason. However, I'm leaving it because batching may be a nice feature to add...
Due to memory issues, I separately ran the following for the 100,000 unguided conformations
dists = Distances(
    origami_transform.inverse(
        torch.load('../../data/samples/origami_64_no_embed_reduction/eval_mode/unguided.pt'),
        final_imsize=2*dists.shape[-1]
    )
).unnormalize_(normalizer=normalizer)
coords = dists[:50_000].coordinates
coords = coords.append(dists[50_000:].coordinates)
torch.save(coords.values.squeeze(),save_dir+f)
'''

directories = [
    '../../data/samples/origami_64_no_embed_reduction/cross_validation/CTCF/',
    '../../data/samples/origami_64_no_embed_reduction/cross_validation/IMR/',
    '../../data/samples/origami_64_no_embed_reduction/eval_mode/'
]

print('To loop')
i = 0
array = int(os.environ['SLURM_ARRAY_TASK_ID'])
for d in directories: 
    files = [f for f in os.listdir(d) if f[-3:] == '.pt']
    save_dir = d + 'corrected/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for f in files: 

        i+= 1
        if i%2 != array:
            continue
        # Avoid recomputing coordinates for already-processed files
        if os.path.exists(save_dir + f):
            continue

        # Load the generated distance maps, unfold them, and place them in a Distances object
        dists = torch.load(d+f)
        print('to computation')
        dists = Distances(
            origami_transform.inverse(
                dists,
                final_imsize=2*dists.shape[-1]
            )
        ).unnormalize_(normalizer=normalizer)

        # Convert the distance map into coordinates. 
        # This also performs the optimization procedure, if needed
        coords = dists.coordinates

        # Save the corrected coordinates. Long-term, should switch to the 
        # save function inside the coordinates object itself.
        torch.save(coords.values.squeeze(),save_dir+f)
    
