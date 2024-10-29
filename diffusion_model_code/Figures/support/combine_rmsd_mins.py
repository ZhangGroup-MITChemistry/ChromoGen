import torch

save_folder = './rmsd_data/'

for task_id in range(100):
    if task_id == 0:
        min_rmsds = torch.load(save_folder + f'tan_on_gen_{task_id}.pt')
    else:
        min_rmsds = torch.min(min_rmsds,torch.load(save_folder + f'tan_on_gen_{task_id}.pt'))

torch.save(min_rmsds,save_folder + 'tan_on_gen.pt')


