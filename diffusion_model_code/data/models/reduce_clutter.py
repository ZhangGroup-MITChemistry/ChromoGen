import os

root_folder = './'
preserve_every = 5
stop_at = None 

folders = [k for k in os.listdir(root_folder) if os.path.isdir(k) and k!='errant_dataloader']

def get_max_milestone(folder):
    max_milestone = 0 
    for filename in os.listdir(folder):
        milestone = int('.'.join(filename.split('.')[:-1]).split('-')[-1])
        if milestone > max_milestone:
            max_milestone = milestone
    return max_milestone

for folder in folders: 
    fp = lambda milestone: root_folder + folder + f'/model-{milestone}.pt'

    up_to = stop_at if stop_at is not None else get_max_milestone(folder)

    for milestone in range(up_to):
        if milestone % preserve_every == 0:
            continue

        if os.path.exists(fp(milestone)): 
            os.remove(fp(milestone))



