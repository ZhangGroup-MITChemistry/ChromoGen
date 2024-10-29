import os

def get_milestone(filepath):
    n = filepath.split('-')[-1] # <milestone>.<filetype>
    return int(n.split('.')[0])

def most_recent_milestone(directory):

    files = os.listdir(directory)
    milestone = -1
    for file in files:
        m = get_milestone(file)
        if m > milestone:
            milestone = m 

    return milestone

def get_model_filepath(model_directory,milestone):
    
    f = model_directory
    f+= '/' if f != '' and f[-1] != '/' else '/'
    f+= f'model-{milestone}.pt'
    
    return f



