import torch

'''
def rename_keys(param_name):
    return param_name.replace(
        'model.classes_mlp',
        'embedder'
    )
'''
'''
def get_rename_fcn(diffusion,params):

    start_val = ''
    for key in params.keys():
        if 'classes_mlp' in key: 
'''

def get_rename_keys(diffusion,param_dict):

    keyword = 'classes_mlp'
    to_remove = None
    for key in param_dict.keys():
        if keyword in key:
            end_idx = key.index(keyword) + len(keyword)
            to_remove = key[:end_idx]
            remainder = key[end_idx:]
            break
    if to_remove is None: 
        return lambda key: key
    
    to_accept = None
    for key in diffusion.embedder.state_dict():
        if remainder not in key:
            continue
        temp = key.replace(remainder,'')
        if to_accept is None or len(temp) < len(to_accept):
            to_accept = temp
    if to_accept is None or to_accept == '':
        to_accept = 'embedder'
    else:
        to_accept = 'embedder.' + to_accept
    
    return lambda key: key.replace(to_remove,to_accept)

    
def load_params(diffusion,filepath):

    # Embedder is no longer separate!
    # Account for this! 

    #all_params = torch.load(filepath)['model']
    '''
    unet_params = {}
    embedder_params = {}
    for key,value in all_params.items():
        if 'classes_mlp' in key: 
            embedder_params[rename_to_embedder(key)] = value
        else:
            unet_params[key] = value
    '''
    '''
    for key,value in all_params.items():
        if 'classes_mlp' not in key:
            continue
        key1 = rename_to_embedder(key)
        all_params[key1] = all_params[key]
        all_params.pop(key);
    '''
    params = torch.load(filepath,map_location=diffusion.device)['model']
    rename_keys = get_rename_keys(diffusion,params)
    params1 = {
        rename_keys(key):value for key,value in params.items() if 'model.classes_mlp0.0' not in key
    }

    #diffusion.model.load_state_dict(unet_params)
    #diffusion.embedder.load_state_dict(embedder_params)
    diffusion.model.null_classes_emb = torch.nn.Parameter(params['model.null_classes_emb'])
    diffusion.load_state_dict(params1)

    diffusion.model.null_classes_emb = torch.nn.Parameter(
        diffusion.embedder(diffusion.model.null_classes_emb.unsqueeze(0)).squeeze()
    )


