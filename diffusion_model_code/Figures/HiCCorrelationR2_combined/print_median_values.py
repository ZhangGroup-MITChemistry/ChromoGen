import pandas as pd

for desc,fp in [
    ('Tan_X','data_Tan_X.pkl'),
    ('GM_X','data_GM_X.pkl'),
    ('IMR_X','data_IMR_X.pkl'),
    ('Tan_no_X','data_Tan_no_X.pkl'),
    ('GM_no_X','data_GM_no_X.pkl'),
    ('IMR_no_X','data_IMR_no_X.pkl')
]:

    data = pd.read_pickle(fp)
    frac = 0 if 'Tan' in desc else 0.5
    print(desc, len(data['corr_dist'][frac]))
    for key in ['corr_dist','corr_prob','r2_prob']:
        print(key,round(data[key][frac].median().tolist(),4))

    print('')



