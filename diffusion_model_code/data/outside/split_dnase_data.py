import pickle

for f in ['GM12878_hg19.pkl','IMR90_hg19.pkl']:

    data = pickle.load(open(f,'rb'))

    for key,item in data.items():
        pickle.dump(item,open('dnase/'+f.replace('.pkl','')+f'_{key}.pkl','wb'))



