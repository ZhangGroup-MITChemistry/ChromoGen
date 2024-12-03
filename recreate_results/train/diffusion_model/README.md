While all of this code exists in the ChromoGen package, I unfortunately didn't have time to test the updated/cleaned-up implementation for training the model (`train_reimplemented.pt`). As such, we provide the uglier version actually used for the paper in `train.py`

For the latter version, the relevant support files are all contained in the `support` directory, as are the files needed to normalize conformations. 
Otherwise, you can simply run `train.py` to train the diffusion model. 

Note that this assumes you have already downloaded the data used in the paper (specifically, the formatted Dip-C conformation file `../../downloaded_data/conformations/DipC/processed_data.h5` and `../../downloaded_data/embeddings/GM12878/chrom_<chrom_name>.tar.gz` for $\text{chrom_name}\in {1,2,3,...,22}$ (chromosome X was excluded from training)). 

Later, you can load these into the updated (and verified!) ChromoGen package using
```python
import ChromoGen

epcot_filepath = 'path/to/EPCOT/model/file.pt'
diffuser_filepath = 'path/to/this/directory/results/model_120.pt'

cgen = ChromoGen.from_files(epcot_filepath,diffuser_filepath)
```
and save an actual ChromoGen file using 
```python
cgen.save('path/to/chromogen/file.pt')
```
which can then be loaded in the future with
```python
cgen = ChromoGen.from_file('path/to/chromoge/file.pt')
```

