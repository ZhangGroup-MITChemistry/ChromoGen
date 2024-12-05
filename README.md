# ChromoGen

## Table of contents

- [Introduction](#introduction)
- [References](#references)
- [Using ChromoGen](#using-chromogen)
    - [Requirements](#requirements)
    - [Installation](#installation)
        - [Clone repository](#clone-repository)
        - [Create and activate a dedicated environment](#create-and-activate-a-dedicated-environment)
        - [Install ChromoGen package](#install-chromogen-package)
    - [Download and reproduce results](#download-and-reproduce-results)
        - [File descriptions and reproducing script locations](#file-descriptions-and-reproducing-script-locations)
        - [Obtain Zenodo-preserved code](#obtain-zenodo-preserved-code)
        - [Obtain Zenodo-preserved data](#obtain-zenodo-preserved-data)
    - [Preparing sequencing inputs](#preparing-sequencing-inputs)
    - [Generating conformations](#generating-structures)
    - [Analyzing conformations](#analyzing-conformations)
        - [Dimensionality](#dimensionality)
        - [Initialize a Conformations object](#initialize-a-conformations-object)
        - [Universal properties/methods](#universal-propertiesmethods)
        - [Distances-specific properties/methods](#distances-specific-propertiesmethods)
        - [Coordinates-specific properties/methods](#coordinates-specific-propertiesmethods)
        - [Hi-C maps](#hi-c-maps)

## Introduction

As introduced in ["ChromoGen: Diffusion model predicts single-cell chromatin conformations"](https://doi.org/10.21203/rs.3.rs-4630850/v1), ChromoGen is a generative AI model that predicts the 3D conformations undertaken by chromatin _in vivo_ while only requiring sequencing data as input. ChromoGen's front-end, adapted from [EPCOT](https://doi.org/10.1093/nar/gkad436) (their [GitHub](https://github.com/liu-bioinfo-lab/EPCOT/tree/main)), extracts structure-influencing features from DNA sequence and DNase-seq data and creates information-rich, low-dimensional encodings; the DNA sequence and DNase-seq data provide the information to make region- and cell type-specific predictions, respectively. The low-dimensional encodings are then passed to ChromoGen's diffusion model backbone, which stochastically generates the conformations, which are independent and identically distributed (assuming the same condition is applied). In this way, ChromoGen avoids the time- and compute-intense navigation of state space required by molecular dynamics simulations to provide rapid predictions. 

Predicting genome conformations (and recreating the results of our manuscript) requires three things:
1. You need to install the ChromoGen package, which corresponds to code inside `./src/`. See [Installation](#installation) for details.
2. You must either train your own version of ChromoGen or download our model parameters. See [Obtain Zenodo-preserved data](#obtain-zenodo-preserved-data)) for details. 
3. You must download/prepare the sequencing data relevant to the assembly and cell type of interest to you. See [Preparing sequencing inputs](#preparing-sequencing-inputs) for details. 

If your main goal is to reproduce our results, see [Download and reproduce results](#download-and-reproduce-results). Otherwise, skip to [Preparing sequencing inputs](#preparing-sequencing-inputs), [Generating conformations](#generating-structures), [Analyzing conformations](#analyzing-conformations) for a more intentional introduction to the ChromoGen package. 

## References

"ChromoGen: Diffusion model predicts single-cell chromatin conformations":
- Research Square-hosted [preprint](https://doi.org/10.21203/rs.3.rs-4630850/v1)
- Zenodo-hosted [dataset](https://doi.org/10.5281/zenodo.14218666)

## Using ChromoGen

### Requirements

To run the installation, you will need to install `conda` or (for faster installation/environment management) `mamba`. Our tests used the `mamba` commands with `mamba 1.4.2` and `conda 23.3.1` installed. 

[comment]: ChromoGen relies on several packages, most principally `Python 3.10.12`, `PyTorch 2.0.1` for the model and data processing, `pyBigWig 0.3.22` for handling BigWig-formatted DNase-seq files, and `matplotlib 3.7.5` for visualization. A number of other packages are required, which are included in the YAML file referenced in the next subsection. 

To make structure predictions with ChromoGen, you'll also need to download the model file (download URL TBD), BigWig-formatted DNAse-seq data for the cell type of interest to you, and create an HDF5-formatted DNA sequence file (see [Preparing sequencing inputs](#preparing-sequencing-inputs)). 

### Installation

#### Clone repository

Navigate to the directory where you want to clone the repository, then run
```bash
git clone git@github.com:ZhangGroup-MITChemistry/ChromoGen.git
```

If you'd prefer to use the Zenodo-preserved version of the code, see [Obtain Zenodo-preserved code](#obtain-zenodo-preserved-code). Note that, as of the Zenodo dataset's creation date, they are identical. 

#### Create and activate a dedicated environment

Now, create a dedicated environment in which to use the ChromoGen package, which we verified using Python 3.10.12. For your reference, this repo includes the YAML file exported from the environment we used, `environment.yml`, but conda unfortunately fails to install directly from that file.

Instead, create a new environment like so:
```bash
conda create -n ChromoGen python=3.10.12 mamba jupyterlab pip=23.2.1 GEOparse=2.0.4 hic2cool --no-default-packages -c conda-forge -c anaconda -c bioconda -c hcc
```
A few notes about this:
1. This will run much faster if you replace `conda` with `mamba`, assuming you have that installed.
2. While not requirements of the ChromoGen package, we also install:
    1. `jupyterlab` so that you can run the jupyter notebooks inside `./recreate_results/create_figures/`;
    2. `mamba` to accelerate any additional changes you want to make to the environment;
    3. `pip 23.2.1` because that's the version we used; and
    4. `GEOparse` and `hic2cool` because certain scripts in `./recreate_results/` require them.

Activate your new environment before moving to the next section:
```bash
conda activate ChromoGen
```

#### Install ChromoGen package

Inside your new `conda` environment, navigate to the directory containing this README file and `setup.py` and run
```bash
pip install . 
```
Once complete, the ChromoGen package can be imported in Python using:
```python
import ChromoGen
``` 

### Download and reproduce results

#### File descriptions and reproducing script locations

All data associated with the first ChromoGen publication is preserved on Zenodo with DOI [`10.5281/zenodo.14218666`](https://doi.org/10.5281/zenodo.14218666). It is spread across six files:
1. `chromogen_code.tar.gz` contains all code and, as of its upload date, is identical to this repo.
2. `epcot_final.pt` contains the fine-tuned EPCOT parameters.
    - To independently fine-tune EPCOT, see `./recreate_results/train/EPCOT/`.
3. `chromogen.pt` contains the complete set of ChromoGen model parameters, including both the relevant EPCOT parameters and all diffusion model parameters.
    - To independently train the diffusion model, see `./recreate_results/train/diffusion_model/README.md`.
4. `conformations.tar.gz` contains all conformations analyzed in the manuscript, including the Dip-C conformations formatted in an HDF5 file, all ChromoGen-inferred conformations, and the MD-generated MD homopolymer conformations. Descriptively named subdirectories organize the data.
    - To generate data equivalent to `conformations/MDHomopolymer/DUMP_FILE.dcd`, run either `job.pbs` or `run.sh` within `./recreate_results/generate_data/conformations/MDHomopolymer/`.
    - To generate data equivalent to `conformations/ChromoGen/genome_wide/` run `./recreate_results/generate_data/conformations/genome_wide.py`
    - To generate data equivalent to `conformations/ChromoGen/specific_regions/` run `./recreate_results/generate_data/conformations/independent_regions.py`
    - To generate data equivalent to `conformations/ChromoGen/unguided/` run `./recreate_results/generate_data/conformations/unguided_conformations_for_S11.py`
    - To recreate `conformations/DipC/processed_data.h5`, run `./recreate_results/outside_data/dipc/process_3DG_science_2018.py`
5. `outside_data.tar.gz` contains two subdirectories:
    1. `inputs` contains our pre-processed genome assembly file (produced as in [Preparing sequencing inputs](#preparing-sequencing-inputs)).
      - You'll also need to place the BigWig files in here, as in the script below.
      - To recreate the included `hg19.h5` file and format the DNase-seq data for training EPCOT, run `./recreate_results/outside_data/sequence_data/prepare_chromogen_inputs.py`.
      - [Preparing sequencing inputs](#preparing-sequencing-inputs) includes general instructions for formatting ChromoGen input data. 
    2. `training_data` contains the Dip-C conformations formatted in an HDF5 file.
      - To recreate this file, run `./recreate_results/outside_data/dipc/process_3DG_science_2018.py`.
6. `embeddings.tar.gz` contains the fine-tuned EPCOT-generated sequence embeddings for each region included in the diffusion model's training set.
    - To reproduce each file, run `./recreate_results/generate_data/EPCOT/generate_embeddings.py --chromosome <chrom>` 23 times using `<chrom>` values of 1, 2, 3, ..., 22, and X.

See `./recreate_results/create_figures/Figure_2/UMAP_scripts/README.md` for instructions on recreating the various UMAP figure panels. All other figures that require computational work are produced by the Jupyter Notebooks within the relevant subdirectories of `./recreate_results/create_figures/`.

Some of these require Hi-C data, which we do not included on Zenodo since we didn't originate the data. This includes:
1. Hi-C data, which can be downloaded and prepared at 20 kb resolution using `./recreate_results/outside_data/hic/download_and_process.sh`;
2. DNase-seq data, which is downloaded by `./recreate_results/outside_data/sequence_data/prepare_chromogen_inputs.py`; and
3. Chromatin tracing structures, which you can download and prepare following the instructions within `./recreate_results/create_figures/Figure_S1/bintu_data/README.md`.

#### Obtain Zenodo-preserved code

If you want to use the Zenodo-preserved code rather than the GitHub-hosted version, then navigate to the directory where you'd like to place the code and run
```bash
wget https://zenodo.org/records/14218666/files/chromogen_code.tar.gz
tar -xvzf chromogen_code.tar.gz
rm chromogen_code.tar.gz
```
and go back to step [Create and activate a dedicated environment](#create-and-activate-a-dedicated-environment). 

Otherwise, continue to the next step. 

#### Obtain Zenodo-preserved data

For the scripts within `./recreate_results/` to run properly, these files must be downloaded and decompressed placed in the correct locations within the repo. To do so, navigate to the directory containing this file and run the following:
```bash
# Create the directories that'll contain the data
mkdir -p recreate_results/downloaded_data/models
cd recreate_results/downloaded_data

# Download all the data
wget https://zenodo.org/records/14218666/files/conformations.tar.gz &
wget https://zenodo.org/records/14218666/files/embeddings.tar.gz &
wget https://zenodo.org/records/14218666/files/outside_data.tar.gz &
cd models
wget https://zenodo.org/records/14218666/files/chromogen.pt &
wget https://zenodo.org/records/14218666/files/epcot_final.pt &
cd ..
wait 

# Untar the three tarballs
tar -xvzf conformations.tar.gz &
tar -xvzf embeddings.tar.gz &
tar -xvzf outside_data.tar.gz &
wait

# Remove the now-unneeded tarballs
rm conformations.tar.gz embeddings.tar.gz outside_data.tar.gz
```

### Preparing sequencing inputs

For the DNase-seq data, simply download the BigWig file of interest to you and note the file's location (which will be needed to [generate conformations](#generating-conformations)). 

The DNA sequence data, however, must be formatted as an HDF5 file using our custom layout. To prepare this, run the following in Python (CLI to come...)
```python
import ChromoGen as cg
cg.prepare_assembly_file(fasta_filepath_url_or_assembly)
```
where `fasta_filepath_url_or_assembly` may be `"hg19"`, `"hg38"`, the URL to a FASTA file of your choice (all of the options so far require an internet connection), or the path to a FASTA file on your local system. If you pass a url, you must also pass `source_type="url"`. If you pass an assembly, you must pass `source_type="assembly"`. 

Additional options are:
- `destination`: The destination filepath of your choice. Default: Same name as the passed FASTA file -- but with h5 extension -- and same directory as source if a file passed, otherwise the same directory you run this. 
- `file_format`: The default is `".h5"`, but if you want to create .npz-formatted files as in the original EPCOT paper, you can pass `".npz"`

### Generating conformations

`./reproduce_results/generate_data/conformations/` contains three scripts that you can run to produce conformations equivalent to those used in the paper. 

Otherwise, here's are some basic instructions to help you write your own scripts:
```python
import ChromoGen as cg

############################################################
# Prepare ChromoGen

# Initialize ChromoGen & load the trained parameters
cgen = cg.from_file('/path/to/model/file.pt')

# Assuming you have a GPU available, move to the GPU to improve speed
cgen.cuda()

# Tell chromogen where the sequencing files are located
cgen.attach_data(
    alignment_filepath="/path/to/HDF5-formatted/DNA/sequence/file.h5",
    bigWig_filepath="/path/to/BigWig-formatted/DNase-seq/file.bw"
)

# If you run into memory contraints, etc., you can try modifying the following.
# The default value is 1000 (so this line doesn't change anything). 
# If you request more conformations than maximum_samples_per_gpu, ChromoGen
# automatically splits the task into multiple batches and combines the results
# before returning them.  
cgen.maximum_samples_per_gpu = 1000 

############################################################
# Choose your generation settings.

######
# Optional keyword arguments to ChromoGen's forward function.
# Unless otherwise noted, the default values are shown.

# Samples to generate per region specified when calling ChromoGen.
# The default value is 1, though typically many more will be desired.
# Must be a positive integer. 
samples_per_region = 1000

# Whether to return coordinates (True) or distance maps (False). Must be a bool. 
# If True, the coordinates will be optimized by the algorithm specified in the paper. 
return_coords=True

# If distance maps are requested (i.e., return_coords=True, otherwise this is ignored), 
# this specifies whether the distance maps should be optimized to ensure physical validity
# (by same process as is used to get coordinates) or whether the raw diffusion model 
# output should be returned. 
correct_distmap=False

# Guidance parameters to use. 
# Where w is the guidance strength mentioned in the paper, cond_scale = w+1. 
# For all structures in the paper, we used w=0 with \phi=0 and w=4 with \phi=8, 
# hence this combination is set as the default value. 
# ChromoGen automatically sorts conformations when returned such that those generated with the 
# smallest cond_scale appear first, largest cond_scale appear last, etc. 
# Within a given cond_scale, conformations generated with different rescaled_phi 
# values are sorted the same way.  Both cond_scales and rescaled_phis are lists of floats. 
cond_scales = [1.,5.]     # Guidance strength + 1, i.e., sample from P(x|c) * P(c|x)^{cond_scale-1}
rescaled_phis = [0.,8.]   # Rescaling coefficient to help minimize artifacts introduced by larger guidanance strengths

# What proportion of total samples to generate with the different guidance parameters specified above. 
# Default (None) generates an equal number of conformations from all guidance parameters. 
# (In this script, that means 500 would be generated with 
# cond_scale=5/rescaled_phi=8 and 500 with cond_scale=1. & rescaled_phi=8
# to obtain the 1000 requested samples in a given region.) 
proportion_from_each_scale = None

# Set to False if you want to sample while ChromoGen is in training mode (such that dropout layers are active)
# and/or outside of torch.inference_mode.
force_eval_mode=True

# This does nothing right now, but will eventually specify whether to automatically
# broadcast ChromoGen across many GPUs to split up the total workload across resources. 
distribute=True

# Whether to suppress messages from ChromoGen. Progress bars will show regardless. 
silent=False

# For ease of notation later in this script
kwargs = {
    'samples_per_region':samples_per_region,
    'return_coords':return_coords,
    'correct_distmap':correct_distmap,
    'cond_scales':cond_scales,
    'rescaled_phis':rescaled_phis,
    'proportion_from_each_scale':proportion_from_each_scale,
    'force_eval_mode':force_eval_mode,
    'distribute':distribute,
    'silent':silent
}

############################################################
# Generate samples
# You can specify the genomic region(s) to analyze one of three ways

##########
# Option A: Specify chromosome & first genomic index (within that chromosome)
# as arguments IN THAT ORDER

# Generate conformations based on chromosome number and starting genomic index (in bp)
conformations = cgen(1,8_680_000,**kwargs)

# You can also pass chromosome names as a string
conformations = cgen('1', 8_680_000,**kwargs)

# You may also use the naming convention often used in mcool files
conformations = cgen('chr1', 8_680_000,**kwargs)

# You may also generate conformations in multiple regions simultaneously
conformations = cgen(1,8_680_000, 1, 29_020_000,**kwargs)

##########
# Option B: List of tuples
conformations = cgen([(1,8_680_000), (1, 29_020_000)],**kwargs)


##########
# Option C: Dictionary with chromosome names as keys, lists of start genomic indices as values
regions = {
    1: [8_680_000, 29_020_000], 
    22:[26_260_000] 
}
conformations = cgen(regions, **kwargs)

############################################################
# Save outputs

# ChromoGen returns a Conformations instance, so you can directly save as:
conformations.save('/desired/filepath.pt')
```

Note that ChromoGen returns Conformations instances with shape (a, b, c, d) with
- a: Number of regions passed to ChromoGen (1, 2, then 3 in the script above)
- b: samples_per_region
- c: Monomer/genomic bin index
- d: Either xyz coordinates (return_coords=True) or Monomer/genomic bin index (return_coords=False)


### Analyzing conformations

ChromoGen returns a Conformations subclass (`ChromoGen.Conformations._Coordinates.Coordinates` if `return_coords=True` above, otherwise `ChromoGen.Conformations._Distances.Distances`). These have a lot of convienient properties/methods, and you can look at the Jupyter notebooks inside `./recreate_results/create_figures/` to see some examples of how to perform various analyses using these. Here is a basic guide. 

#### Dimensionality

Conformations have three or more dimensions (... optional batch dimensions ..., mandatory batch dimension, genomic bin/monomer index, xyz positions (if coordinates object) or genomic bin/monomer index (if distances)). 

#### Initialize a Conformations object

ChromoGen returns a Conformations object. Otherwise, you initialize these objects several ways, including:

1. Load from a .pt file (as is typically saved by the class)
```python
from ChromoGen import Conformations

fp = 'path/to/save/file.pt'
coords = Conformations(fp)
```

2. Load from a DCD file. You must pass num_beads (number of monomers in the chain), the path to a topology file, or an mdtraj.Topology instance.
```python
from ChromoGen import Conformations

dcd_fp = 'path/to/trajectory/file.dcd'
top_fp = 'path/to/topology/file.psf' # or whichever other topology file format
dcd_coords1 = Conformations(dcd_fp, topology_file=top_fp)
dcd_coords2 = Conformations(dcd_fp, num_beads=dcd_coords1.num_beads)
dcd_coords3 = Conformations(dcd_fp, topology_file=dcd_coords1.trajctory.topology)
```

3. Initialize from torch.Tensor (assuming it has >= 2 dimensions in order mentioned in prior subsection)
```python
from ChromoGen import Conformations
import torch

####
# Generate 1000 random walks, each with 100 steps (per dimensions described in the prior subsection)

# Choose direction of each step
random_walk = torch.randn(1000, 100, 3)

# Set each step length to 1
random_walk/= torch.linalg.vector_norm(random_walk, dim=-1, keepdim=True)

# Take the walk
random_walk.cumsum_(-2)

# Also pretend we started with distances in a second case
random_walk_dists = torch.cdist(random_walk, random_walk)

####
# Convert to Conformations classes for convenient analysis

# This returns a ChromoGen.Conformations._Coordinates.Coordinates instance
random_walk = Conformations(random_walk)

# This returns a ChromoGen.Conformations._Distances.Distances instance
random_walk_dists = Conformations(random_walk_dists)
```

4. Initialize from a NumPy ndarray
```python
from ChromoGen import Conformations
import mdtraj as md

####
# Use MDTraj to load coordinates saved in a format not specifically supported by 
# ChromoGen.Conformations
xtc_fp = 'path/to/MD/simulation/output.xtc'
top_fp = 'path/to/topology.pdb'
traj = md.load(xtc_fp, top=top_fp)

####
# Place in ChromoGen.Conformations._Coordinates.Coordinates
# object for convenient analysis
xtc_coords = Conformations(traj.xyz)
```

**NOTE:** By default, conformations with NaN values will be included, which may well ruin your analysis. You can automatically drop those conformations using the `drop_invalid_conformations` keyword argument (default: False). 
```python
coords = Conformations('/path/to/saved/coords.pt', drop_invalid_conformations=True)
```

#### Universal properties/methods

All Conformations instances (subclasses of `ChromoGen.Conformations._ConformationsABC.ConformationsABC`, specifically the Coordinates, Distances, and to-be-deprecated Trajectory classes) have several shared properties, including

```python

# using coords as example, but again, these are present in ALL classes. 
coords = Conformations(torch.rand(1000,3)) # Automatically reshaped to size (1,1000,3)


#########
# Convert between representations/classes 

# Get pairwise distances in a Distances instance
dists = conformations.distances

# Get xyz coordinates in a Coordinates instance
coords1 = coords.coordinates

# Get xyz coordinates inside a Trajectory instance, which uses MDTraj for a lot of things
traj = coords.trajectory

#########
# Basic information about the instance (and manipulations, where possible)

# Get the number of beads/monomers/genomic bins
num_beads = coords.num_beads

# Get the object's dimensions/shape
shape = coords.shape

# Get the number of conformations across all batch dimensions
n_conformations = len(coords) 

# Is there one batch dimension? 
has_one_batch_dimension = coords.is_flat

# Reshape such that there is only one batch dimension
coords1 = coords.flatten()
coords.flatten_() # in-place

#########
# Index: Same as torch.Tensor, but must return valid shape (e.g., can't drop x-, y-, or z-axis)

coords1 = coords[0,:32,:] # coords1.shape == (1,32,3)

#########
# Access the torch.Tensor, some of its properties, and manipulate some of them

# The Tensor
coords_pytorch = coords.values

# Get the datatype
dtype = coords.dtype

# Change the datatype
coords = coords.double()
coords = coords.float()
coords = coords.to(torch.double)
coords.float_()         # in-place (at least, affect this instance)
coords.double_()        # in-place
coords.to_(torch.float) # in-place

# Get the device
device = coord.device

# Change the device
coords = coords.cuda()
coords = coords.cpu()
coords = coords.to('cuda:0')
coords.cuda_()                     # in-place (at least, affect this instance)
coords.cpu_()                      # in-place
coords.to_(torch.device('cuda:0')) # in-place

# Change datatype AND device
coords = coords.to(dtype=torch.double, device='cpu') # in-place (at least, affect this instance)
coords.to(dtype=torch.float, device='cuda')          # in-place

#########
# Combine conformations instances

# Concatenate. cat(), append(), and extend() are synonyms, as are cat_(), 
# append_(), and extend_()
coords1 = coords.cat(coords) # Default dimension is 0. Can pass dim=<dimension>
coords1.cat_(coords) # in-place
coords1 = coords.append([coords, coords, coords]) # also valid
coords1 = coords.extend([coords.values, coords, coords.distances, coords.coordinates]) # return type determined by the instance whose method is called

# Stack
coords1 = coords.stack(coords) # Shape (1, 1000, 3) -> (2,1,1000,3). Can also specify stack dimension
coords1.stack_(coords1) # in-place, (2,1,1000,3) -> (2,2,1,1000,3)

#########
# Compute Hi-C map. These return ChromoGen.data_utils.HiCMap.HiCMap instances

# Using same method as in paper
hic_map1 = coords.compute_hic()

# Can specify your own r_c, sigma, and exponent for the conversion function.
# Exponent is called decay_rate in the jupyter notebooks
hic_map2 = coords.compute_hic(r_c=8, sigma=3, exponent=6000) 

# Can also use a simple threshold to compute Hi-C
hic_map3 = coords.compute_hic(r_c=2, threshold=True) 

# Can also get a list of single-cell maps 
# (list of ChromoGen.data_utils.HiCMap.HiCMap instances if one batch dimension, 
# and nested lists are match the batch dimensions otherwise)
schic_maps1 = coords.compute_hic(r_c=2, threshold=True, single_cell=True)
schic_maps2 = coords.compute_hic(single_cell=True)

#########
# Save as pt 
coords.save('path/to/file.pt')
```

#### Distances-specific properties/methods

```python

# Compute coordinates from distances WITHOUT our optimization step. 
# Note: If the distance map is exact, it'll default to this anyway. 
coords2 = dists.uncorrected_coordinates

##########
# Basic distance map properties 
median_dists = dists.median
mean_dists = dists.mean
standard_deviation_of_distances_at_each_interaction_pair = dists.std
variance_of_distances_at_each_interaction_pair = dists.var

# Normalize the distance map as in the paper
norm_dists = dists.normalize() # use normalize_() to do this in-place

# Unnormalize dists
true_dists = norm_dists.unnormalize() # use unnormalize_() to do this in-place

# Fold distance map as in the paper
folded_dists = dists.fold() # use fold_() to do this in-place

# Unfold distance maps
standard_dist_rep = folded_dists.unfold() # use unfold_() to do this in-place

##########
# Plotting

# Can always do this. Will default to the first conformation if multiple are present. 
fig,ax,im,cbar = median_dists.plot()

# If multiple conformations are present, can choose one of two ways
fig,ax,im,cbar = dists[desired_index].plot()
fig,ax,im,cbar = dists.plot(desired_index) # this one only works if there's just one batch dimension

# Can also pass fig, ax, set cbar_orientation to horizontal, etc. See the Jupyter notebooks 
# for examples
```

#### Coordinates-specific properties/methods


```python

####
# Compute radius of gyration

# Use PyTorch implementation 
radii_of_gyration = coords.compute_rg() 
    
# Use MDTraj implementation. Relatively slow
radii_of_gyration1 = coords.compute_rg(use_mdtraj=True)

####
# Center coordinates
centered_at_origin = coords.center_coordinates() 
coords.center_coordinates_() # in-place

####
# Save conformations in DCD format VMD visualization.
# This has the same args, kwargs as mdtraj.Trajectory.save_dcd
coords.save_dcd('path/to/file.dcd') 

####
# Superpose. 
# Have yet to implement a PyTorch version, so you must pass use_mdtraj=True
reference = coords[0]
superposed_coords = coords.superpose(reference, use_mdtraj=True)

####
# Compute RMSD. If use_mdtraj=False (default), be sure to center coordinates first
rmsds1 = coords.rmsd(coords[0])
rmsds2 = coords.rmsd(coords) # obviously, gives all 0's
rmsds3 = coords.rmsd(coords[0], use_mdtraj=True)
rmsds4 = coords[0].rmsd(coords, use_mdtraj=True)

####
# Drop invalid conformations (rather than doing it at initialization)
# Note that this flattens the Coordinates instance, leaving just one batch dimension.
coords1 = coords.drop_invalid_conformations()
coords.drop_invalid_conformations() # in-place

```

#### Hi-C maps

If you predicted a Hi-C map (see [Universal properties/methods](#universal-properties/methods)), you can plot it with
```python
hic_map.plot()
```
You can plot two Hi-C maps -- one on the upper triangle, one on the lower triangle -- using
```python
hic_map1.plot_with(hic_map2) # optional: self_on_upper, bool, default = True
```
In both cases, you can specify vmin, vmax, cbar_orientation='horizontal', fig, ax, etc. See Jupyter Notebooks for examples.

You can also compute insulation scores using the same approach as in the paper using
```python
hic_map.compute_insulation()
```
with optional argument `window_size` that defaults to 7. 

You can compute the mean value of any diagonal using
```python
mean1 = hic_map.diagonal_mean()  # default: Main diagonal
mean2 = hic_map.diagonal_mean(1) # set offset to 1, meaning first off-diagonal
```

You can normalize a Hi-C map (multiply by some constant such that the first off-diagonal's mean equals some chosen value) using
```python
# Normalize to selected constants
hic_map1 = hic_map.normalize()  # Default: Set first off-diagonal average to 1
hic_map1 = hic_map.normalize(2) # Manually choose first off-diagonal average to 2
hic_map1.normalize_(3) # Set it to 3 in-place

# Normalize to some other map for easier visualization
hic_map1 = hic_map.normalize_as(hic_map2)
hic_map1.normalize_as_(hic_map3) # in-place
```
