import GEOparse
import pycurl
import certifi
import os

####
# Define the relevant metadata
accession = "GSE117876"
destdir = './test/'
gse = GEOparse.get_GEO(geo=accession, destdir=destdir)

####
# This script can download files in parallel. 
# If nproc is set to None, then the script will
# use the number of processors visible to os.cpu_count(). 
# Otherwise, it will use min(nproc,os.cpu_count())
# threads in parallel while downloading files. 
nproc = 10
if nproc is None:
    nproc = os.cpu_count()
else:
    nproc = min(nproc,os.cpu_count())

####
# Select the data we want to download. 
# These are tuples (<cell type>, <analysis type>), where
# <cell type> can be 'GM12878' or 'PBMC', and 
# <analysis type> can be 'Dip-C' or 'hickit'
data_desired = []
data_desired.append( ('GM12878','Dip-C') )
#data_desired.append( ('GM12878','hickit') )
#data_desired.append( ('PBMC','Dip-C') )
#data_desired.append( ('PBMC','hickit') ) 

####
# Define some support functions
def parse_title(title):

    cell_type,_,cell_number = title.split(' ')

    if 'hickit' in cell_number: 
        cell_number,_ = cell_number.split('_') 
        analysis_type = 'hickit'
    else:
        analysis_type = 'Dip-C' 

    return cell_type,analysis_type,cell_number

####
# Remove gsms that we aren't interested in 
if len(data_desired) != 4: 
    # otherwise, data_desired contains all combinations of cell type/analysis technique
    samples_to_exclude = []
    for acc, gsm in gse.gsms.items():

        # Get the metadata needed to place the file in the correct location 
        cell_type, analysis_type, cell_number = parse_title(gsm.metadata['title'][0])

        # Ensure we wish to download this experiment.
        if (cell_type,analysis_type) not in data_desired: 
            samples_to_exclude.append(acc)
    for acc in samples_to_exclude: 
        gse.gsms.pop(acc)

#### 
# Download all of the relevant files associated with the remaining data.
gse.download_supplementary_files(nproc=nproc)


 
