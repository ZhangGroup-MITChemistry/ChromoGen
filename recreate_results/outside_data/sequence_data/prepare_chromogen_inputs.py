from pathlib import Path
import pycurl
import warnings
from ChromoGen import prepare_assembly_file

def download_data(url,destination,overwrite=False):
    if isinstance(destination,(str,Path)):
        destination = Path(destination)
        if not overwrite and destination.exists():
            warnings.warn(f'{destination} already exists. Skipping download of {url}.')
        destination.parent.mkdir(exist_ok=True,parents=True)
        destination = destination.open("wb")
    # Else, assume we have a pre-opened destination
    curl = pycurl.Curl()
    curl.setopt(pycurl.URL, url)
    curl.setopt(pycurl.WRITEDATA, destination)
    curl.setopt(curl.NOPROGRESS, False)
    curl.perform()
    curl.close()


if __name__ == '__main__':

    ###########################
    # Your selections. Including all data from the manuscript by default
    # Can also give path to FASTA file with source_type = 'file' or another download url with source_type='url'
    assembly = 'hg19'
    source_type = 'assembly'
    assembly_formats = ['.h5','.npz'] # .h5 for chromogen rewrite of EPCOT, .npz for EPCOT training on original code. 
    cell_types = ['GM12878','IMR90']
    max_workers = None # threads. default: number cpus + 4. The assembly processing takes quite a bit of RAM, so that'd be the main reason to reduce
    chromogen_inputs_dir = Path('./chromogen_inputs/')
    epcot_inputs_dir = Path('./epcot_training_inputs/')

    bigwig_download_urls = {
        'hg19':{
            'GM12878':'https://encode-public.s3.amazonaws.com/2017/09/06/e2259e48-add5-4b57-bc75-e27b290b954f/ENCFF901GZH.bigWig',
            'IMR90':'https://encode-public.s3.amazonaws.com/2017/09/27/ab857d71-030e-41f1-9034-69c1355a6821/ENCFF291DOH.bigWig'
        }
    }

    ###########################
    # Download the BigWig files
    from concurrent.futures import ThreadPoolExecutor
    urls = bigwig_download_urls.get(assembly)
    print('Attempting DNase-seq downloads',flush=True)
    if not urls:
        print(f'No DNase-seq download links are included for assembly {assembly}.',flush=True)
    elif max_workers == 1:
        for ct in cell_types:
            url = urls.get(ct)
            if url:
                download_data(url, chromogen_inputs_dir / f'{ct}.bigWig')
            else:
                print(f'No {assembly}-aligned DNase-seq download links are included for cell type {ct}.',flush=True)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Track futures so that we can raise any exceptions that are otherwise buried
            futures = [] 
            for ct in cell_types:
                url = urls.get(ct)
                if url:
                    futures.append(
                        executor.submit(download_data, url, chromogen_inputs_dir / f'{ct}.bigWig')
                    )
                else:
                    print(f'No {assembly}-aligned DNase-seq download links are included for cell type {ct}.',flush=True)
            for future in futures:
                future.result()
    print('Done with DNase-seq downloads',flush=True)

    ###########################
    # Download the FASTA file and parse it. 
    # This is done separately for the EPCOT training data (.npz files) 
    # and ChromoGen input (.h5 file). The content is identical, but there are...
    # a lot of benefits to using the HDF5 format. 

    # These processes are multithreaded internally. So, don't make an executor here. 
    for fmt in assembly_formats:
        if fmt == '.npz':
            destination = epcot_inputs_dir / assembly
        else:
            destination = chromogen_inputs_dir           
            
        prepare_assembly_file(
            fasta_filepath_url_or_assembly=assembly,
            destination=destination,
            source_type='assembly',
            file_format=fmt,
            # See ChromoGen.data_utils.FASTAHandler.FASTAHandler.from_url and ...from_file
            # for additional kwargs
        )


