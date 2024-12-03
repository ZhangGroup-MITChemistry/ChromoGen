import GEOparse
import pycurl
import certifi
import os
from pathlib import Path

class DipCDownloader:
    '''
    Likely not generalizable to many papers, but gets the myriad files we want for
    this particular study fairly easily. 
    '''

    @staticmethod
    def __parse_gsm_title(title):
        cell_type,_,cell_number = title.split(' ')

        if 'hickit' in cell_number: 
            cell_number,_ = cell_number.split('_') 
            analysis_type = 'hickit'
        else:
            analysis_type = 'Dip-C' 
    
        return cell_type,analysis_type,cell_number

    @staticmethod
    def from_GSE(
        gse_accession,    # GSE accession number
        save_dir='./',         # Directory in which data should be saved
        cell_and_analysis_types=None, # List of tuples listing cell type & analysis type desired.
                                 # For GSE117876, options would some non-empty subset of
                                 # [('GM12878','Dip-C'), ('GM12878','hickit'), ('PBMC','Dip-C'), ('PBMC','hickit')]
                                 # If None, don't select based on this
        gsm_accessions=None,     # Alternatively, select by GSM accessions (list of strings). If None, don't subset by this. 
                                 # Relevant to this paper: GSM3271347-GSM3271371
        nproc = os.cpu_count()   # Number of files to download concurrently. 
        
    ):
        # Ensure download directory exists. 
        #save_dir = Path(save_dir)
        #save_dir.mkdir(exist_ok=True,parents=True)
        
        # Get GSE info
        gse = GEOparse.get_GEO(geo=gse_accession, destdir=save_dir)

        # Select gsms that we are interested in, remove others
        '''
        gsm_accessions = gsm_accessions if gsm_accessions is not None else gsm_accessions
        
        if cell_and_analysis_types is None:
            if gsm_accessions is not None:
                gse.gsms = {gsm_acc:gsm for gsm_acc,gsm in gse.gsms.items() if a
        '''
        ##
        # Select gsms that we are interested in, remove others
        # We'll download the UNION of what is in the provided gsm_accessions if both were provided
        if cell_and_analysis_types is not None:
            gsm_accessions = [] if gsm_accessions is None else gsm_accessions
            
            for gsm_acc, gsm in gse.gsms.items():

                if gsm_acc in gsm_accessions:
                    # Since we're downloading union, don't bother checking if it's already selected in gsm_acc
                    continue
        
                # Get the metadata needed to place the file in the correct location 
                cell_type, analysis_type, cell_number = DipCDownloader.__parse_gsm_title(gsm.metadata['title'][0])
        
                # Check if we want to download this experiment.
                if (cell_type,analysis_type) in cell_and_analysis_types:
                    gsm_accessions.append(gsm_acc)
                    
        if gsm_accessions is not None:
            gse.gsms = {gsm_acc:gsm for gsm_acc,gsm in gse.gsms.items() if gsm_acc in gsm_accessions}

        #####
        # Download the selected GSM files. 
        # The Dip-C data of interest is in the SI files, so just download those. 

        # There doesn't appear to be a destdir (or equivalent) argument in this file, so we'll 
        # just go ahead and change directories
        cwd = os.getcwd()
        os.chdir(save_dir)
        gse.download_supplementary_files(nproc=nproc)
        os.chdir(cwd)

if __name__ == '__main__':

    acc = "GSE117876"
    save_dir = Path(__file__).parent
    DipCDownloader.from_GSE(
        gse_accession = acc, 
        save_dir = Path(__file__).parent,
        #cell_and_analysis_types=[('GM12878','Dip-C')], # Alternative option
        gsm_accessions=[f'GSM32713{k}' for k in range(47,72)],
        nproc = os.cpu_count()
    )
    # Remove the larger GSE file
    (save_dir/'GSE117876_family.soft.gz').unlink()
    # Rename the directory containing all the downloaded data to something more descriptive
    (save_dir/(acc+'_Supp')).rename(save_dir/'tan_single-cell_2018')

 
