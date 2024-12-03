'''
NOTE: This script computes embeddings using double precision. 
This is because we observe a slight numerical inconsistency between different runs 
when using single precision (on order 1e-6). Because this difference is independent 
of the random seed, etc., and isn't fixed by adding lines such as
    torch.backends.cudnn.deterministic = True # Otherwise, small numerical differences arise. 
    torch.backends.cudnn.benchmark = False
we decided to compute the embeddings with double precision for reproducibilities' sake. 
Simply switch to single precision to roughly halve inference time. Using cudnn, etc., will accelerate it further. 
'''

if __name__ == '__main__':
    
    from ChromoGen.model import EPCOT
    from ChromoGen.data_utils import EPCOTInputLoader
    from pathlib import Path
    import torch
    import pandas as pd
    import argparse
    from tqdm.auto import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('--single_precision',action='store_true')
    parser.add_argument('--cell_type',type=str,default='GM12878')
    parser.add_argument('--chromosome',type=str,default='1')
    args = parser.parse_args()
    
    ####################################################
    # Basic preparation 
    
    # Selected values
    dtype = torch.float if args.single_precision else torch.double
    chrom = args.chromosome
    cell_type = args.cell_type
    
    # Filepaths
    this_dir = Path(__file__).parent
    save_fp = this_dir / f'{cell_type}/chrom_{chrom}.tar.gz'
    model_fp = this_dir.parent.parent / 'downloaded_data/models/epcot_final.pt'
    data_dir = this_dir.parent.parent / 'outside_data/sequence_data/'
    alignment_filepath = data_dir / 'hg19/hg19.h5'
    bigWig_filepath = data_dir / f'DNase_seq/hg19/{cell_type}.bigWig'

    # Ensure the save directory exists
    save_fp.parent.mkdir(exist_ok=True,parents=True)
    
    # Load/format the start indices for the dataloader
    start_indices = pd.read_pickle('my_dict.pickle')
    start_indices = {
        chrom:[int(start_kb * 1000) for start_kb,_ in start_indices[chrom]]
    }
    
    # Prepare data loader
    data_loader = EPCOTInputLoader(
        alignment_filepath,
        bigWig_filepath,
        resolution = 1_000, # resolution relevant to EPCOT
        num_bins=1280,      # bins as far as EPCOT is concerned
        pad_size=300,       # padding on each side of each bin (in bp)
        chroms=[chrom],
        batch_size=1,              # How many 'regions' to return at once.
        shuffle=False,             # Whether to shuffle the indices between epochs
        dtype=dtype,
        device=None,
        store_in_memory=True, # Whether to hold the data in (CPU!) memory rather than accessing the file every time.
        bigWig_nan_to=0,  # Replace NaN's in the bigWig file with this value when loaded. Use None to leave them as NaN's
    )
    
    # Model preparation
    epcot = EPCOT.from_file(model_fp).as_sequence_embedder()
    epcot.requires_grad_(False)
    epcot.eval()
    epcot.to(dtype)
    if torch.cuda.is_available():
        epcot.cuda()

    #######
    # Create all the embeddings & save with pandas
        
    si = start_indices[chrom].copy()
    si.sort()
    device = epcot.device
    data = {}
    for start in tqdm(si):
        
        # Load input data
        x = data_loader.fetch(chrom, start) 
        
        # Embed the data & store in the dict
        # Weird indexing is just for backward compatability for something weird we did way back when
        data[(1_300_000, chrom, start)] = epcot(x.to(dtype=dtype, device=device))

    data = pd.DataFrame({'data':data})
    data.index = data.index.set_names(['Region_Length', 'Chromosome', 'Genomic_Index'])

    # Save
    data.to_pickle(save_fp)
