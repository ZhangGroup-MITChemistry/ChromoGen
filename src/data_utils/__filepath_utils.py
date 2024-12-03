from pathlib import Path

def is_fasta(filepath):
    f = Path(filepath)
    sfxs = f.suffixes
    if not sfxs:
        # FASTA files should have >= 1 file extension
        return False
    if len(sfxs) == 1 or sfxs[-1] != '.gz':
        # If only one extension and/or this is NOT a gzipped file, 
        # the only/final extension should indicate the file type
        sfx = sfxs[-1]
    else:
        # If it's gzipped the second-to-last extension should indicate file type. 
        # The first check in if guarantees the second-to-last suffix exists in cases
        # where a file only has a .gz extension, as odd as that would be. 
        sfx = sfxs[-2]

    return sfx in ['.fasta', '.fas', '.fa', '.fna', '.ffn', '.faa', '.mpfa', '.frn']

def fasta_fp_to_h5_fp(filepath,verify_fasta=True):
    f = Path(filepath)
    if verify_fasta:
        assert is_fasta(f), f'Filepath {filepath} does not appear to be a FASTA file.'
    if f.suffix == '.gz':
        f = f.with_suffix('') # Need to peel off two extensions, first here second at return
    return f.with_suffix('.h5')
    
def format_h5_filepath(alignment_filepath,parse_fasta_if_needed=False,interactive=False):
    
    # If the provided filepath is a FASTA file, check if the equivalent h5 file exists. If not, 
    # process it if need be (and user desires it), THEN get back to this initialization. \
    f = Path(alignment_filepath)

    # Want to return file with suffix .h5
    sfxs = f.suffixes
    if not sfxs:
        # Assume user sent filepath without extension for simplicity
        h5_f = f.with_suffix('.h5')
    elif sfxs[-1] == '.h5':
        # Already has the proper extension
        h5_f = f
    elif is_fasta(f):
        # If the parent directory exists, 
        h5_f = fasta_fp_to_h5_fp(f, verify_fasta=False)
    else:
        # Perhaps the suffix was malformed... 
        # or perhaps this file has extra periods in its name. 
        # Check if the latter exists. Otherwise, assume the former. 
        if (f1:=f.with_suffix(sfxs[-1] + '.h5')).is_file():
            h5_f = f1
        else:
            h5_f = f.with_suffix('.h5')
    return h5_f

