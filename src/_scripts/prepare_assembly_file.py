'''
Greg Schuette 2024
'''
def prepare_assembly_file(
    fasta_filepath_url_or_assembly,
    destination=None,
    *,
    source_type='file',
    file_format='.h5',
    **kwargs
):

    if kwargs.get('filetype') is not None:
        assert source_type is None, 'filetype and source_type are synonyms. Cannot define both.'
        source_type = kwargs.pop('filetype')
    
    from pathlib import Path
    
    assert isinstance(source_type,str), (
        'source_type must be a string ("file", "url", or "assembly"). '
        f'Received {type(source_type).__name__}'
    )
    assert source_type in ['file','url','assembly'], (
        'source_type must be a string ("file", "url", or "assembly"). '
        f'Received {type(source_type).__name__} instance.'
    )
    if source_type in ['url','assembly']:
        assert isinstance(fasta_filepath_url_or_assembly,str), \
        f'{source_type.title()} sources require string value for fasta_filepath_url_or_assembly.'
    else:
        assert isinstance(fasta_filepath_url_or_assembly,(str,Path)), (
            'Filepath should be string or pathlib.Path object. '
            f'Received {type(fasta_filepath_url_or_assembly).__name__} instance.'
        )

    if source_type == 'assembly':
        assemblies = {
            'hg19':'https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz',
            'hg38':'https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz'
        }
        assert (url:= assemblies.get(fasta_filepath_url_or_assembly)), \
        f'Unrecognized assembly, {fasta_filepath_url_or_assembly}. Valid choices are {list(assemblies)}'
        if destination is None:
            destination = f'./{fasta_filepath_url_or_assembly}'
        fasta_filepath_url_or_assembly = url
        source_type='url'
    
    from ..data_utils.FASTAHandler import FASTAHandler

    if source_type == 'url':
        if destination is None:
            destination = './'
        
        FASTAHandler.from_url(
            url=fasta_filepath_url_or_assembly,
            save_dir=destination, 
            filetype=file_format,
            **kwargs
        )
        
    else:
        FASTAHandler.from_file(
            filepath=fasta_filepath_url_or_assembly,
            save_dir=destination, 
            filetype=file_format,
            **kwargs
        )
#'''
