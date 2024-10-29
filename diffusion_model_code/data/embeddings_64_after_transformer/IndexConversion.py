'''
You may note that, throughout our code, different regions of the genome are identified by their region_idx (region index/indices), which refers to their index in the embedding files rather than their genomic coordinates/index. 

Here, we provide an interface that converts between region indices and genomic coordinates so that you can identify the regions referenced by each file or choose to selectively investigate regions of your choice.  
'''
import pandas as pd

class IndexConversion:

    # Load the file that keeps track of the relationship between 
    # our region indices and actual genomic coordinates. 
    # The conversion is identical in both IMR-90 and GM12878 embedding schemes. 
    __rosetta_stone = pd.read_pickle('rosetta_stone.pkl')

    @staticmethod
    def __fetch_chrom_data(chrom):
        # Convert to string (if necessary) and remove the 
        # 'chr' people may be used to including due to Hi-C datasets
        # typically including it. 
        chrom1 = str(chrom).replace('chr','')

        # Ensure that the provided chromosome is a rosetta_stone key
        if chrom1 not in IndexConversion.__rosetta_stone:
            add_quotes = lambda x: "'" + str(x) + "'"
            raise Exception(
                f'Argument chrom={chrom} does not correspond to any chromosome identifier.' + 
                '\n' + 
                f"Valid options are: {', '.join([add_quotes(k) for k in range(1,23)])}, and 'X'."
            )
        return IndexConversion.__rosetta_stone[chrom1]


    @staticmethod
    def region_idx_to_coordinates(chrom: str | int, region_idx: int):
        '''
        Chromosome identifier (chrom) can be passed as either an integer (1-22) or
        a string ('1', '2', ..., '22', 'X', 'chr1', 'chr2', ..., 'chr22', 'chrX'). 

        The embedding index, region_idx, must be an integer. 

        Returns the first and last genomic coordinates of this region, i.e., 
        the region spans [first_genomic_coordinate, last_coordinate_plus_one) of 
        chromosome chrom. 

        Values are returned in units of bp. 
        '''
        this_chrom_data = IndexConversion.__fetch_chrom_data(chrom)
        assert len(this_chrom_data) > region_idx, (
            f'The provided region_idx value ({region_idx}) is out of bounds for chromosome {chrom},' + 
            '\n' + 
            'for which {len(this_chrom_data)} embeddings are available.'
        )
        first_genomic_coordinate = this_chrom_data[region_idx][-1]
        last_coordinate_plus_one = first_genomic_coordinate + 1_280_000
        return first_genomic_coordinate, last_coordinate_plus_one

    @staticmethod
    def first_coordinate_to_region_idx(chrom: str | int, first_genomic_coordinate: int):
        '''
        :chrom: The chromosome identifier (see idx_to_coordinates description for details). 
        :first_genomic_coordinate: The first genomic coordinate, in bp, of the region of interest. 

        This returns the index of the embedding associated with the region spanning
        [first_genomic_coordinate,first_genomic_coordinate + 1_280_000) bp in chromosome {chrom}. 

        Note that the first coordinate is divisible by 20,000 in all regions for which embeddings 
        are available, and not all regions are available due to the considerations mentioned in 
        our manuscript. 
        '''
        this_chrom_data = IndexConversion.__fetch_chrom_data(chrom)

        for region_idx,(*_, first_coord) in enumerate(this_chrom_data):
           if first_coord == first_genomic_coordinate:
               return region_idx 

        start_Mb = round(first_genomic_coordinate/1e6,6)
        stop_Mb = round(start_Mb + 1.28,6)
        raise Exception(
            f'No embedding exists for the region spanning {start_Mb}-{stop_Mb} Mb in chromosome {chrom}.'
        )

