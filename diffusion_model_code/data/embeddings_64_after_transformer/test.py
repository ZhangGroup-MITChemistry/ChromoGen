from IndexConversion import IndexConversion

region_idx = 500 
for chrom in [str(k) for k in [*range(1,23),'X']]:
    print(f'Chromosome {region_idx} corresponds to ')
    start,stop = IndexConversion.region_idx_to_coordinates(chrom,region_idx)

    print(
        f'Region index {region_idx} in chromosome {chrom} corresponds to genomic '+
        f'coordinates {round(start/1e6,3)}-{round(stop/1e6,3)} Mb.'
    )

    try:
        region_idx_ = IndexConversion.first_coordinate_to_region_idx(chrom,start)
        if 500==region_idx_:
            print('This is successfully recovered.')
        else:
            print(f'ERROR: IndexConversion.first_coordinate_to_region_idx() returned region_idx of {region_idx_}.')
    except:
        print('ERROR: IndexConversion.first_coordinate_to_region_idx() failed to identify this region.')

    print('')

