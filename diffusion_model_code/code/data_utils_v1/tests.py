for cell_numbers in [[1],[17],[1,17],[2,3,4,5]]:
    
    dl = DataLoader(
        dataset_filepath,
        segment_length=nbeads,
        batch_size=batch_size,
        normalize_distances=True,
        geos=None,
        organisms=None,
        cell_types=None,
        cell_numbers=train_cells,
        chroms=None,
        replicates=None,
        shuffle=True,
        allow_overlap=False,
        two_channels=False,
        try_GPU=True,
        mean_dist_fp='../../data/mean_dists.pt',
        mean_sq_dist_fp='../../data/squares.pt'
    )

    assert len(dl.coords) == dl.coord_info['idx_max'].iloc[-1]+1
    for k in range(1,len(dl.coord_info)):
        assert dl.coord_info.loc[k,'idx_min'] == dl.coord_info.loc[k-1,'idx_max']+1