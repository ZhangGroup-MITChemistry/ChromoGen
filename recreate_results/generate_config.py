from pathlib import Path
import json 

root_dir = Path(__file__).parent
filepaths = {'root_dir':str(root_dir)}

# Downloaded data + URLs
'''
down_dir = root_dir / 'downloaded_data'
down_mod_dir = down_dir/'model_files'
down_emb_dir = down_dir/'embeddings'
filepaths.update({
    'downloaded_data':str(down_dir),
    'chromogen_filepath_downloaded':[
        str(down_mod_dir/'chromogen.pt'),
        'https://TO-BE-DETERMINED'
    ],
    'diffusion_filepath_downloaded':[
        str(down_mod_dir/'diffusion_final.pt'),
        'https://TO-BE-DETERMINED'
    ],
    'epcot_filepath_downloaded':[
        str(down_mod_dir/'epcot_final.pt'),
        'https://TO-BE-DETERMINED'
    ],
    'embedding_dir_down':[
        str(down_emb_dir/)
    ]
})
'''


# Data filepaths
chrom_list = list(range(1,23)) + ['X']
filepaths.update({
    'data':{
        'embeddings':{
            'downloaded':{
                'GM12878':{
                    chrom:f'downloaded_data/embeddings/GM12878/chrom_{chrom}.tar.gz' for chrom in chrom_list
                },
                'IMR90':{
                    chrom:f'downloaded_data/embeddings/IMR90/chrom_{chrom}.tar.gz' for chrom in chrom_list
                }
            },
            'recreated':{
                'GM12878':{
                    chrom:f'generate_data/embeddings/GM12878/chrom_{chrom}.tar.gz' for chrom in chrom_list
                },
                'IMR90':{
                    chrom:f'generate_data/embeddings/IMR90/chrom_{chrom}.tar.gz' for chrom in chrom_list
                }
            }
        },
        'models':{
            'downloaded':{
                'EPCOT':'downloaded_data/models/epcot_final.pt',
                'diffusion_model':'downloaded_data/models/diffusion_final.pt',
                'chromogen':'downloaded_data/models/chromogen.pt',
                'UNet':'downloaded_data/models/unet.pt'
            },
            'recreated':{
                'EPCOT':'train/EPCOT/models/GM12878_256.pt',
                'diffusion_model':'train/diffusion_model/model-120.pt',
                'chromogen':'train/chromogen.pt',
                'UMAP':'train/UMAP/UMAP.pt'
            }
        },
        'conformations':{
            'downloaded':{ # A ton of them, so... just going to give the directory & let scripts systematically choose files
                'genome_wide':'downloaded_data/conformations/genome_wide/',
                'independent_regions':'downloaded_data/conformations/independent_regions/',
                'unguided':'downloaded_data/conformations/unguided/', # Used for Fig. S11
            },
            'recreated':{
                'genome_wide':'generate_data/conformations/genome_wide/',
                'independent_regions':'generate_data/conformations/independent_regions/',
                'unguided':'generate_data/conformations/unguided/', # Used for Fig. S11
            }
        },
        'outside':{
            'inputs':{
                'downloaded':{
                    'alignment':{
                        'FASTA':'downloaded_data/outside_data/inputs/hg19.fa.gz',
                        'h5':'downloaded_data/outside_data/inputs/hg19.h5',
                        'npz':{
                            chrom:f'downloaded_data/outside_data/inputs/hg19/chr{chrom}.npz' for chrom in chrom_list
                        }
                    },
                    'DNase-seq':{
                        'GM12878':{
                            'BigWig':'downloaded_data/outside_data/inputs/GM12878_hg19.bigWig',
                            'pkl':'downloaded_data/outside_data/sequence_data/DNase_seq/GM12878_hg19.pkl'
                        },
                        'IMR90':{
                            'BigWig':'downloaded_data/outside_data/inputs/IMR90_hg19.bigWig',
                            'pkl':'downloaded_data/outside_data/sequence_data/DNase_seq/GM12878_hg19.pkl'
                        }
                    }
                },
                'recreated':{
                    'alignment':'outside_data/sequence_data/hg19.h5',
                    'DNase-seq':{ # There's no post-processing with BigWig files, so just leave in the download folder...
                        'GM12878':{
                            'BigWig':'downloaded_data/outside_data/inputs/GM12878_hg19.bigWig',
                            'pkl':'outside_data/sequence_data/DNase_seq/GM12878_hg19.pkl'
                        },
                        'IMR90':{
                            'BigWig':'downloaded_data/outside_data/inputs/IMR90_hg19.bigWig',
                            'pkl':'outside_data/sequence_data/DNase_seq/GM12878_hg19.pkl'
                        }
                    }
                }
            },
            'Hi-C':{
                'downloaded':{
                    'GM12878':{
                        'hic':'downloaded_data/outside_data/hic/GM12878_hg19.hic',
                        'mcool':'downloaded_data/outside_data/hic/GM12878_hg19.mcool'
                    },
                    'IMR90':{
                        'hic':'downloaded_data/outside_data/hic/IMR90_hg19.hic',
                        'mcool':'downloaded_data/outside_data/hic/IMR90_hg19.mcool'
                    },
                }
            }
        }
    },
    'scripts':{
        'train':{
            'EPCOT':'./train/EPCOT/downstream_train_hic.py',
            'diffusion':'./train/diffusion_model/train.py',
            'UMAP':'./train/UMAP/train_UMAP.py'
        },
        'generate_conformations':{
            'genome_wide':'generate_data/conformations/genome_wide.py',
            'independent_regions':'generate_data/conformations/independent_regions.py',
            'unguided':'generate_data/conformations/unguided.py', # Used for Fig. S11
        }
    },
})

###########
# Insert root directory everywhere
def add_root(fps,root_dir=root_dir):
    fp1 = {}
    for key, path in fps.items():
        if isinstance(path,dict):
            fp1[key] = add_root(path,root_dir)
        else:
            fp1[key] = str(root_dir / path)

    return fp1

filepaths = add_root(filepaths)

######################
# URLs for downloading data
urls = {}

######################
# Combine and save
config = {
    'filepaths':filepaths,
    'URLs':urls
}

json.dump(config, (root_dir/'config.json').open('w'),indent=2)

