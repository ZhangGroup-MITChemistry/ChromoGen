import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

d = Path('/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/')
f = Path('./file_list.txt')

extensions_to_ignore = ['.pdf','.itp','.egg-info','.pyc','.png','.out','.swp','.hdf5','.linux-x86_64','.pt','.psf','.xyz','.pkl','.pdb','.log']

if f.is_file():
    files = [Path(ff.strip()) for ff in f.open('r').readlines()]
else:
    files = list(d.rglob('*'))
    f.open('w').write(
        '\n'.join([str(ff) for ff in files])
    )


dest1 = Path('./diffusion_model_code/')
def copy_one_file(f,dest1=dest1):
    dest = dest1 / f.relative_to('/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/')
    if dest.exists():
        return
    dest.parent.mkdir(exist_ok=True,parents=True)
    dest.write_bytes(f.read_bytes())

with ThreadPoolExecutor() as executor:
    for f in files:
        if f.suffix in extensions_to_ignore or f.is_dir():
            continue
        executor.submit(copy_one_file,f)

        #dest = dest1 / f.relative_to('/home/gridsan/gschuette/refining_scHiC/revamp_with_zhuohan/')
        #if dest.exists():
        #    continue
        #dest.parent.mkdir(exist_ok=True,parents=True)
        #dest.write_bytes(f.read_bytes())


