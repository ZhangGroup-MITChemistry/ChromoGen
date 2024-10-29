from pathlib import Path
import mdtraj as md
import os
from tqdm.auto import tqdm

# To help parse data
def interpret_tan_dcd(dcd_file):
    f = '.'.join(dcd_file.split('/')[-1].split('.')[:-1])
    details = f.split('_')
    chrom = details[1]
    region_idx = details[3]
    nbins = details[5]
    return chrom, region_idx, nbins

def interpret_gen_dcd(dcd_file):
    f = '.'.join(dcd_file.split('/')[-1].split('.')[:-1])
    details = f.split('_')
    region_idx = details[1]
    chrom = details[5]
    return region_idx, chrom

# Center the tan conformations
tan_dcd_files = [str(f) for f in Path('./Tan/').rglob('*.dcd')]
gen_dcd_files = [str(f) for f in Path('./origami_64_no_embed_reduction/dcd_files/aligned/').rglob('*.dcd')]
completed = []
os.makedirs('./tmp/')
for dcd_file in tqdm(tan_dcd_files,desc='Outer Loop'):
    psf_file = dcd_file.replace('.dcd','.psf')
    t = md.load(dcd_file,top=psf_file)
    #t.center_coordinates()
    for i in range(len(t)):
        t.xyz[i,:,0]-= t.xyz[i,:,0].mean()
        t.xyz[i,:,1]-= t.xyz[i,:,1].mean()
        t.xyz[i,:,2]-= t.xyz[i,:,2].mean()

    t.save_dcd(dcd_file)

    chrom, region_idx, nbins = interpret_tan_dcd(dcd_file)

    for dcd_file1 in tqdm(gen_dcd_files,desc='Inner Loop',leave=False):
        region_idx1,chrom1 = interpret_gen_dcd(dcd_file1)
        if chrom1 != chrom or region_idx1 != region_idx:
            continue
        psf_file1 = dcd_file1.replace('.dcd','.psf')
        gen_sample = md.load(dcd_file1,top=psf_file1)
        temp_files = []
        for frame in range(len(gen_sample)):
            temp = gen_sample[frame]
            temp.superpose(t,frame=frame)
            temp.save_dcd(f'./tmp/{frame}.dcd')
            temp_files.append(f'./tmp/{frame}.dcd')
        gen_sample = md.load(temp_files,top=psf_file1)
        gen_sample.save_dcd(dcd_file1)
        for file in temp_files:
            os.remove(file)

        completed.append(dcd_file1)

    gen_dcd_files = [f for f in gen_dcd_files if f not in completed]
os.rmdir('./tmp/')

print(f"Skipped Files: {gen_dcd_files}")


