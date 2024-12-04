import sys
sys.path.insert(0,'../../code/data_utils/SampleClass/')
from Coordinates import Coordinates
from pathlib import Path

for f in Path('.').glob('*.pt'):
    ff = str(f)
    Coordinates(ff).trajectory.save_dcd(ff.replace('.pt','.dcd'))

