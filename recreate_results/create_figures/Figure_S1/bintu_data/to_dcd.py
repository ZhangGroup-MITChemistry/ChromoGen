import sys
sys.path.insert(0,'../../code/data_utils/SampleClass/')
from Coordinates import Coordinates
from pathlib import Path

for f in Path('.').glob('*.pt'):
    ff = str(f)
    c=Coordinates(ff)
    c._values/=100
    c._values-= c.values.mean(1,keepdim=True)
    c.trajectory.save_dcd(ff.replace('.pt','.dcd'))

