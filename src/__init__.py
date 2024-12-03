from . import model, Conformations
from ._scripts import  *
from .Conformations import _conformations_init as Conformations
from .model.ChromoGen import ChromoGen


########################################
# Ways to initialize ChromoGen
def from_file(filepath,**init_kwargs):
    return ChromoGen.from_file(filepath)

def from_files(epcot_filepath,diffuser_filepath,**init_kwargs):
    return ChromoGen.from_files(epcot_filepath,diffuser_filepath,**init_kwargs)

