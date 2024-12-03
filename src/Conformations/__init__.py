from ._Coordinates import Coordinates
from ._Distances import Distances
from ._Trajectory import Trajectory
from ._OrigamiTransform import OrigamiTransform

# Generic conformations loader. 
def _conformations_init(x,representation=None,**kwargs):
    from pathlib import Path
    import torch
    if representation is not None:
        assert (
            isinstance(representation,str)
        ), (
            'When specified, representation (second positional argument) '
            'must be "coordinates", "distances", or "trajectory".'
        )
    
    if isinstance(x,(str,Path)):
        f = Path(x)
        assert f.exists(), f'The provided filepath, {x}, does not exist.'
        assert (sfx:=f.suffix), \
        f'Expected the provided filepath, {x}, to have a file extension.'
        match f.suffix:
            case '.pt':
                x = torch.load(f,map_location='cpu')

            case '.dcd':
                num_beads = kwargs.get('num_beads')
                topology_file = kwargs.get('topology_file')
                topology = kwargs.get('topology')
                assert (
                    kwargs.get('num_beads') is not None or 
                    kwargs.get('topology_file') is not None or 
                    kwargs.get('topology') is not None
                ), (
                    'When passing a .dcd file, you must also specify num_beads, '
                    'provide the path to a topology file, or provide an mdtraj.Topology '
                    'instance so that the .dcd file can be properly parsed'
                )
                trajectory = Trajectory.from_dcd(str(x),**kwargs)
                if representation is not None:
                    return getattr(trajectory,representation)
                return trajectory

            case _:
                raise Exception(
                    f'For now, only files with ".pt" and ".dcd" extensions can be used.'
                )
                
    from ._ConformationsABC import ConformationsABC
    if isinstance(x,ConformationsABC):
        return x

    if not isinstance(x,torch.Tensor):
        try:
            x = torch.tensor(x)
        except:
            raise Exception(
                'x must either be the filepath of a .pt file '
                '(string or pathlib.Path object), a torch.Tensor, '
                'or something that can be converted to a torch.Tensor. '
                f'Input of type {type(x).__name__} cannot be.'
            )

    initial_shape = tuple(x.shape)
    while x.ndim < 3:
        x = x.unsqueeze(0)

    if representation is None:
        # Infer the input type. Always pass Coordinates rather than Trajectory when relevant
        if x.shape[-1] != 3:
            assert x.shape[-2]==x.shape[-1], (
                f'Invalid input shape {initial_shape}. All coordinates objects '
                'should should have size 3 in the last dimension (for XYZ coordinates), '
                'while all distance maps should have the same size in the last two dimensions. '
            )
            return Distances(x,**kwargs)
        if x.shape[-2] == 3:
            # Ambiguous... size 3 in both of the last two dimensions. 
            import warnings
            if torch.allclose(x,x.mT) and (x>=0).all():
                # LIKELY distances
                if torch.tensor(x.shape[:-2]).prod() > 1:
                    assumed_type = 'distance maps'
                else:
                    assumed_type = 'a distance map'
                x = Distances(x,**kwargs)
                
            else:
                # LIKELY coordinates
                assumed_type = 'coordinates'
                kwargs['drop_invalid_conformations'] = kwargs.get('drop_invalid_conformations',False)
                x = Coordinates(x,**kwargs)
                
            warnings.warn(
                f'Input with shape {initial_shape} is ambiguous. ''\n'
                f"I'm assuming the input represents {assumed_type}. "
                'If this is incorrect, please specify what this object is by passing '
                '`representation="distances"`, `representation="coordinates"`, or '
                '`representation="trajectory"`.'
            )
            return x
        kwargs['drop_invalid_conformations'] = kwargs.get('drop_invalid_conformations',False)
        return Coordinates(x,**kwargs)


    # The user specified the input type
    match representation.lower():
        case 'distances':
            return Distances(x,**kwargs)
        case 'coordinates':
            return Coordinates(x,**kwargs)
        case 'trajectory':
            return Trajectory(x,**kwargs)
        case _:
            raise Exception(
                f'The provided representation, {representation}, is invalid. '
                'Choose from "distances" if it is stacked distance maps, "coordinates" if '
                'they are coordinates that you wish to place in a Coordinates object, or '
                '"trajectory" if they are coordinates you\'d like to place in a Trajectory '
                'object.'
            )


##########
# Load functions need to be placed here


##########
# 
