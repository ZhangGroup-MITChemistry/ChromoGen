'''
TODO: 
1. Fix num_beads, etc., functions for when in folded state.
2. Add functionality to fold more than once. 
'''

import torch
from ConformationsABC import ConformationsABC
from OrigamiTransform import OrigamiTransform
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Initialization
def format_dists(dists):
    '''
    Also handle filepaths, trajectories
    '''
    t = type(dists)
    if issubclass(t,ConformationsABC):
        return dists.distances.values
    if t != torch.Tensor:
        try:
            dists = torch.Tensor(dists)
        except:
            raise Exception('Data must be convertible to torch.Tensor object, '+\
                            f'but the provided {t} object cannot be!')

    dimension_msg = f'Input shape {dists.shape} is not valid!\n'
    dimension_msg+= 'Dimensions should be one of the following:\n'
    dimension_msg+= '\t(num_atoms,num_atoms): Dimensions of a distance map; or\n'
    dimension_msg+= '\t(batch_dim1,...,batch_dimN,num_atoms,num_atoms).\n'
    assert dists.ndim > 1, dimension_msg
    assert dists.shape[-2]==dists.shape[-1], dimension_msg
    if dists.ndim == 2:
        dists = dists.unsqueeze(0)
    
    return dists

########################################################
# Normalizer functions
def is_normalized(dist_maps):
    return (dist_maps <= 1).all()
    
class Normalizer:

    mean_dists = [1.2799496560668104,2.066471439135653,2.6251417027053523,3.03980959577955,3.3656535619385846,3.6348899477810908,3.8663401053698108,
                  4.0711327455976125,4.255789951838822,4.424580681174291,4.580639985355445,4.726371221595456,4.863209117513583,4.9926025587095175
                  ,5.115564473243971,5.232801498125753,5.3450081465615,5.4529965431292045,5.55697346288983,5.657034250051045,5.753686253992502,
                  5.847153536806088,5.938102731286845,6.026579187516907,6.112889355868144,6.197175037521096,6.279537560375267,6.360051654118017,
                  6.438946558476546,6.516253145856243,6.592017512386089,6.6663998429450215,6.739245586935708,6.81087962168326,6.881503105772223,
                  6.951071310114832,7.019638593298146,7.087266864233727,7.153829479360961,7.219365248758809,7.283760065722475,7.347189735746156,
                  7.409989108982501,7.472207986253817,7.533460393327083,7.593653876007603,7.653085351528603,7.711970207997877,7.770079963816016,
                  7.827395596498657,7.883933791552487,7.939994877298068,7.995270353852504,8.05034493984945,8.10527840088238,8.159675434917968,
                  8.213457480153197,8.267120730882352,8.320208738310315,8.373193483057584,8.42606991106077,8.4789946384002,8.530419775572579,
                  8.580017178475794,8.59289264678955,8.639609336853027,8.685870170593262,8.731688499450684,8.77707290649414,8.822037696838379,
                  8.86658763885498,8.910737037658691,8.954490661621094,8.997862815856934,9.040855407714844,9.083480834960938,9.12574577331543,
                  9.167658805847168,9.209226608276367,9.250455856323242,9.291356086730957,9.331929206848145,9.372184753417969,9.412129402160645,
                  9.45176887512207,9.491108894348145,9.530155181884766,9.568913459777832,9.607388496398926,9.645583152770996,9.683507919311523,
                  9.721162796020508,9.758557319641113,9.79569149017334,9.832573890686035,9.869203567504883,9.905590057373047,9.941734313964844,
                  9.977643013000488,10.01331901550293,10.048763275146484,10.083985328674316,10.118979454040527,10.153761863708496,10.188324928283691,
                  10.222677230834961,10.256821632385254,10.29076099395752,10.32449722290039,10.358037948608398,10.391379356384277,10.424528121948242,
                  10.457486152648926,10.49025821685791,10.522843360900879,10.555249214172363,10.587472915649414,10.619521141052246,10.65139389038086,
                  10.683093070983887,10.714624404907227,10.745986938476562,10.777183532714844,10.808218002319336,10.839093208312988,10.869808197021484,
                  10.900364875793457,10.930767059326172,10.961016654968262,10.991117477416992,11.021065711975098,11.050869941711426,11.080527305603027,
                  11.110041618347168,11.13941478729248,11.16865062713623,11.197744369506836,11.226698875427246,11.255524635314941,11.284213066101074,
                  11.312769889831543,11.34119987487793,11.369498252868652,11.39766788482666,11.425714492797852,11.453636169433594,11.481433868408203,
                  11.509109497070312,11.536664009094238,11.564101219177246,11.591423034667969,11.618623733520508,11.645711898803711,11.672684669494629,
                  11.699545860290527,11.726295471191406,11.752934455871582,11.779463768005371,11.805886268615723,11.832200050354004,11.858409881591797,
                  11.884514808654785,11.910512924194336,11.936410903930664,11.962207794189453,11.987902641296387,12.013501167297363,12.0389986038208,
                  12.064397811889648,12.089701652526855,12.114908218383789,12.140018463134766,12.1650390625,12.189961433410645,12.214794158935547,
                  12.23953628540039,12.264187812805176,12.288749694824219,12.313222885131836,12.337605476379395,12.36190414428711,12.386114120483398,
                  12.410243034362793,12.43427848815918,12.458237648010254,12.482109069824219,12.505899429321289,12.529607772827148,12.553236961364746,
                  12.576783180236816,12.600248336791992,12.623636245727539,12.646946907043457,12.670178413391113,12.693333625793457,12.716413497924805,
                  12.739415168762207,12.762340545654297,12.785194396972656,12.80797290802002,12.830679893493652,12.853310585021973,12.875870704650879,
                  12.898361206054688,12.920778274536133,12.943126678466797,12.96540355682373,12.9876127243042,13.00975227355957,13.031825065612793,
                  13.053829193115234,13.075765609741211,13.097634315490723,13.119439125061035,13.141176223754883,13.162850379943848,13.184456825256348,
                  13.206003189086914,13.227483749389648,13.2489013671875,13.270258903503418,13.29155158996582,13.312780380249023,13.333951950073242,
                  13.355060577392578,13.376104354858398,13.397096633911133,13.4180269241333,13.438892364501953,13.45970630645752,13.480460166931152,
                  13.501152992248535,13.521788597106934,13.542373657226562,13.562893867492676,13.583362579345703,13.603772163391113,13.624131202697754,
                  13.644433975219727,13.664680480957031,13.684871673583984,13.705009460449219,13.725092887878418,13.74512767791748,13.765104293823242,
                  13.78503131866455,13.804903030395508,13.824724197387695,13.844497680664062,13.864214897155762,13.883885383605957,13.903502464294434,
                  13.92307186126709,13.942591667175293,13.962058067321777,13.981478691101074]

    mean_sq_dists = [1.9060106423758427, 4.869646163456119, 7.884701976607313, 10.682196301686051, 13.238809660058902, 15.592728266675003,
                     17.787220404253823, 19.857768060067617, 21.82647294101988, 23.70786961305501, 25.514243351153375, 27.25612074112904, 
                     28.94114606154483, 30.576070844415828, 32.16724545648308, 33.718546804028215, 35.23345503534113, 36.71634013539073, 
                     38.1697755668579,39.59288570001414, 40.990580756341814, 42.36289260846268, 43.71531881665955, 45.04990396545458, 
                     46.36814345653637, 47.67028859115439, 48.957569507379525, 50.23151157429649, 51.49388271186868, 52.74379928293248, 
                     53.980442318023044, 55.20493655594276, 56.41699653262215, 57.62066320549066, 58.81841754419876, 60.008977053786055,
                     61.18971581522482, 62.36199785747272, 63.525135940602276, 64.68007295062995, 65.82533243309831, 66.96166559515537, 
                     68.09338072777075, 69.22281328579808, 70.34303778838253, 71.45562364538031, 72.56637378663568, 73.67720379269474, 
                     74.7843851310144, 75.88459755716947, 76.97556446582357, 78.0613669972507, 79.13800065061407, 80.21583400279823, 
                     81.29045307355094, 82.35950952831347, 83.42225009582933, 84.48983513436808, 85.55059770072324, 86.61863567068244, 
                     87.696378227443, 88.78071074050564, 89.85646857941398, 90.90884431010834, 91.64350891113281, 92.64923095703125, 
                     93.65061950683594, 94.64773559570312, 95.64068603515625, 96.6295394897461, 97.61439514160156, 98.59526824951172, 
                     99.57230377197266, 100.54552459716797, 101.5150146484375, 102.48075103759766, 103.44294738769531, 104.4015121459961, 
                     105.35670471191406, 106.3083724975586, 107.25662231445312, 108.20156860351562, 109.14328002929688, 110.08170318603516,
                     111.01698303222656, 111.94914245605469, 112.87815856933594, 113.80413818359375, 114.72720336914062, 115.64720916748047,
                     116.56441497802734, 117.47870635986328, 118.39016723632812, 119.29879760742188, 120.2047119140625, 121.10792541503906, 
                     122.00848388671875, 122.90635681152344, 123.80158996582031, 124.6943130493164, 125.5843734741211, 126.47203826904297, 
                     127.35723876953125, 128.23997497558594, 129.1202392578125, 129.99813842773438, 130.87367248535156, 131.74691772460938, 
                     132.6178436279297, 133.48654174804688, 134.35289001464844, 135.21707153320312, 136.0791015625, 136.93882751464844, 
                     137.7965545654297, 138.6520538330078, 139.5054931640625, 140.3568878173828, 141.2061767578125, 142.05343627929688, 
                     142.8987274169922, 143.74200439453125, 144.58328247070312, 145.42262268066406, 146.2600860595703, 147.09559631347656, 
                     147.92921447753906, 148.7609405517578, 149.5909423828125, 150.41896057128906, 151.24517822265625, 152.06967163085938, 
                     152.89236450195312, 153.7132568359375, 154.53244018554688, 155.34994506835938, 156.1656951904297, 156.9796905517578, 
                     157.7921142578125, 158.60275268554688, 159.41175842285156, 160.2192840576172, 161.02505493164062, 161.82919311523438, 
                     162.63180541992188, 163.43284606933594, 164.23231506347656, 165.0302734375, 165.82664489746094, 166.6214599609375, 
                     167.41485595703125,168.2066650390625,168.9970245361328,169.7859649658203,170.57334899902344,171.35931396484375,
                     172.1439666748047, 172.927001953125, 173.708740234375, 174.48902893066406, 175.26800537109375, 176.0455322265625, 
                     176.82176208496094, 177.59652709960938, 178.3699951171875, 179.14210510253906, 179.9130096435547, 180.68247985839844, 
                     181.45065307617188, 182.21759033203125, 182.983154296875, 183.74758911132812, 184.5105743408203, 185.27244567871094, 
                     186.032958984375, 186.79229736328125, 187.5504150390625, 188.30735778808594, 189.06300354003906, 189.81741333007812, 
                     190.57073974609375, 191.3228759765625, 192.0738525390625, 192.82354736328125, 193.57212829589844, 194.31956481933594, 
                     195.06581115722656, 195.81094360351562, 196.5550537109375, 197.29786682128906, 198.03956604003906, 198.78024291992188,
                     199.51995849609375, 200.25848388671875, 200.99575805664062, 201.73220825195312, 202.46734619140625, 203.20159912109375,
                     203.93470764160156, 204.66676330566406, 205.3978271484375, 206.12777709960938, 206.85690307617188, 207.58473205566406,
                     208.31170654296875, 209.03765869140625, 209.7625732421875, 210.48641967773438, 211.2093963623047, 211.93136596679688, 
                     212.65231323242188, 213.37242126464844, 214.09144592285156, 214.80935668945312, 215.52659606933594, 216.24266052246094,
                     216.95777893066406, 217.6721649169922, 218.38552856445312, 219.09786987304688, 219.8094024658203, 220.5200958251953, 
                     221.22967529296875, 221.93838500976562, 222.64630126953125, 223.35316467285156, 224.059326171875, 224.76449584960938, 
                     225.46881103515625, 226.17222595214844, 226.8747100830078, 227.57638549804688, 228.2772216796875, 228.97720336914062, 
                     229.67630004882812, 230.37449645996094, 231.07188415527344, 231.7686004638672, 232.46435546875, 233.1592559814453, 
                     233.85340881347656, 234.54666137695312, 235.23912048339844, 235.9307861328125, 236.6216278076172, 237.31161499023438, 
                     238.00074768066406, 238.68927001953125, 239.3768768310547, 240.06370544433594, 240.74986267089844, 241.43519592285156, 
                     242.11968994140625, 242.80345153808594, 243.48634338378906, 244.1687469482422]
    
    def __init__(
        self,
        mean_dist_fp=None,
        mean_sq_dist_fp=None
    ):

        if mean_dist_fp is None:
            mean_dist = torch.tensor(self.mean_dists,dtype=torch.double)
        else:
            mean_dist = torch.load(mean_dist_fp).flatten().double()
            
        if mean_sq_dist_fp is None:
            mean_square_dist = torch.tensor(self.mean_sq_dists,dtype=torch.double)
        else:
            mean_square_dist = torch.load(mean_sq_dist_fp).flatten().double()
            
        self.dist_std = (mean_square_dist - mean_dist**2).sqrt()
        self.inv_beta = torch.sqrt( 2*mean_square_dist/3 )
        self.inv_beta_sigmoid = torch.sigmoid( -self.inv_beta/self.dist_std )
        self.complement_inv_beta_sigmoid = 1 - self.inv_beta_sigmoid
    
    def to(self,*args,**kwargs):
        '''
        Primarily for moving to the device of whichever object is being worked on
        '''
        self.dist_std = self.dist_std.to(*args,**kwargs)
        self.inv_beta = self.inv_beta.to(*args,**kwargs)
        self.inv_beta_sigmoid = self.inv_beta_sigmoid.to(*args,**kwargs)
        self.complement_inv_beta_sigmoid = self.complement_inv_beta_sigmoid.to(*args,**kwargs)

    def __prep_for_comp(self,dists):
        n = dists.shape[-1]
        assert dists.ndim > 1 and dists.shape[-2] == n, \
        'Expected square distance matrices in the final two dimensions, '+\
        'but received object with shape {dists.shape}'

        # Always perform computations with high precision to avoid numerical issues with 
        # the logit function.
        dists1 = dists.double()

        # Also use the GPU for performance... transferring data back and forth may actually
        # slow things down on small batches, but that's a problem to think about later. 
        if torch.cuda.is_available():
            dists1 = dists1.cuda()

        # Diagonal values should always equal 0, but are occasionally non-zero due to 
        # numerical imprecision with the U-Net. Fix that here. 
        dists1[...,range(n),range(n)] = 0

        # If the matrices are symmetric, just perform the computation once
        # and broadcast the results from the upper triangle to the lower triangle
        broadcast = torch.allclose(dists1,dists1.transpose(-2,-1))

        # Indices to grab upper/lower triangle
        i,j = torch.triu_indices(n,n,1)
        
        # Separation index. Subtract 1 since the mean distances/square distances
        # objects were saved without self interaction (so index 0 corresponds to 
        # objects with 1 bond between them)
        sep = j - i - 1

        # Move this object's tensors to the same device as the distances object
        self.to(dists1.device)
        
        return dists1, broadcast, i, j, sep

    def __normalize(self,dists,sep):
        '''
        Normalize distances. This takes data ALREADY
        indexed with torch.triu_indices indexing, as in self.unnormalize()
        '''

        dists-= self.inv_beta[sep].expand(*dists.shape[:-1],-1)
        dists/= self.dist_std[sep].expand(*dists.shape[:-1],-1)
        dists.sigmoid_()
        dists-= self.inv_beta_sigmoid[sep].expand(*dists.shape[:-1],-1)
        dists/= self.complement_inv_beta_sigmoid[sep].expand(*dists.shape[:-1],-1)
        
        return dists
    
    def normalize(self,dists):

        if is_normalized(dists):
            return dists

        # This fucntion moves data to the GPU (if available) and uses double precision,
        # but we want to return an object with the original dtype and on the original device, 
        # so track them
        return_dtype = dists.dtype
        return_device = dists.device

        # Placed some data prep code into another function since it's used
        # exactly the same in both normalize and unnormalize functions
        dists, broadcast, i, j, sep = self.__prep_for_comp(dists)

        # Normalize the values
        dists[...,i,j] = self.__normalize(dists[...,i,j],sep)
        if broadcast:
            dists[...,j,i] = dists[...,i,j]
        else:
            dists[...,j,i] = self.__normalize(dists[...,j,i],sep)

        return dists.to(dtype=return_dtype,device=return_device)
        
    def __unnormalize(self,dists,sep):
        '''
        Unnormalize distances. This takes data ALREADY
        indexed with torch.triu_indices indexing, as in self.unnormalize()
        '''
        dists*= self.complement_inv_beta_sigmoid[sep].expand(*dists.shape[:-1],-1)
        dists+= self.inv_beta_sigmoid[sep].expand(*dists.shape[:-1],-1)
        dists.logit_()
        dists*= self.dist_std[sep].expand(*dists.shape[:-1],-1)
        dists+= self.inv_beta[sep].expand(*dists.shape[:-1],-1)

        # The logit function causes values at 1 to become infinite-valued, an
        # artifact handled during the conversion to coordinates. 
        # However, the -infinity values (which actually end up near 0 due to the 
        # other conversions performed) are easier to handle here
        dists[dists<1e-8] = .01
        return dists
    
    def unnormalize(self,dists):

        if not is_normalized(dists):
            return dists

        # Placed some data prep code into another function since it's used
        # exactly the same in both normalize and unnormalize functions
        dists1, broadcast, i, j, sep = self.__prep_for_comp(dists)

        # Unnormalize the values
        dists[...,i,j] = self.__unnormalize(dists1[...,i,j],sep).to(dtype=dists.dtype,device=dists.device)
        if broadcast:
            dists[...,j,i] = dists[...,i,j]
        else:
            dists[...,j,i] = self.__unnormalize(dists1[...,j,i],sep).to(dtype=dists.dtype,device=dists.device)

        return dists

########################################################
# Analytically convert distance maps into coordinates
def dists_to_coords(dists,device=None,num_dimensions=3,num_attempts=None,error_threshold=1e-4):

    # Use high-precision values throughout calculation, but return same dtype as provided
    # Same for device
    return_dtype = dists.dtype
    return_device = dists.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dists = dists.double().to(device)

    # Initialize the object to hold coordinates
    coords = torch.empty_like(dists[...,:num_dimensions])

    # Choose beads to set at the origin when computing conformations, which 
    # occasionally fails due to numerical issues with the distance maps generated
    # by the diffusion model 
    n = dists.shape[-1]
    num_attempts = n if num_attempts is None else min(num_attempts,n)
    origins = [i*dists.shape[-1]//num_attempts for i in range(num_attempts)]

    # We'll ignore infinite values in the target distance maps, an artifact discussed further in correct_coords()
    inf_mask = torch.where(dists.isinf())

    # Convert distances to coordinates
    # Keep track of the set of coordinates that best agrees with the target distance map
    # to reduce the amount of optimization needed when correcting coordinates
    zero = torch.tensor(0.).to(dtype=dists.dtype,device=dists.device)
    for i,o in enumerate(origins):
        # Fill coords with zeros so the compute_new_dimension function operates as desired
        coords.fill_(0)
        
        # Keep track of reference indices as compute_new_dimension is called repeatedly
        reference_indices = []

        # Compute the new dimensions
        for _ in range(num_dimensions):
            compute_new_dimension(coords,dists,reference_indices,initial_index=o)

        # If any of these freshly-computed coordinates are closer to the desired result than
        # the best already encountered, update the best_coords/best_errors object
        errors = (torch.cdist(coords,coords)-dists).square_()
        errors[inf_mask] = 0 # Send errors at interactions with infinite distance in the target map to 0
        errors[errors.isnan()] = torch.inf # If NaN values appear, treat that conformation as infinitely wrong
        errors = errors.mean((-2,-1))
        if i == 0:
            best_coords = coords.clone()
            best_errors = errors.clone()
        else:
            mask = errors < best_errors
            if mask.any():
                best_coords[mask] = coords[mask]
                best_errors[mask] = errors[mask]
        if (best_errors < error_threshold).all():
            break
    return best_coords.to(dtype=return_dtype,device=return_device)
    
    '''
    # Convert distances to coordinates
    mask = torch.ones(*coords.shape[:-2],dtype=bool,device=coords.device)
    for o in origins:
        # The coordinates that still need to be computed due to failures in prior loop iterations
        temp_coords = coords[mask]
        temp_coords.fill_(0)
        temp_dists = dists[mask]
        
        # Keep track of reference indices
        reference_indices = []

        for _ in range(num_dimensions):
            compute_new_dimension(temp_coords,temp_dists,reference_indices,initial_index=o)
        coords[mask] = temp_coords
        mask = ~coords.isfinite().all(-1).all(-1)
        if not mask.any():
            break
    return coords.to(dtype=return_dtype,device=return_device)
    '''
    
def x_dot_y(x_norm,y_norm,x_minus_y_norm):
    # From known vector norms
    y_norm = y_norm.expand_as(x_norm)
    return (x_norm**2 + y_norm**2 - x_minus_y_norm**2) / 2

def select_new_indices(dist_from_origin,coords):
    dist_not_accounted_for = (dist_from_origin.square() - coords.square().sum(-1,keepdim=True))

    # Numerical precision occasionally causes small negative values to appear... to avoid NaN results, set those to 0!
    dist_not_accounted_for[dist_not_accounted_for<0] = 0
    dist_not_accounted_for.sqrt_()
    # Select maximum value to minimize numerical errors with division on this value later
    #return dist_not_accounted_for.max(-2,keepdim=True)
    return dist_not_accounted_for.median(-2,keepdim=True)
    
def compute_new_dimension(coords,dists,reference_indices,initial_index=None):
    # Everything operates in-place
    if len(reference_indices) == 0:
        # Set a bead at the origin
        if initial_index is None: # Choose the central bead if none specified. 
            idx = torch.tensor(dists.shape[-1]//2).expand_as(dists[...,:1]).to(dists.device)
        else: 
            idx = torch.tensor(initial_index).expand_as(dists[...,:1]).to(dists.device)
        reference_indices.append(idx)

    ri = reference_indices # ease of notation
    x_norm = dists.gather(-1,ri[0]) # Distance from origin
    
    coord_value, idx = select_new_indices(x_norm,coords)
    idx = idx.expand_as(dists[...,:1])
    dim = len(ri) - 1
    y_norm = x_norm.gather(-2,idx) # Distance from origin for new reference bead
    x_minus_y_norm = dists.gather(-1,idx) # Distance between all beads and the new reference bead
    
    new_coord_values = x_dot_y(x_norm,y_norm,x_minus_y_norm)
    if dim > 0:
        selected_coord_prior_values = coords[...,:dim].gather(-2,idx.expand_as(coords[...,:dim]))
        new_coord_values-= (selected_coord_prior_values * coords[...,:dim]).sum(-1,keepdim=True) # Dot product
    new_coord_values/= coord_value
    coords[...,dim:dim+1] = new_coord_values
    
    ri.append(idx)

########################################################
# Optimize coordinates to a reference distance map
'''
def smooth_transition_loss(
    output,
    target,
    r_c=1.0, # Transition distance from x**2 -> x**(long_scale)
    long_scale=1
):
    '#''
    Reduces to smooth L1 loss if  long_scale == 1
    '#''
    # Scale to ensure the two functions have the same slope at r_c
    m = 2 / long_scale
    # Shift to ensure the two functions have the same value at r_c
    b = 1 - m
    
    loss = 0
    difference = (output - target).abs() / r_c
    mask = difference < 1
    if mask.any():
        #loss = loss + difference[mask].square().sum()
        loss = loss + torch.nansum(difference[mask].square())
    mask = ~mask
    if mask.any():
        #loss = loss + (m*difference[mask]**long_scale + b).sum()
        loss = loss + torch.nansum( m*difference[mask]**long_scale + b )

    return loss

def loss_fcn(coords,target_dists,r_c=1.0,long_scale=1/8,proportional=False):
    dists = torch.cdist(coords,coords)
    i,j = torch.triu_indices(dists.shape[-1],dists.shape[-1],1)
    output,target = dists[...,i,j],target_dists[...,i,j]
    if proportional:
        # Adding .0001 for numerical stability where VERY small values appear
        return smooth_transition_loss((output+.0001)/(target+.0001),torch.ones_like(output))
    else:
        return smooth_transition_loss(output,target)
'''

def smooth_transition_loss(
    output,
    target,
    r_c=1.0, # Transition distance from x**2 -> x**(long_scale)
    long_scale=1
):
    '''
    Reduces to smooth L1 loss if  long_scale == 1
    '''
    # Scale to ensure the two functions have the same slope at r_c
    m = 2 / long_scale
    # Shift to ensure the two functions have the same value at r_c
    b = 1 - m
    
    # 
    difference = torch.fmax(
        (output - target).abs() / r_c,
        ((output+1e-6)/(target+1e-6)-1).abs()
    )

    # Replace infinite values with NaN's, which are ignored below. 
    # Infinite values are an artifact of applying the logit function to values of 
    # 1 from the normalized distance maps produced by the U-Net. 
    # This causes these points to have no impact on the optimization, which is valid 
    # since we don't know the true distance intended at these points. 
    difference[difference.isinf()] = torch.nan

    # Remove explicit outliers, which are likely a result of numerical imprecision
    difference[difference > 25*torch.nanmean(difference)] = torch.nan

    loss = 0
    mask = difference < 1
    if mask.any(): # SSE for errors < 1
        loss = loss + torch.nansum( difference[mask].square() )
    mask = ~mask
    if mask.any(): # Decaying slope for errors > 1, minimizing the impact of outliers
        loss = loss + torch.nansum( m*difference[mask]**long_scale + b )

    return loss

def loss_fcn(coords,target_dists,r_c=1.0,long_scale=1/8,near_neighbors_scales=[10,8,6,4,2]):
    dists = torch.cdist(coords,coords)
    i,j = torch.triu_indices(dists.shape[-1],dists.shape[-1],1)
    output,target = dists[...,i,j],target_dists[...,i,j]

    bond_dists = j - i
    for k,scale in enumerate(near_neighbors_scales):
        if scale != 1:
            # Provide additional weight to distances between sequentially proximal interactions, which  
            # otherwise end up being a worse match to the target than the larger-distance interactions
            idx = torch.where(bond_dists == k+1)[0]
            output[...,idx] = output[...,idx] * scale
            target[...,idx] = target[...,idx] * scale

    return smooth_transition_loss(output,target)

def correct_coords(
    coords,
    target_dists,
    *,
    min_loss_change=1e-6,
    num_iterations=1_000,
    lr=.1,
    lr_decay=0,
    weight_decay=0,
    r_c=1.0,
    long_scale=1/8
):

    return_dtype = coords.dtype
    return_device = coords.device
    coords = coords.double()
    target_dists = target_dists.double()
    if torch.cuda.is_available():
        coords = coords.cuda()
        target_dists = target_dists.cuda()

    coords.requires_grad_(True)
    
    optimizer = torch.optim.Adagrad(
        [coords],
        lr=lr,
        lr_decay=lr_decay,
        weight_decay=weight_decay
    )
    
    prior_loss = loss_fcn(coords,target_dists).detach()

    exit_loss_condition = min_loss_change * target_dists.numel()
    with tqdm(initial = 0, total = num_iterations, leave=None) as pbar:
        for i in range(num_iterations):
            optimizer.zero_grad()
            loss = loss_fcn(coords,target_dists,r_c=r_c,long_scale=long_scale)
            loss.backward()
            optimizer.step()

            loss = loss.detach()
            if i > 0 and abs(prior_loss - loss) < exit_loss_condition:
                print(f'Change in loss ({abs(loss - prior_loss)/target_dists.numel()}) is less than tolerance ({min_loss_change}).'+\
                      '\n'+f'Final error: {loss/target_dists.numel()}')
                break
            prior_loss = loss
            
            pbar.set_description(f'Correcting Distance Maps. loss per distance value: {loss/target_dists.numel():.4f}')
            pbar.update(1)
            #if (i+1)%100 == 0 or i==0:
            #    pbar.set_description(f'Correcting Distance Maps. loss per distance value: {loss/target_dists.numel():.4f}')
            #    if i > 0:
            #        pbar.update(100)
    
    return coords.detach().to(dtype=return_dtype,device=return_device)

def smooth_transition_loss_by_sample(
    output,
    target,
    r_c=1.0, # Transition distance from x**2 -> x**(long_scale)
    long_scale=1/8,
    use_gpu=True,
    high_precision=True
):
    '''
    Reduces to smooth L1 loss if  long_scale == 1.
    
    Rather than summing over ALL data, sum over the final two 
    dimensions (corresponding to individual distance maps). 
    '''
    # Scale to ensure the two functions have the same slope at r_c
    m = 2 / long_scale
    # Shift to ensure the two functions have the same value at r_c
    b = 1 - m
    
    return_device = output.device
    return_dtype = output.dtype
    if use_gpu and torch.cuda.is_available():
        output = output.cuda()
        target = target.cuda()
    if high_precision:
        output = output.double()
        target = target.double()

    losses = (output - target).abs_()
    del output, target
    losses/= r_c
    
    if losses.is_cuda:
        '''
        This is slower than using masking, but it avoid memory issues associated
        with mask indexing (torch turns bool masks into int64 indexing arrays) while
        remaining faster than some alternative low-memory options I tried. 
        '''
    
        #losses = torch.where(
        #    losses < 1,
        #    losses.square(),
        #    m*losses.pow(long_scale)+b,
        #    out=losses
        #)
        torch.where(
            losses < 1,
            losses.square(),
            m*losses.pow(long_scale)+b,
            out=losses
        )
        
    else:
        '''
        Assume that these memory issues don't arise on the CPU
        '''
        mask = losses < 1
        if mask.any():
            losses[mask] = losses[mask]**2
        mask^= True
        if mask.any():
            losses[mask] = m*losses[mask]**long_scale + b
        del mask
    

    return losses.sum((-1,-2)).to(dtype=return_dtype,device=return_device)

########################################################
# Visualization
from mpl_toolkits.axes_grid1 import make_axes_locatable
def plot_dist_map(
    dists,
    fig=None,
    ax=None,
    cmap='RdBu',
    xticks=[],
    xticklabels=None,
    yticks=[],
    yticklabels=None,
    xlabel='Genomic index',
    ylabel='Genomic index',
    cbar_label='Distance',
    cbar_orientation=None,
    cbar_ticks=None,
    **kwargs
):
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    im = ax.matshow(dists.cpu().numpy(),cmap=cmap,**kwargs)

    # Ensure colorbar is the same size
    divider = make_axes_locatable(ax)
    if cbar_orientation == 'horizontal':
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
    else:
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.set_xlabel(xlabel)
        
    if cbar_orientation is not None:
        cbar = fig.colorbar(im,cax=cax,label=cbar_label,orientation=cbar_orientation,ticks=cbar_ticks)
    else:
        cbar = fig.colorbar(im,cax=cax,label=cbar_label,ticks=cbar_ticks)
    #cbar = fig.colorbar(im,label=cbar_label)
    
    ax.set_xticks(xticks,labels=xticklabels)
    ax.set_yticks(yticks,labels=yticklabels)

    ax.set_ylabel(ylabel)

    return fig, ax, im, cbar
    
    

########################################################
# Main class
class Distances(ConformationsABC):

    def __init__(
        self,
        input,
        description = 'Distance'
    ):
        self.__dist_maps = format_dists(input)
        self.__description = description
        self.__origami_transform = OrigamiTransform()
        self.__is_folded = input.ndim > 3 and input.shape[-3] == 2

    ########################################################
    # Needed for much of the functionality in Sample superclass
    @property
    def _values(self):
        return self.__dist_maps

    @_values.setter
    def _values(self,c):
        self.__dist_maps = c

    ########################################################
    # Basic data manipulation
    def flatten(self):
        return Distances(self.values.flatten(0,-3),self.__description)
    
    ########################################################
    # Distance Statistics
    @property
    def mean(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return Distances(self.flatten().values.to(device).mean(0).to(self.device),'Mean Distance')

    @property
    def median(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return Distances(self.flatten().values.to(device).median(0)[0].to(self.device),'Median Distance')

    @property
    def mean_by_bond_separation(self):
        out = self.mean
        n = out.num_beads
        for i in range(1,n):
            out.values[...,range(n-i),range(i,n)] = out.values[...,range(n-i),range(i,n)].mean()
            out.values[...,range(i,n),range(n-i)] = out.values[...,range(n-i),range(i,n)]
        return out

    @property
    def std(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return Distances(self.flatten().values.to(device).std(0).to(self.device),'Standard Deviation')

    @property
    def var(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return Distances(self.flatten().values.to(device).var(0).to(self.device),'Variance')
    
    
    ########################################################
    # Convert to/from the normalized form used with the U-Net
    def normalize_(self,normalizer=None,*args,**kwargs):
        normalizer = Normalizer(*args,**kwargs) if normalizer is None else normalizer
        self.__dist_maps = normalizer.normalize(self.__dist_maps)
        return self

    def normalize(self,*args,**kwargs):
        return self.clone().normalize_(*args,**kwargs)

    def unnormalize_(self,normalizer=None,*args,**kwargs):
        normalizer = Normalizer(*args,**kwargs) if normalizer is None else normalizer
        self.__dist_maps = normalizer.unnormalize(self.__dist_maps)
        return self

    def unnormalize(self,*args,**kwargs):
        return self.clone().unnormalize_(*args,**kwargs)

    ########################################################
    # Helps compare these samples to samples from other classes
    #@property
    def __is_exact(self,other_distances):
        n = self.num_beads
        i,j = torch.triu_indices(n,n,1)
        corrcoef = torch.corrcoef(
            torch.stack(
                [
                    self.values[...,i,j].flatten(),
                    other_distances[...,i,j].flatten()
                ],
                dim=0
            )
        )[0,1]
        return corrcoef == 1
        
    @property
    def is_exact(self):
        '''
        Convert to coordinates -- without optimization -- and back
        to distances. Even when using an exact solution, torch.allclose
        doesn't work due to numerical precision issues. However, using
        the pearson correlation coefficient seems to work
        '''
        #reconstructed_dists = coords_to_dists(dists_to_coords(self.values))
        self.uncorrected_coordinates.distances
        return self.__is_exact(reconstructed_dists)

    ########################################################
    # Optimize a set of coordinates to best match the given
    # distance maps
    def correct_coordinates(
        self,
        coords=None,
        *,
        min_loss_change=1e-6,
        num_iterations=10_000, # This is a bit overkill, but trying to get the best results possible for the paper 
        lr=.1,
        lr_decay=0,
        weight_decay=0,
        r_c=1.0,
        long_scale=1/8
    ):
        coords = self if coords is None else coords
        if issubclass(type(coords),ConformationsABC):
            if type(coords) == Distances: # To avoid recursion
                coords = dists_to_coords(coords.values)
            else:
                coords = coords.coordinates.values

        if type(coords) != torch.Tensor:
            coords = torch.tensor(coords)
        coords = coords.to(dtype=self.dtype,device=self.device)

        coords = correct_coords(
            coords,
            self.values,
            min_loss_change=min_loss_change,
            num_iterations=num_iterations,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            r_c=r_c,
            long_scale=long_scale
        )

        return coords

    ########################################################
    # Visualization
    def plot(
        self,
        selection=0, # integer index, 'mean', 'mean_by_bond_separation', 'std', or 'var'
        *,
        fig=None,
        ax=None,
        cmap='RdBu',
        xticks=[],
        xticklabels=None,
        yticks=[],
        yticklabels=None,
        xlabel='Genomic index',
        ylabel='Genomic index',
        cbar_label=None,
        cbar_orientation=None,
        **kwargs
    ):
        
        if len(self) == 1:
            if cbar_label is None:
                cbar_label = self.__description
            return plot_dist_map(
                self.values[0,...].squeeze().cpu(),
                fig=fig,
                ax=ax,
                cmap=cmap,
                xticks=xticks,
                xticklabels=xticklabels,
                yticks=yticks,
                yticklabels=yticklabels,
                xlabel=xlabel,
                ylabel=ylabel,
                cbar_label=cbar_label,
                cbar_orientation=cbar_orientation,
                **kwargs
            )
            
        if type(selection) == int:
            to_plot = self[selection]

        elif selection == 'mean':
            to_plot = self.mean

        elif selection == 'median':
            to_plot = self.median

        elif selection == 'mean_by_bond_separation':
            to_plot = self.mean_by_bond_separation

        elif selection == 'std':
            to_plot = self.std

        elif selection == 'var':
            to_plot = self.var
        
        else:
            raise Exception(f"Selection should be an integer index, 'mean', 'mean_by_bond_separation', 'std', or 'var'")

        return to_plot.plot(
            fig=fig,
            ax=ax,
            cmap=cmap,
            xticks=xticks,
            xticklabels=xticklabels,
            yticks=yticks,
            yticklabels=yticklabels,
            xlabel=xlabel,
            ylabel=ylabel,
            cbar_label=cbar_label,
            **kwargs
        )

    def plot_with(
        self,
        other,
        *,
        fig=None,
        ax=None,
        cmap='RdBu',
        xticks=[],
        xticklabels=None,
        yticks=[],
        yticklabels=None,
        xlabel='Genomic Index',
        ylabel='Genomic Index',
        cbar_label=None,
        cbar_orientation=None,
        **kwargs
    ):
        assert type(other) == type(self), f'Expected {type(self)} object as input \'other\', but received {type(other)}'
        
        assert len(self) == 1 and len(other) == 1, 'Both Distances objects to be compared should have length 1, '+\
        f'but self has length {len(self)} and other has length {len(other)}'

        n = self.num_beads
        assert n == other.num_beads, 'Both distances objects have an equal number of beads, but '+\
        f'self has {n} while other has {other.num_beads}'

        to_plot = self.clone().flatten()
        i,j = torch.triu_indices(n,n,1)
        to_plot.values[0,j,i] = other.flatten().values[0,i,j]

        return to_plot.plot(
            fig=fig,
            ax=ax,
            cmap=cmap,
            xticks=xticks,
            xticklabels=xticklabels,
            yticks=yticks,
            yticklabels=yticklabels,
            xlabel=xlabel,
            ylabel=ylabel,
            cbar_label=cbar_label,
            cbar_orientation=cbar_orientation,
            **kwargs
        )

    def plot_dist_vs_separation(
        self,
        *,
        ax=None,
        xlabel='Bond Separation',
        ylabel='Distance',
        **kwargs
    ):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = None

        to_visualize = self.mean_by_bond_separation
        n = to_visualize.num_beads
        ax.plot(range(1,n),to_visualize.values[0,1:],**kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return fig, ax

    ########################################################
    # Fold/unfold 
    def fold_(self):
        if self.__is_folded:
            return self
        self.__dist_maps = self.__origami_transform(self.values.unsqueeze(-3))
        self.__is_folded = True
        return self
    
    def fold(self):
        if self.__is_folded:
            return self
        return self.clone().fold_()

    def unfold_(self):
        if not self.__is_folded:
            return self
        self.__dist_maps = self.__origami_transform.inverse(self.values,2*self.values.shape[-1]).squeeze(-3)
        self.__is_folded = False
        return self

    def unfold(self):
        if not self.__is_folded:
            return self
        return self.clone().unfold_()
        

    ########################################################
    # Converting between sample subclasses
    
    # Always be able to return coordinates object
    @property
    def uncorrected_coordinates(self):
        return Coordinates(dists_to_coords(self.values),drop_invalid_conformations=False)
        
    @property
    def coordinates(self,drop_invalid_conformations=False):
        '''
        Optimize coordinates with default optimization values if necessary
        '''
        u_coords = self.uncorrected_coordinates
        if u_coords == self: # Essentially, checks is_exact
            return u_coords
        coords = self.correct_coordinates(u_coords) # This isn't working for some reason
        #coords = self.correct_coordinates()
        return Coordinates(coords,drop_invalid_conformations=drop_invalid_conformations)

    # Always be able to return trajectory object
    @property
    def trajectory(self): 
        return self.coordinates.trajectory

    # Always be able to return distance maps 
    @property
    def distances(self):
        return self



from Coordinates import Coordinates
