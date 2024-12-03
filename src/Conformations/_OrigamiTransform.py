'''
Greg Schuette 2024
'''
import torch

def remove_diagonal(x):
    s = x.shape[-1]
    i,j = torch.triu_indices(s,s,1)
    x2 = torch.empty(*x.shape[:-2],s-1,s-1,dtype=x.dtype,device=x.device)
    #x2[...,i,j-1] = x[...,i,j]
    x2[...,j-1,i] = x[...,j,i] #overwrites on the diagonal
    x2[...,i,j-1] = x[...,i,j]
    return x2

def add_diagonal(x,diag_value=0):
    s = x.shape[-1]
    i,j = torch.triu_indices(s,s,0)
    x2 = torch.empty(*x.shape[:-2],s+1,s+1,dtype=x.dtype,device=x.device)
    x2[...,i,j+1] = x[...,i,j]
    x2[...,j+1,i] = x[...,j,i] 
    i = torch.arange(s+1)
    x2[...,i,i] = diag_value
    return x2

def transpose_minor_axis(tensor):
    return tensor.flip(-1).transpose(-2,-1).flip(-1)

def ceil(val):
    f_val = float(val) # in case a string representing an int is passed or something like that
    i_val = int(val)
    return i_val if i_val==f_val else i_val+1

def is_odd(val):
    val = float(val)
    i_val = int(val)
    if val != i_val:
        # non-integers aren't counted as odd or even
        return False
    return (val%2) == 1

class OrigamiTransform:

    def __init__(self,num_reduction_steps=1,drop_lower_triangle=True,preserve_diagonal=False):
        self.num_reduction_steps = num_reduction_steps
        self.drop_lower_triangle = drop_lower_triangle
        self.preserve_diagonal = preserve_diagonal

        # Ensure valid inputs were chosen
        self.__verify_and_format_settings(None,None,None)
        
    #############################
    # Verifying user inputs
    #############################
    def __verify_and_format_settings(self,num_reduction_steps,drop_lower_triangle,preserve_diagonal):

        # Should diagonals be preserved? 
        pd = self.preserve_diagonal if preserve_diagonal is None else preserve_diagonal
        assert type(pd) == bool, f'preserve_diagonal argument should be a boolean value, but received {type(pd)}'
        
        # How many folding operations to perform?
        n = self.num_reduction_steps if num_reduction_steps is None else num_reduction_steps
        try:
            nn = int(n)
            assert nn == n
            n = nn
        except:
            raise Exception(f'num_reduction_steps must take an integer value, but received type {type(n)}')
        assert n > -1, f'Number of reduction steps cannot be negative! Received {n}'

        # Should the lower triangle be dropped?
        dlt = self.drop_lower_triangle if drop_lower_triangle is None else drop_lower_triangle
        assert type(dlt) == bool, f'drop_lower_triangle must be of type bool, but received input of type {type(drop_lower_triangle)}'

        return n,dlt,pd
    
    def __verify_and_format_call_input(self,input,num_reduction_steps,drop_lower_triangle,preserve_diagonal):

        n,dlt,pd = self.__verify_and_format_settings(num_reduction_steps,drop_lower_triangle,preserve_diagonal)
        
        # Input tensor to be folded
        assert type(input) == torch.Tensor, f"Input should be a torch Tensor, but received object of type {type(input)}."
        assert input.ndim > 1, f'Object should have at least two dimensions, but provided input has {input.ndim} dimensions!'
        assert input.shape[-1] == input.shape[-2], f'Final two dimensions should be of equal size, but provided input has shape {input.shape}.'
        
        
        if input.ndim == 2: # Add channel dimension to input if one is needed
            input = input.unsqueeze(0)

        return input, n, dlt

    #############################
    # Forward operation
    #############################
    def __fold_with_drop(self,input,preserve_diagonal):
        '''
        Input size: ... x C x H x W, where H == W
        Output size: ... x 2C x ceil(H/2) x ceil(W/2)

        Drops the lower triangle. 
        '''
        
        if preserve_diagonal:
            input = add_diagonal(input)

        if is_odd(input.shape[-1]):
            input = add_diagonal(input)
            
        W2 = input.shape[-1]//2

        ###
        # The upper right quadrant is placed in the front channels
        front = input[...,:W2,W2:]

        ###
        # The triangles remaining from the upper diagonal after removing
        # the upper right quadrant are folded underneath

        # Place the triangle (including diagonal) to the left of the quadrant already taken in the 
        # upper triangle of the back channels
        back = input[...,:W2,:W2].triu(0)

        # Place the triangle (excluding diagonal) underneath the quadrant already taken in the 
        # lower triangle of the back channels
        back+= transpose_minor_axis(input[...,W2:,W2:].triu(1)).transpose_(-2,-1)

        # Reverse the order of the x axis to simulate a "fold" that occured when the objects are contactenated,
        # and contactenate them, and return the result!
        return torch.cat(
            [
                front,
                back.flip(-1)
            ],
            dim=-3
        )

    def __fold_without_drop(self,input):
        # Fold the square input over the main diagonal... or at least that's what the upper triangle
        # looks like, and we can use __fold_with_drop to go further. 
        input = torch.cat(
            [
                input,
                input.flip(-3).transpose_(-2,-1)
            ],
            dim=-3
        )

        return self.__fold_with_drop(input,preserve_diagonal=True)
    
    
    def __call__(self,input,num_reduction_steps=None,drop_lower_triangle=None,preserve_diagonal=None):
        '''
        ::input:: Tensor to be folded.
            - Can be of size (any number of batch dimensions) x C x H x W, where H must equal W. 
            - Image dimensions H and W are required, while batch and channel (C) dimensions are optional. 
                - If input.ndim == 2, then a channel dimension is added since the number of channels 
                  increases during this transformation such that input has size C x H x W with C=1. 
                - If input.ndim > 2, then the third-to-last dimension is expanded directly as the channel dimension
                
        ::num_reduction_steps:: Number of times to fold the data. Must be an integer >= 0. 
        
        ::drop_lower_triangle:: Boolean value indicating whether to drop the lower triangle, removing redundant 
            information if the input is symmetric across the main diagonal. 
            - If True, the first fold yields 2C dimension
            - If False, the first fold yields 4C dimensions
            - Regardless, all further folds multiply the number of channels by 4.

        EFFECTS ON DIMENSIONS
        If necessary, a channel dimension is added to the input, yielding: 
        Input size: ... x C x H x W (the channel dimension is added if necessary) 

        Redefining num_reduction_steps as n for ease of notation, the output size is:
            - ... x (4**(n-1) * 2C) x (H // 2**n) x (H // 2**n) if drop_lower_triangle is True, or
            - ... x (4**n * C) x (H // 2**n) x (H // 2**n) if drop_lower_triangle is False. 

        NOTES: 
            - This transformation destroys information along the main diagonal, so don't use if that information is important!
            - For the transformation to be valid, H=W must be divisible by 2**num_reduction_steps
        '''

        #############################
        # Prepare/verify input
        input,n,dlt = self.__verify_and_format_call_input(input,num_reduction_steps,drop_lower_triangle,preserve_diagonal)

        #############################
        # Perform the transformation

        # If n is 0, then no folding steps are taken!
        if n == 0:
            return input

        # If drop_lower_triangle, then the first fold drops the lower triangle. 
        # Either way, define output for use in the following loop. 
        output = self.__fold_with_drop(input,preserve_diagonal) if dlt else input

        # Perform all folds that avoid dropping the lower triangle. 
        n_dropping_folds = int(dlt)
        for i in range(n_dropping_folds,n):
            output = self.__fold_without_drop(output)

        return output

    #############################
    # Invert the transformation
    #############################

    def __reverse_fold_with_drop_(self,output):
        '''
        Everything except reflecting across the diagonal, dealing with preservation conditions on diagonal
        '''
        # Final size details
        C1 = output.shape[-3]
        C = C1 // 2
        assert 2*C == output.shape[-3], f'Folded objects are expected to have even channels, but provided object with {C1} channels!'

        unfolded_quadrant = output[...,:C,:,:]
        folded_triangles = output[...,C:,:,:].flip(-1)

        input = torch.cat(
            [
                torch.cat(
                    [
                        folded_triangles.triu(0),    # Upper left
                        unfolded_quadrant            # Upper right
                    ],
                    dim=-1
                ),
                torch.cat(
                    [
                        unfolded_quadrant,            # Lower left, just need the dimensions here
                        transpose_minor_axis(folded_triangles.tril(0)).transpose_(-2,-1) #Lower right
                    ],
                    dim=-1
                )
            ],
            dim=-2
        )

        return input.triu_(0)

    def __reverse_fold_with_drop(self,output,diag_was_preserved):

        input = self.__reverse_fold_with_drop_(output)

        # Make symmetric
        input = input + input.transpose(-2,-1)

        if diag_was_preserved:
            # If diagonal was preserved, then the diagonal in the intermediate object
            # was artificial, so remove it. 
            input = remove_diagonal(input)
        else:
            # If diagonal wasn't preserved, then whichever values were broadcast there (use case: constant value
            # on the diagonal, e.g. covariance matrices) are currently doubled, so halve them.  
            n = input.shape[-1]
            input[...,range(n),range(n)]/= 2

        return input

    def __reverse_fold_without_drop(self,output):

        was_odd = output.shape[-1] % 2 == 1
        input = self.__reverse_fold_with_drop_(output)

        # Diagonal always preserved in this case, so the current diagonal is entirely artificial. Remove it. 
        input = remove_diagonal(input)

        # Undo the square -> triangle fold
        C = input.shape[-3] // 2
        input[...,:C,:,:]+= input[...,C:,:,:].flip(-3).transpose_(-2,-1)
        input = input[...,:C,:,:]

        # The last operation doubled values along the diagonal, so halve them 
        n = input.shape[-1]
        input[...,range(n),range(n)]/= 2

        return input

    def inverse(self,output,final_imsize,*,num_unfolding_steps=None,dropped_lower_triangle=None,preserved_diagonal=None):
        '''
        Reverse the operations described in the __call__ function
        '''

        assert type(output) == torch.Tensor, f'Expected torch Tensor, but received {type(output)}'
        assert output.ndim > 2, f'Must have at least 3 dimensions in output. Received {output.shape}'
        assert output.shape[-2]==output.shape[-1], f'Last two dimensions of output must be equal in size. Received {output.shape}'

        # Verify/format the inputs
        n,dlt,pd = self.__verify_and_format_settings(num_unfolding_steps,dropped_lower_triangle,preserved_diagonal)

        if n == 0:
            return output

        # Channels in the folded state
        C = output.shape[-3]
        
        # Channels in the unfolded state
        if dlt:
            c1 = C / (2 * 4**(n-1))
        else:
            c1 = C / 4**n
        c = int(c1)
        if c != c1:
            raise Exception(
                f'The starting object\'s number of channels ({C}) and specified dropped_lower_triangle value ({dlt}) '+\
                f'would require final number of channels to be {c1}, which is invalid!'
            )

        # Image size expected at each step
        imsizes=[final_imsize]
        for i in range(n):
            n1 = imsizes[0]
            if i > 0 or pd or not dlt:
                n1 = n1 + 1
            if n1%2 == 1:
                n1 = n1 + 1
            imsizes.insert(0,n1//2)

        assert imsizes[0] == output.shape[-1], 'The provided settings expect the folded object to have size '+\
        f'{imsizes[0]} in the final two dimensions, but received size {output.shape[-1]}'

        n1 = n - 1
        for i,imsize in enumerate(imsizes[1:]):
            
            if i == n1 and dlt:
                output = self.__reverse_fold_with_drop(output,pd)
            else:
                output = self.__reverse_fold_without_drop(output)
            
            if output.shape[-1] != imsize:
                output = remove_diagonal(output)

        return output

if __name__ == '__main__':
    # Validate the code with several options. 
    origami_transform = OrigamiTransform()
    
    print('Correct\t| Num Beads\t| Preserve Diagonal | Num Steps\t| Drop LT')
    for n in [64,65]:
        for preserve_diagonal in [True,False]:
            a = torch.arange(n).float().reshape(1,1,1,n).repeat(5,1,n,1).triu_(1-int(preserve_diagonal))
            a = a + a.transpose(-2,-1)
            a[...,range(n),range(n)]/=2
            for num_steps in [1,2,3]:
                for drop_lower_triangle in [True,False]:
                    b = origami_transform(a,num_reduction_steps=num_steps,
                                          preserve_diagonal=preserve_diagonal,
                                          drop_lower_triangle=drop_lower_triangle)
                    aa = origami_transform.inverse(b,final_imsize=a.shape[-1],
                                                   num_unfolding_steps=num_steps,
                                                   preserved_diagonal=preserve_diagonal,
                                                   dropped_lower_triangle=drop_lower_triangle)
                    s = f'{(a==aa).all()}' + '\t'
                    s+= f'  {n}' + '\t\t'
                    s+= f'  {preserve_diagonal}' + '\t\t'
                    s+= f'      {num_steps}' + '\t\t'
                    s+= f'  {drop_lower_triangle}'
                    print(s)
        

        
