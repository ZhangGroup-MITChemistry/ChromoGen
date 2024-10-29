import torch
from torch import nn

# Support functions
class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class Stacker(nn.Module):
    def forward(self,x):
        h = x.shape[-2] // 2
        return torch.cat(
            (
                x[...,:h,:],
                x[...,h:,:].flip(-2) # keep sequentially close embedding params close together
            ),
            dim=-3
        )

# Reinsert the embedding at each layer 
class EmbedWithReinsertion(nn.Module):

    def __init__(
        self,
        dim,
        embedding_dimensions
    ):
        super().__init__()
        
        
        classes_dim = dim * 4
        c,h,w = embedding_dimensions
        c1 = c

        if h == 512:
            self.fold_data = True
        elif h == 260:
            self.fold_data = False
        else:
            raise Exception('These embeddings dimensions haven\'t been accounted for!')
        
        s = w #256 # Final size, square, after first layer

        self.layers = nn.ModuleList([])#[]
        if h == 260:
            self.stacker = None
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels=c,out_channels=2*c,kernel_size=7,stride=1, padding=(1,3)), # (1,260,256) -> 2x256x256
                nn.GELU()
            ))
            padding,padded_size = (1,3), 262
        elif h == 512:
            self.stacker = Stacker()
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels=2*c,out_channels=2*c,kernel_size=7,stride=1, padding=(3,3)), # (2,256,256) -> 2x256x256
                nn.GELU()
            ))
            c1*= 2
            padding,padded_size = (3,3), 262
        else:
            raise Exception('These embeddings dimensions haven\'t been accounted for!')

        self.size_reducers = nn.ModuleList([])#[]
        c*= 2 # update number channels 
        for k in range(4): # Overall change: (2,256,256) -> (32,16,16)
            
            # Lengthen, mix data
            self.layers.append( # (C,H,W) -> (2*C-1,H,W)
                nn.Sequential(
                    nn.Conv2d(c,2*c-c1,kernel_size=3,stride=1, padding=2) # (C,H,W) -> (2*C,H,W)
                )
            )
            c*= 2 # update # channels
            c-= c1
    
            # Mix again without lengthening 
            self.layers[-1].append( # (2*C-1,H,W) -> (2*C-1,H,W)
                nn.Conv2d(c,c,kernel_size=3,stride=1, padding=2) 
            )
    
            # Pool to decrease data size 
            self.layers[-1].append(
                nn.AdaptiveMaxPool2d((s//2,s//2)) # (2*C-1,H,W) -> (2*C-1,H/2,W/2) 
            )
            s//=2 # Update height/width
    
            # Activate 
            self.layers[-1].append(nn.GELU())

            #############
            # this will be used to append to the known data, for (2*C-1,H/2,W/2) -> (2*C,H/2,W/2)
            #kernel_size = (padded_size - s) + 1
            #self.size_reducers.append(
            #    nn.Conv2d(c1,1,kernel_size=kernel_size,stride=1,padding=padding)#nn.AdaptiveMaxPool2d((s,s))
            #)
            self.size_reducers.append(
                nn.AdaptiveMaxPool2d((s,s))
            )
            c+= c1

        self.linear_out = nn.Sequential(
            Flatten(),
            nn.Linear(c*s**2,classes_dim)
        )

    def forward(self,batch):
        if self.stacker is not None:
            batch = self.stacker(batch)
        out = batch.clone()
        for i,layer in enumerate(self.layers):
            out = layer(out)
            if i > 0:
                out = torch.cat(
                    (
                        out,
                        self.size_reducers[i-1](batch)
                    ),
                    dim=-3
                )
        out = self.linear_out(out) 
        return out

# Original embedder; data is not reinserted in each layer. 
class Embed(nn.Module):

    def __init__(
        self,
        dim,
        embedding_dimensions
    ):
        super().__init__()

        classes_dim = 4 * dim
        c,h,w = embedding_dimensions
        
        s = embedding_dimensions[-1] #256 # Final size, square
        
        if embedding_dimensions[-2] == 260:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=c,out_channels=2*c,kernel_size=7,stride=1, padding=(1,3)), # (1,260,256) -> 2x256x256
                nn.GELU()
            )
        elif embedding_dimensions[-2] == 512:
            self.model = nn.Sequential(
                Stacker(), # (1,512,256) -> (2,256,256)
                nn.Conv2d(in_channels=2*c,out_channels=2*c,kernel_size=7,stride=1, padding=(3,3)), # (2,256,256) -> 2x256x256
                nn.GELU()
            )
        else:
            raise Exception('These embeddings dimensions haven\'t been accounted for!')

        c*= 2 # update number channels 
        for k in range(4): # Overall change: (2,256,256) -> (32,16,16)
            
            # Lengthen, mix data
            self.model.append(
                nn.Conv2d(c,2*c,kernel_size=3,stride=1, padding=2) # (C,H,W) -> (2*C,H,W)
            )
            c*= 2 # update # channels

            # Mix again without lengthening 
            self.model.append( # (2*C,H,W) -> (2*C,H,W)
                nn.Conv2d(c,c,kernel_size=3,stride=1, padding=2) 
            )

            # Pool to decrease data size 
            self.model.append(
                nn.AdaptiveMaxPool2d((s//2,s//2)) # (2*C,H,W) -> (2*C,H/2,W/2) 
            )
            s//=2 # Update height/width

            # Activate 
            self.model.append(nn.GELU()) 

        self.model.append(Flatten())
        self.model.append( nn.Linear(c*s**2,classes_dim) )

    def forward(self,x):
        return self.model(x) 
