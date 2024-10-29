import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch import Tensor
from einops.layers.torch import Rearrange
from einops import rearrange
from typing import Optional
import copy

from torch import einsum

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        conv_kernel_size1 = 10
        conv_kernel_size2 = 8
        pool_kernel_size1 = 5
        pool_kernel_size2 = 4
        self.conv_net = nn.Sequential(
            nn.Conv1d(5, 256, kernel_size=conv_kernel_size1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv1d(256, 256, kernel_size=conv_kernel_size1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size1, stride=pool_kernel_size1),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.1),
            nn.Conv1d(256, 360, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv1d(360, 360, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_kernel_size2, stride=pool_kernel_size2),
            nn.BatchNorm1d(360),
            nn.Dropout(p=0.1),
            nn.Conv1d(360, 512, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Conv1d(512, 512, kernel_size=conv_kernel_size2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2))
        self.num_channels = 512
    def forward(self, x):
        out = self.conv_net(x)
        return out

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    def forward(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu"
                 ):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        if num_decoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation)
            encoder_norm = nn.LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos_embed=None, mask=None):
        src = src.permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory.transpose(0,1)

class AttentionPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pool_fn = Rearrange('b (n p) d-> b n p d', n=1)
        self.to_attn_logits = nn.Parameter(torch.eye(dim))

    def forward(self, x):
        attn_logits = einsum('b n d, d e -> b n e', x, self.to_attn_logits)
        x = self.pool_fn(x)
        logits = self.pool_fn(attn_logits)

        attn = logits.softmax(dim = -2)
        return (x * attn).sum(dim = -2).squeeze()

class Tranmodel(nn.Module):
    def __init__(self, backbone, transformer, bins=200, max_bin=10, in_dim=64,embed_dim=256):
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        '''
        if backbone is None:
            self.backbone = CNN()
        else:
            self.backbone = backbone
        if transformer is None: 
            self.transformer = **kwargs
        self.transformer = transformer
        '''
        hidden_dim = transformer.d_model
        self.input_proj = nn.Conv1d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.bins=bins
        self.max_bin=max_bin
        self.attention_pool = AttentionPool(hidden_dim)
        self.project=nn.Sequential(
            Rearrange('(b n) c -> b c n', n=bins*5),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=15, padding=7,groups=hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Conv1d(hidden_dim, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.cnn=nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=15, padding=7),
            nn.BatchNorm1d(embed_dim),
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1),
            nn.Dropout(0.2),
            Rearrange('b c n -> b n c')
        )

    @property 
    def device(self): 
        return [*self.backbone.conv_net[0].parameters()][0].device
    
    def forward(self, input):
        input=rearrange(input,'b n c l -> (b n) c l')
        #input=Rearrange(input,'b n c l -> (b n) c l')
        src = self.backbone(input)
        src = self.input_proj(src)
        src = self.transformer(src)
        src = self.attention_pool(src)
        src = self.project(src)
        src = self.cnn(src)
        return src

    ###########################
    # For ease, I'm not generalizing this for now. Stick with the defaults used to this point in the CNN
    def get_pretrained_model(
        bins=260,#200,
        in_dim=64,
        max_bin=10,
        nheads=4,
        hidden_dim=512,
        embed_dim=256,
        dim_feedforward=1024,
        enc_layers=1,
        dec_layers=2,
        dropout=0.2,
        param_filepath="../../data/models/hic_GM12878_transformer.pt",
        allow_further_training=False
    ):

        backbone = CNN()
        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers
        )

        model = Tranmodel(
            backbone=backbone,
            transformer=transformer,
            embed_dim=embed_dim,
            bins=bins,
            max_bin=max_bin, 
            in_dim=in_dim
        )

        model.load_state_dict(torch.load(param_filepath,map_location='cpu'))
        model.requires_grad_(allow_further_training)
        if not allow_further_training:
            model.eval() 
            
        
        return model

        
