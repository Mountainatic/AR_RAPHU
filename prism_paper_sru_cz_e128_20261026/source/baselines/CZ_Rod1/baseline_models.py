"""Common availability adapter; official iTransformer and TimeXer architectures."""
from pathlib import Path
from types import SimpleNamespace
import sys, math
import torch
from torch import nn
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor'/'TimeXer'))
from models.iTransformer import Model as InvertedTransformer
from models.TimeXer import Model as ExogenousTransformer

class LSTM(nn.Module):
    def __init__(self,c,d):
        super().__init__();self.rnn=nn.LSTM(c,d,num_layers=2,dropout=.1,batch_first=True)
        self.head=nn.Linear(d,1)
    def forward(self,x):return self.head(self.rnn(x)[0][:,-1]).squeeze(-1)

class Transformer(nn.Module):
    def __init__(self,c,d,L):
        super().__init__();self.embed=nn.Linear(c,d)
        position=torch.arange(L).float().unsqueeze(1);div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.)/d))
        pe=torch.zeros(1,L,d);pe[0,:,0::2]=torch.sin(position*div);pe[0,:,1::2]=torch.cos(position*div)
        self.register_buffer('pe',pe)
        layer=nn.TransformerEncoderLayer(d,4,4*d,.1,activation='gelu',batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.head=nn.Linear(d,1)
    def forward(self,x):return self.head(self.encoder(self.embed(x)+self.pe)[:,-1]).squeeze(-1)

class Official(nn.Module):
    def __init__(self,name,c,d,L):
        super().__init__()
        config=SimpleNamespace(task_name='long_term_forecast',features='MS',seq_len=L,pred_len=1,
            use_norm=True,patch_len=8,enc_in=c,d_model=d,dropout=.1,embed='timeF',freq='t',
            factor=5,n_heads=4,d_ff=4*d,activation='gelu',e_layers=2)
        self.model=(InvertedTransformer if name=='iTransformer' else ExogenousTransformer)(config)
    def forward(self,x):return self.model(x,None,None,None)[:,-1,-1]

def build(name,c,d,L):
    if name=='LSTM':return LSTM(c,d)
    if name=='Transformer':return Transformer(c,d,L)
    return Official(name,c,d,L)
