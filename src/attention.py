import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
logger = logging.getLogger(__name__)

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, mask=None, dropout=None):
    B,_,T,d_k = query.size()
    scores = torch.matmul(query,key.transpose(-2,-1))/(math.sqrt(d_k)) # dot product b/w query,key

    if mask is not None:
        scores = scores.masked_fill(mask[:,:,:T,:T]==0,-1e10) # make future words unreachable -inf
    prob_attn = F.softmax(scores,dim=-1) # calculating probs
    if dropout is not None:
        prob_attn = dropout(prob_attn) # pass through dropout

    return torch.matmul(prob_attn,value) # attn_weights * value # weighted sum of values. each emb idx is a weighted sum of all other emb idxs of all T values

class CausalSelfAttention(nn.Module):


    def __init__(self, config):
        super().__init__()
        d_model = config.n_embd
        self.n_head = config.n_head
        assert d_model % config.n_head == 0 # d_model/n_head are divisble
        self.d_k = d_model//self.n_head

        self.linears = clones(nn.Linear(d_model,d_model),4) # key, value, query, out_proj
        
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)

        block_size = config.block_size
        # to hide future words
        subsequent_mask = torch.tril(torch.ones(block_size,block_size)).view(1,1,block_size,block_size)
        self.register_buffer("mask",subsequent_mask) # to make sure it is stored in states dict while saving model

      
    def forward(self, x, layer_past=None):
        B,T,d_model = x.size()
        query,key,value = [l(x).view(B,-1,self.n_head,self.d_k).transpose(1,2) for l,x in zip(self.linears,(x,x,x))]
        #print(x.shape)
        y = attention(query,key,value,mask=self.mask,dropout=self.attn_drop)
        
        y = y.transpose(1,2).contiguous().view(B,T,d_model)
        #print(y.shape)
        return self.resid_drop(self.linears[-1](y)) #pass through a linear and dropout



class SynthesizerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.w1 = nn.Linear(config.n_embd, config.n_embd)
        self.w2 = nn.Parameter(torch.zeros(config.n_embd // config.n_head,
            config.block_size-1)) #d_k,T
        self.b2 = nn.Parameter(torch.zeros(config.block_size-1)) #T
        # value projection
        self.value = nn.Linear(config.n_embd, config.n_embd) #dmodel,dmodel
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd) #dmodel,dmodel
        # causal mask to ensure that attention is only applied to the left in
        #     the input sequence
        self.register_buffer("mask", torch.tril(
            torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size)) #mask
        self.n_head = config.n_head
        self.block_size = config.block_size

        nn.init.uniform_(self.w2,-0.001,0.001)

    def forward(self, x, layer_past=None):
        B, T, C = x.size()
        # @ : The matrix multiplication(s) are done between the last two dimensions
        d_k = C//self.n_head
        relu_out = F.relu(self.w1(x)).\
            view(B,T,self.n_head,d_k).transpose(1,2)     

        v = self.value(x).view(B,T,self.n_head,d_k).transpose(1,2)   
        scores = (relu_out@self.w2)  + self.b2  
        
        scores = scores[:,:,:T,:T] # to ensure it runs for T<block_size
        scores = scores.masked_fill(self.mask[:,:,:T,:T]==0,-1e10)
        prob_attn = F.softmax(scores,dim=-1)
        y = prob_attn@v

        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.resid_drop(self.proj(y))
        return y