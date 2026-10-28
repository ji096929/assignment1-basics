import torch
import torch.nn as nn
from einops import einsum
import math

class MyLinear(nn.Module):
    def __init__(self,in_features,out_features,device=None,dtype=None):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype))
        std=math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.weight,mean=0,std=std,a=-3*std,b=3*std)
    def forward(self,x):
        return einsum(x,self.weight,"... in_feature ,out_feature in_feature->... out_feature")

class MyEmbedding(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
        nn.init.trunc_normal_(self.weight,0,1,-3,3)
    def forward(self,token_ids:torch.Tensor):
        return self.weight[token_ids]

class MyRMSNorm(nn.Module):
    def __init__(self,d_model,eps:float =1e-5,device=None,dtype=None):
        super().__init__()
        self.eps=eps
        self.d_model=d_model
        self.g=nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))

    def forward(self,x:torch.Tensor):
        in_dtype=x.dtype
        x=x.to(torch.float32)
        ms=torch.mean(x**2,-1,keepdim=True)
        rms=torch.sqrt(ms+self.eps)
        x_norm=x/rms
        result=x_norm*self.g
        return result.to(in_dtype)
