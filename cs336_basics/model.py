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

