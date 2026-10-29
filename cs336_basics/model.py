import torch
import torch.nn as nn
from einops import einsum
import math
import torch.nn.functional as F

class MyLinear(nn.Module):
    def __init__(self,in_features,out_features,device=None,dtype=None):
        super().__init__()
        self.weight=nn.Parameter(torch.empty(out_features,in_features,device=device,dtype=dtype))
        std=math.sqrt(2/(in_features+out_features))
        nn.init.trunc_normal_(self.weight,mean=0,std=std,a=-3*std,b=3*std)
    def forward(self,x)->torch.Tensor:
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

class MySwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int|None = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            calc_ff = int(8 * d_model / 3)
            self.d_ff = 64 * ((calc_ff + 63) // 64)
        else:
            self.d_ff = d_ff
        self.W1=MyLinear(d_model,d_ff)
        self.W2=MyLinear(d_ff,d_model)
        self.W3=MyLinear(d_model,d_ff)
    
    def forward(self,x:torch.Tensor):
        gate=F.silu(self.W1(x))
        value=self.W3(x)
        hidden=gate*value
        out=self.W2(hidden)
        return out

class MyRoPE(nn.Module):
    def __init__(self,theta:float,d_k:int,max_seq_len:int,device=None):
        super().__init__()
        k=torch.arange(0,d_k//2,device=device).float()
        freqs=1.0/(theta**(2*k/d_k))
        pos=torch.arange(max_seq_len,device=device).float()
        angles=torch.outer(pos,freqs)
        self.register_buffer("cos_cached",torch.cos(angles),persistent=False)
        self.register_buffer("sin_cached",torch.sin(angles),persistent=False)

    def forward(self,x:torch.Tensor,token_positions:torch.Tensor):
        cos=self.cos_cached[token_positions]
        sin=self.sin_cached[token_positions]
        x_even=x[...,0::2]
        x_odd=x[...,1::2]
        x_even_rot=x_even*cos-x_odd*sin
        x_odd_rot=x_even*sin+x_odd*cos
        return torch.stack([x_even_rot,x_odd_rot],dim=-1).flatten(-2)
    
def softmax(x:torch.Tensor,i:int):
    max_values,_=torch.max(x,dim=i,keepdim=True)
    x_shifted=x-max_values
    exp_x=torch.exp(x_shifted)
    sum_exp=torch.sum(exp_x,dim=i,keepdim=True)
    return exp_x/sum_exp

def scaled_dot_product_attention(Q:torch.Tensor,K:torch.Tensor,V:torch.Tensor,mask=None):
    d_k=Q.size(-1)
    scores=einsum(Q,K,"... seq_q d_k,... seq_k d_k->... seq_q seq_k")/math.sqrt(d_k)
    if mask is not None:
        scores=scores.masked_fill(mask==False,float('-inf'))
    attn_weights=softmax(scores,-1)
    output=einsum(attn_weights,V,"... seq_q seq_k,... seq_k d_v->... seq_q d_v")
    return output

class MyMultiheadSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads,theta=None,max_seq_len=None):
        super().__init__()
        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads
        self.q_proj=MyLinear(d_model,d_model)
        self.k_proj=MyLinear(d_model,d_model)
        self.v_proj=MyLinear(d_model,d_model)
        self.o_proj=MyLinear(d_model,d_model)
        if theta is not None and max_seq_len is not None:
            self.myrope=MyRoPE(theta,self.d_k,max_seq_len)

    def forward(self,x:torch.Tensor,token_positions=None):
        batch,seq_len,_=x.shape
        #投影
        q=self.q_proj(x)
        k=self.k_proj(x)
        v=self.v_proj(x)
        #切多头
        q=q.view(batch,seq_len,self.num_heads,self.d_k).transpose(1,2)
        k=k.view(batch,seq_len,self.num_heads,self.d_k).transpose(1,2)
        v=v.view(batch,seq_len,self.num_heads,self.d_k).transpose(1,2)
        #旋转编码
        if token_positions is not None:
            q=self.myrope(q,token_positions)
            k=self.myrope(k,token_positions)
        #构造因果掩码
        mask=torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool,device=x.device))
        #注意力计算
        attention=scaled_dot_product_attention(q,k,v,mask)
        #拼接
        attention=attention.transpose(1,2).contiguous().view(batch,seq_len,self.d_model)
        #糅杂
        attention=self.o_proj(attention)
        return attention