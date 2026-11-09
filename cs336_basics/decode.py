import argparse
import numpy as np
import torch
import os
import wandb
from cs336_basics import model as model_impl

@torch.no_grad()
def decode(model:model_impl.MyTransformerLm,prompt:torch.Tensor,max_length,temperature,top_p,eos_token_id):
    model.eval()
    device=prompt.device
    generated=prompt.clone()
    for _ in range(max_length):
        input_ids=generated
        logit=model(input_ids)
        #只需要取最后一个位置的预测结果
        next_token_logits=logit[:,-1,:]
        #与温度系数做计算
        if temperature>0:
            next_token_logits=next_token_logits/temperature
        else:
            next_token=torch.argmax(next_token_logits,dim=-1,keepdim=True)
            generated=torch.cat([generated,next_token],dim=-1)
            continue
        #应用核采样，topk
        if top_p<1:
            sorted_logits,sorted_indices=torch.sort(next_token_logits,descending=True,dim=-1)
            sorted_probs=torch.softmax(sorted_logits,dim=-1)
            cumulative_probs=torch.cumsum(sorted_probs,dim=-1)
            #找到累计概率超过p的位置
            sorted_indices_to_remove=cumulative_probs>top_p
            #把第一个超过p的保留下来
            sorted_indices_to_remove[...,1:]=sorted_indices_to_remove[...,:-1].clone()
            sorted_indices_to_remove[...,0]=0
            #被剔除的设置为负无穷
            sorted_logits=sorted_logits.masked_fill(sorted_indices_to_remove,float('-inf'))
            next_token_logits=sorted_logits.scatter(-1,sorted_indices,sorted_logits)
        #重新计算softmax
        probs=torch.softmax(next_token_logits,-1)
        #采样
        next_token=torch.multinomial(probs,1)
        #拼接
        generated=torch.cat([generated,next_token],-1)
        #判断是否到达eos
        if eos_token_id is not None:
            if(next_token==eos_token_id).all():
                break
    return generated