from cs336_basics.tokenizer import Tokenizer
from multiprocessing import Pool
import numpy as np
import regex as re
from collections import Counter,defaultdict
import heapq
from typing import BinaryIO
import os
import pickle

TRAIN_PATH="/home/lwz/cs336lab/assignment1-basics/tinystories_vocab_merges.pkl"
#/home/lwz/cs336lab/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt

def tokenize_dataset():
    with open(TRAIN_PATH,"rb") as f:
        vocab,merges=pickle.load(f)
    tok=Tokenizer(vocab,merges,["<|endoftext|>"])
    blocks=[]
    n=0
    with open("/home/lwz/cs336lab/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt","r",encoding="utf-8") as f:
       for line in f:
           blocks.append(np.array(tok.encode(line),dtype=np.int16))
           n+=1
           if n%20000==0:
               print(f"已编码{n}行")
    arr=np.concatenate(blocks)
    np.save("tinystories_train.npy",arr)
    return arr
if __name__ == "__main__":
    tokenize_dataset()