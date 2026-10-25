from multiprocessing import Pool
import regex as re
from collections import Counter,defaultdict
import heapq
from typing import BinaryIO
import os

class Tokenizer:
    def __init__(self,vocab:dict[int,bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str]|None=None):
        self.vocab=vocab
        #获取每个pair的优先级，方便后续encode的时候知道预分词内的这堆pair哪个最先需要合并
        self.ranks={pair:i for i,pair in enumerate(merges)}
        self.bytes_to_id={b:i for i,b in vocab.items()}
        self.special_to_id = {t: self.bytes_to_id[t.encode()] for t in special_tokens or []}
        self.special_tokens=special_tokens
    
    def _encode_normal_text(self,text):
        final_ids=[]
        PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for m in re.finditer(PAT,text):
            #原本是字符串，需要先转为一个个字节
            encoded_bytes=m.group().encode("utf-8")
            pretoken = [bytes([b]) for b in encoded_bytes]
            #不断找到rank最小的相邻对合并，直到没有相邻对在表中
            while True:
                pairs=zip(pretoken[:-1],pretoken[1:])
                valid_pairs=[p for p in pairs if p in self.ranks]
                if not valid_pairs:
                    break
                #选出rank最小的pair
                best_pair=min(valid_pairs,key=lambda p:self.ranks[p])
                i=0
                while i<len(pretoken)-1:
                    if pretoken[i]==best_pair[0] and pretoken[i+1]==best_pair[1]:
                        pretoken[i]=best_pair[0]+best_pair[1]
                        del pretoken[i+1]
                    i+=1
            for token in pretoken:
                final_ids.append(self.bytes_to_id[token])
        return final_ids

    def encode(self,text:str):
        final_ids=[]
        if not self.special_to_id:
            return self._encode_normal_text(text)
        #1.构造正则，按长度降序，加re.escape防止转义，加（）保留分隔符
        pattern="("+"|".join(re.escape(t) for t in sorted(self.special_tokens,key=len,reverse=True))+")"
        #2.切分，得到片段
        segments=re.split(pattern,text)
        #3.逐段判断
        for seg in segments:
            if not seg:
                continue
            if seg in self.special_to_id:
                #单个token直接apend
                final_ids.append(self.special_to_id[seg])
            else:
                final_ids.extend(self._encode_normal_text(seg))
        return final_ids
    
    def encode_iterable(self,iterable):
        for chunk in iterable:
            yield from self.encode(chunk)
    
    def decode(self,ids:list[int]):
        data=b"".join(self.vocab[i] for i in ids)
        return data.decode("utf-8",errors="replace")

# if __name__ == "__main__":
    
    


