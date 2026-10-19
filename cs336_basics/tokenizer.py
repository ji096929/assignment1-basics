import regex as re
from collections import Counter

def init_vocab(special_tokens:list[str])->dict[int,bytes]:
    #初始化256个初始词
    vocab={i:bytes([i]) for i in range(256)}
    current_id=256

    for token in special_tokens:
        vocab[current_id]=token.encode("utf-8")
        current_id+=1

    return vocab



def run_train_bpe(input_path:str,vocab_size:int,special_tokens:list[str]):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        #词频表，用来统计pair时乘数量，键内部是多个分词，内部逐渐合并
        word_freqs:Counter[tuple[bytes, ...]] =Counter()
        #pair的频率
        counts:Counter[tuple[bytes, bytes]] =Counter()
        #merges合并列表,存的是所有best pair
        merges:list[tuple[bytes,bytes]]=list()
        #词表，只增不减
        vocab=init_vocab(special_tokens)
        #在正则化后的文本中统计词频
        for m in re.finditer(PAT,text):
            #原本是字符串，需要先转为一个个字节
            encoded_bytes=m.group().encode("utf-8")
            word_tuple = tuple(bytes([b]) for b in encoded_bytes)
            word_freqs[word_tuple]+=1
        for i in range(vocab_size-256-len(special_tokens)):
            #在词频表中统计相邻对的频率
            for word,freq in word_freqs.items():
                for pair in zip(word,word[1:]):
                    counts[pair]+=freq
            #挑选频率最大&字典序最大
            best_pair=max(counts,key=lambda p:(counts[p],p))
            merges.append(best_pair)
            #遍历并替换
            new_word_freq:Counter[tuple[bytes, ...]] =Counter()
            for word,freq in word_freqs.items():
                #每个词内合并，先用不变的tuple转换为可变list
                word_list=list(word)  
                i=0
                while i<len(word_list)-1:
                    if word_list[i]==best_pair[0] and word_list[i+1]==best_pair[1]:
                        word_list[i]=best_pair[0]+best_pair[1]
                        del word_list[i+1]
                        i+=2
                    else:
                        i+=1
                new_word=tuple(word_list)
                new_word_freq[new_word]+=freq  
            word_freqs=new_word_freq
        return vocab,merges


        


if __name__ == "__main__":
    vocab,merges=run_train_bpe("/home/lwz/cs336lab/assignment1-basics/cs336_basics/tokenizer.py", 500, ["<|endoftext|>"])
    print(vocab)
    


