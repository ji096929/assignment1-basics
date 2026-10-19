import regex as re
from collections import Counter,defaultdict

def init_vocab(special_tokens:list[str]):
    #初始化256个初始词
    vocab={i:bytes([i]) for i in range(256)}
    current_id=256
    for token in special_tokens:
        vocab[current_id]=token.encode("utf-8")
        current_id+=1
    return vocab,current_id



def run_train_bpe(input_path:str,vocab_size:int,special_tokens:list[str]):
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        #预分词频表，用来统计pair时乘数量，键内部是多个token，内部逐渐合并
        #预分词内部组合起来是保持不变的，但是组合方式会变化，新旧形态也不能算作同一个预分词
        pretoken_freqs:Counter[tuple[bytes, ...]] =Counter()
        #merges合并列表,存的是所有best pair
        merges:list[tuple[bytes,bytes]]=list()
        #token表，只增不减
        vocab,current_id=init_vocab(special_tokens)
        #在正则化后的文本中统计预分词词频
        text_chunks=text.split("<|endoftext|>")
        for chunk in text_chunks:
            for m in re.finditer(PAT,chunk):
                #原本是字符串，需要先转为一个个字节
                encoded_bytes=m.group().encode("utf-8")
                pretoken_tuple = tuple(bytes([b]) for b in encoded_bytes)
                pretoken_freqs[pretoken_tuple]+=1
        #建立pair在哪个预分词的索引，当更新bestpair的时候只更新相关索引或者含bestpair的预分词，减少循环次数
        pair_on_pretoken:defaultdict[tuple[bytes, bytes],set[tuple[bytes,...]]]=defaultdict(set)
        #pair的频率
        counts:Counter[tuple[bytes, bytes]] =Counter()
        for pretoken,freq in pretoken_freqs.items():
            for pair in zip(pretoken,pretoken[1:]):
                pair_on_pretoken[pair].add(pretoken)
                #在预分词词频表中统计相邻对的频率
                counts[pair]+=freq
        for _ in range(vocab_size-256-len(special_tokens)):
            #挑选频率最大&字典序最大
            best_pair=max(counts,key=lambda p:(counts[p],p))
            merges.append(best_pair)
            vocab[current_id]=best_pair[0]+best_pair[1]
            current_id+=1
            #遍历并替换
            #扫这个预分词旧形态，扣除它贡献的所有的pair的次数，后面再以新形态加回来
            impacted_pretokens=pair_on_pretoken[best_pair]
            #这里要在快照里遍历，不然就是边遍历边discard/add了
            for impacted_pretoken in list(impacted_pretokens):
                for pair in zip(impacted_pretoken,impacted_pretoken[1:]):
                    #减去对应pair的频率，删除pair对应的词索引
                    counts[pair]-=pretoken_freqs[impacted_pretoken]
                    pair_on_pretoken[pair].discard(impacted_pretoken)
                    if not pair_on_pretoken[pair]:  # 如果空了
                        del pair_on_pretoken[pair]
                #生成新词
                impacted_pretoken_list=list(impacted_pretoken)
                i=0
                while i<len(impacted_pretoken_list)-1:
                    if impacted_pretoken_list[i]==best_pair[0] and impacted_pretoken_list[i+1]==best_pair[1]:
                        impacted_pretoken_list[i]=best_pair[0]+best_pair[1]
                        del impacted_pretoken_list[i+1]
                    i+=1
                new_pretoken=tuple(impacted_pretoken_list)
                #这时要以新形态加回来所有pair的次数，并统计一下pair在什么预分词上
                for pair in zip(new_pretoken,new_pretoken[1:]):
                    pair_on_pretoken[pair].add(new_pretoken)
                    counts[pair]+=pretoken_freqs[impacted_pretoken]
                #增加新形态预分词的频率，删去旧形态预分词的频率
                pretoken_freqs[new_pretoken]+=pretoken_freqs[impacted_pretoken]
                del pretoken_freqs[impacted_pretoken]
                
        return vocab,merges


        


if __name__ == "__main__":
    vocab,merges=run_train_bpe("/home/lwz/cs336lab/assignment1-basics/cs336_basics/tokenizer.py", 500, ["<|endoftext|>"])
    print(vocab)
    


