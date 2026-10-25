from multiprocessing import Pool
import regex as re
from collections import Counter,defaultdict
import heapq
from typing import BinaryIO
import os

class RevPair:
    __slots__ = ('p',)
    def __init__(self, p):
        self.p = p
    def __lt__(self, other):          # 堆只需要 <
        return self.p > other.p       # 故意反着比

def init_vocab(special_tokens:list[str]):
    #初始化256个初始词
    vocab={i:bytes([i]) for i in range(256)}
    current_id=256
    for token in special_tokens:
        vocab[current_id]=token.encode("utf-8")
        current_id+=1
    return vocab,current_id

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def process_chunk(input_path:str,start,end):
    with open(input_path, "rb") as f:
        f.seek(start)                 # 1. 把光标移到起点
        raw = f.read(end - start)   # 2. 读取长度刚好是差值的字节数
        chunk=raw.decode("utf-8",errors="ignore")
        PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        #预分词频表，用来统计pair时乘数量，键内部是多个token，内部逐渐合并
        #预分词内部组合起来是保持不变的，但是组合方式会变化，新旧形态也不能算作同一个预分词
        pretoken_freqs:Counter[tuple[bytes, ...]] =Counter()
        #在正则化后的文本中统计预分词词频
        chunk_chunks=chunk.split("<|endoftext|>")
        for chunk1 in chunk_chunks:
            for m in re.finditer(PAT,chunk1):
                #原本是字符串，需要先转为一个个字节
                encoded_bytes=m.group().encode("utf-8")
                pretoken_tuple = tuple(bytes([b]) for b in encoded_bytes)
                pretoken_freqs[pretoken_tuple]+=1
        return pretoken_freqs
        

def run_train_bpe(input_path:str,vocab_size:int,special_tokens:list[str]):
    with open(input_path, "rb") as f:
        bounds=find_chunk_boundaries(f,12,b"<|endoftext|>")
    #merges合并列表,存的是所有best pair 
    merges:list[tuple[bytes,bytes]]=list()
    #token表，只增不减
    vocab,current_id=init_vocab(special_tokens)
    tasks=[(input_path,start ,end)for start,end in zip(bounds[:-1],bounds[1:])]
    with Pool(6) as pool:
        results=pool.starmap(process_chunk,tasks)
    pretoken_freqs:Counter[tuple[bytes, ...]] =Counter()
    for partial_pretoken_freqs in results:
        pretoken_freqs+=partial_pretoken_freqs
    #建立pair在哪个预分词的索引，当更新bestpair的时候只更新相关索引或者含bestpair的预分词，减少循环次数
    pair_on_pretoken:defaultdict[tuple[bytes, bytes],set[tuple[bytes,...]]]=defaultdict(set)
    #pair的频率
    counts:Counter[tuple[bytes, bytes]] =Counter()
    for pretoken,freq in pretoken_freqs.items():
        for pair in zip(pretoken,pretoken[1:]):
            pair_on_pretoken[pair].add(pretoken)
            #在预分词词频表中统计相邻对的频率
            counts[pair]+=freq
    heap = []
    for pair, freq in counts.items():
        # 存负频次。pair 是 (bytes, bytes)，Python 可以直接比较 bytes 元组
        heapq.heappush(heap, (-freq, RevPair(pair)))
    for _ in range(vocab_size-256-len(special_tokens)):
        #挑选频率最大&字典序最大
        #best_pair=max(counts,key=lambda p:(counts[p],p))
        #这里直接用max复杂度为n，太高了，在这里花费时间很长
        # === 惰性删除：清掉堆顶的过期记录 ===
        while heap:
            neg_freq, rp  = heap[0]
            current_freq = counts.get(rp.p, 0)  # 当前真实频次
            
            if -neg_freq == current_freq and current_freq > 0:
                # 有效！就是它
                break
            else:
                # 过期了，扔掉
                heapq.heappop(heap)
        
        if not heap:
            break  # 堆空了，没 pair 可合并了
        neg_freq, rp = heapq.heappop(heap)
        best_pair = rp.p   # 从包装里取回真正的 pair
        merges.append(best_pair)
        vocab[current_id]=best_pair[0]+best_pair[1]
        current_id+=1
        
        #扫这个预分词旧形态，扣除它贡献的所有的pair的次数，后面再以新形态加回来
        impacted_pretokens=pair_on_pretoken[best_pair]
        #这里要在快照里遍历，不然就是边遍历边discard/add了
        for impacted_pretoken in list(impacted_pretokens):
            freq = pretoken_freqs[impacted_pretoken]
            #减去对应pair的频率，删除pair对应的词索引
            for pair in zip(impacted_pretoken,impacted_pretoken[1:]):
                counts[pair]-=freq
                pair_on_pretoken[pair].discard(impacted_pretoken)
                if not pair_on_pretoken[pair]:  # 如果空了
                    del pair_on_pretoken[pair]
                #把新值插入，不用考虑旧值
                if counts[pair] > 0:
                    heapq.heappush(heap, (-counts[pair], RevPair(pair)))

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
                counts[pair]+=freq
                heapq.heappush(heap, (-counts[pair], RevPair(pair)))

            #增加新形态预分词的频率，删去旧形态预分词的频率
            pretoken_freqs[new_pretoken]+=freq
            del pretoken_freqs[impacted_pretoken]
                
    return vocab,merges


        


if __name__ == "__main__":
    vocab,merges=run_train_bpe("/home/lwz/cs336lab/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", 10000, ["<|endoftext|>"])
    print(vocab)
    


