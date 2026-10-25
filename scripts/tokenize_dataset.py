from cs336_basics.tokenizer import Tokenizer
from multiprocessing import Pool
import numpy as np
import regex as re
from collections import Counter,defaultdict
import heapq
from typing import BinaryIO
import os
import pickle

TOKENIZER_PKL="/home/lwz/cs336lab/assignment1-basics/tinystories_vocab_merges.pkl"
#/home/lwz/cs336lab/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt

INPUT_PATH="/home/lwz/cs336lab/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"

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

def process_chunk(tok,input_path:str,start,end):
    with open(input_path,"rb") as f:
        f.seek(start)                 # 1. 把光标移到起点
        raw = f.read(end - start)   # 2. 读取长度刚好是差值的字节数
        chunk=raw.decode("utf-8",errors="ignore")
        ids=tok.encode(chunk) 
    return np.array(ids,dtype=np.int16)

def tokenize_dataset():
    with open(TOKENIZER_PKL,"rb") as f:
        vocab,merges=pickle.load(f)
    tok=Tokenizer(vocab,merges,["<|endoftext|>"])
    with open(INPUT_PATH, "rb") as f:
        bounds=find_chunk_boundaries(f,6,b"<|endoftext|>")
    tasks=[(tok,INPUT_PATH,start ,end)for start,end in zip(bounds[:-1],bounds[1:])]
    with Pool(6) as pool:
        results=pool.starmap(process_chunk,tasks)
    arr=np.concatenate(results)
    np.save("tinystories_train.npy",arr)
    print(f"ok")
    return arr
if __name__ == "__main__":
    tokenize_dataset()