from cs336_basics import tokenizer
import psutil
import pickle
import time
import resource
import sys



t0=time.perf_counter()
vocab,merges=tokenizer.run_train_bpe("/home/lwz/cs336lab/assignment1-basics/tests/fixtures/tinystories_sample_5M.txt", 10000, ["<|endoftext|>"])
t_delta=time.perf_counter()-t0
usage = resource.getrusage(resource.RUSAGE_SELF)
max_mem=usage.ru_maxrss
max_length_token=max(vocab.values(),key=len)
data_to_save = {
    "t_delta": t_delta, 
    "max_mem": max_mem, 
    "max_length_token": max_length_token
}
with open("/home/lwz/cs336lab/assignment1-basics/dump/train_bpe.txt", "wb") as f:
    pickle.dump(data_to_save,f)
print(t_delta,max_mem,max_length_token)