import random
import torch
from torch.utils.data import Dataset
import argparse
from scipy.stats import truncnorm


class NameDataset(Dataset):
    def __init__(self, pretraining_dataset, data):
        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad
        self.itos = pretraining_dataset.itos 
        self.stoi = pretraining_dataset.stoi 
        self.block_size = pretraining_dataset.block_size
        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))

    def __len__(self):
        # returns the length of the dataset
        return len(self.data) - 1

    def __getitem__(self, idx):
        #print(idx)
        inp, oup = self.data[idx].split('\t')
        #print(self.data[idx].split('\t'))
        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR
        x = x + self.PAD_CHAR*(self.block_size - len(x))
        y = self.PAD_CHAR*(len(inp)-1) + x[len(inp):]
        
        x = x[:-1]
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)
        return x, y

class CharCorruptionDataset(Dataset):
    def __init__(self, data, block_size):
        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars 
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n')

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # TODO [part e]: see spec above
        """
        takes an index and returns a data point (x, y) where
        x and y are Long tensors of length self.block_size. 
        """
        #x = torch.zeros((self.block_size))
        doc = self.data[idx]
        min_len, max_len = 4,int(self.block_size*7/8)
        trunc_len = random.randint(min_len,max_len)
        trunc_len = min(len(doc),trunc_len)
        #start_idx = random.randint(0,len(doc)-trunc_len)
        #trunc_doc = doc[start_idx:start_idx+trunc_len]
        trunc_doc = doc[:trunc_len]

        #masked_len = random.randint(int(1/8*trunc_len),int(3/8*trunc_len)) # masked content length 1/8-3/8 -> avg = 4/8 

        # print(doc)
        # print(trunc_doc)
        # print(len(trunc_doc),rand_trunc_start,content_len)
        #assert trunc_len >= 4, (len(doc), trunc_len, content_len, doc, idx)
        
        start_idx = random.randint(1,trunc_len-2)
        end_idx = random.randint(start_idx+1,trunc_len-1)
        #prefix_len = random.randint(1,trunc_len-content_len-1)
        
        prefix = trunc_doc[:start_idx]
        masked_content = trunc_doc[start_idx:end_idx]
        
        suffix = trunc_doc[end_idx:]

        masked_string = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_content
        masked_string = masked_string + (self.block_size-len(masked_string)) * self.MASK_CHAR
        x = torch.tensor([self.stoi[z] for z in masked_string[:-1]], dtype = torch.long)
        y = torch.tensor([self.stoi[t] for t in masked_string[1:]], dtype = torch.long)
        #print(masked_string)
        return x, y
