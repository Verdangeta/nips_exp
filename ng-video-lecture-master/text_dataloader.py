import torch
from torch.utils.data import DataLoader, Dataset


class Dataset_text(Dataset):
    """
    Simple Torch Dataset for char generation
    """

    def __init__(self,
                 dataset_path = "input.txt",
                 split='train',
                 block_size = 256,
                 train_size = 0.9):
        self.split = split
        self.block_size = block_size
        self.train_size = train_size
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # here are all the unique characters that occur in this text
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
        decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
        
        data = torch.tensor(encode(text), dtype=torch.long)
        
        n = int(self.train_size*len(data)) # first 90% will be train, rest val
        
        self.train_data = data[:n]
        self.val_data = data[n:]

    def __len__(self):
        return len(self.train_data) - self.block_size if self.split=="train" else len(self.val_data) - self.block_size

    def __getitem__(self, index):
        if self.split=="train":
            x = self.train_data[index:index+self.block_size]
            y = self.train_data[index+1:index+self.block_size+1]
        else:
            x = self.val_data[index:index+self.block_size]
            y = self.val_data[index+1:index+self.block_size+1]
        return x, y