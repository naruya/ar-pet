from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
from itertools import chain
import pickle


class PetDataset(Dataset):
    def __init__(self, device, transform=None):
        
        self.imgs = np.load("../data/dataset/imgs.npy")[:2027]
        self.imgs = (self.imgs.transpose(0,3,1,2).astype(np.float32) / 255.)
        self.baks = np.load("../data/dataset/baks.npy")[:2027]
        self.baks = (self.baks.transpose(0,3,1,2).astype(np.float32) / 255.)
        
        print("using partial dataset")
        print("using partial dataset")
        print("using partial dataset")
        print("using partial dataset")
        print("using partial dataset")
        print("using partial dataset")
        print("using partial dataset")
        print("using partial dataset")
        print("using partial dataset")
        print("using partial dataset")
        
        aux_files = sorted(glob.glob("../data/dataset/aux/*.pkl"))[:7]
        self.auxs = [pickle.load(open(file, 'rb')) for file in aux_files]
        self.auxs = np.array(list(chain.from_iterable(self.auxs)))
        
        print(self.imgs.shape, self.baks.shape, self.auxs.shape)
        assert len(self.imgs) == len(self.baks) == len(self.auxs), "error"
        self.L = len(self.imgs)
        self.transform = transform

    def __len__(self):
        return self.L

    def __getitem__(self, idx):
        # out = [self.imgs[idx], self.baks[idx]]
        # if self.transform:
        # out = self.transform(inp)
        # return out
        return {"img": self.imgs[idx], "bak": self.baks[idx], "aux": self.auxs[idx]}


class PetDataLoader(DataLoader):
    def __init__(self, device, B=256, shuffle=True):
        super(PetDataLoader, self).__init__(
            PetDataset(device), batch_size=B, shuffle=shuffle, drop_last=True)