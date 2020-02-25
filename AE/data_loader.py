from torch.utils.data import Dataset, DataLoader
import numpy as np


class PetDataset(Dataset):
    def __init__(self, device, transform=None):
        
        self.imgs = np.load("../data/dataset/imgs.npy")
        self.imgs = (self.imgs.transpose(0,3,1,2).astype(np.float32) / 255.)
        self.baks = np.load("../data/dataset/baks.npy")
        self.baks = (self.baks.transpose(0,3,1,2).astype(np.float32) / 255.)
        assert len(self.imgs) == len(self.baks), "error"
        self.L = len(self.imgs)
        self.transform = transform

    def __len__(self):
        return self.L

    def __getitem__(self, idx):
        # out = [self.imgs[idx], self.baks[idx]]
        # if self.transform:
        # out = self.transform(inp)
        # return out
        return {"img": self.imgs[idx], "bak": self.baks[idx]}


class PetDataLoader(DataLoader):
    def __init__(self, device, B=256):
        super(PetDataLoader, self).__init__(
            PetDataset(device), batch_size=B, shuffle=True, drop_last=True)