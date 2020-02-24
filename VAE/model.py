import torch
from torch import nn, optim
from utils import init_weights
from torch.nn import functional as F
import numpy as np


class AE(nn.Module):
    def __init__(self, device):
        super(AE, self).__init__()
        self.device = device
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.distributions = nn.ModuleList([self.encoder, self.decoder])
        init_weights(self)
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, feed_dict, return_img=False):
        img, bak = feed_dict["img"].to(self.device), feed_dict["bak"].to(self.device)
        yaw = self.encoder(img)
        _img = self.decoder(yaw, bak)
        if return_img:
            return _img
        loss = torch.sum((_img - img)**2) / len(img)
        return loss
    
    def train_(self, feed_dict):
        self.train()
        self.optimizer.zero_grad()
        loss = self.forward(feed_dict)
        loss.backward()
        self.optimizer.step()
        return loss

    def test_(self, feed_dict):
        self.eval()
        with torch.no_grad():
            loss = self.forward(feed_dict)
        return loss
    
    def reconst(self, feed_dict):
        _img = self.forward(feed_dict, return_img=True)
        return _img
    
    def sample_img(self, feed_dict):
        bak = feed_dict["bak"].to(self.device)
        yaw = [2 * np.pi * y / 64 for y in range(64)]
        yaw = [[np.cos(y), np.sin(y)] for y in yaw]
        yaw = torch.from_numpy(np.array(yaw).astype(np.float32)).to(self.device)
        _img = self.decoder(yaw, bak[:64])
        return _img


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc1 = nn.Linear(1024, 2)

    def forward(self, x):
        h = F.relu(self.conv1(x))  # 31x31
        h = F.relu(self.conv2(h))  # 14x14
        h = F.relu(self.conv3(h))  # 6x6
        h = F.relu(self.conv4(h))  # 2x2
        h = h.view(x.size(0), 1024)
        yaw = self.fc1(h)  # 1024 -> 2
        print(torch.norm(yaw))
        yaw = yaw / torch.norm(yaw)
        return yaw


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(2, 1024)
        self.conv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, yaw, bak):
        h = self.fc1(yaw)  # linear
        h = h.view(yaw.size(0), 1024, 1, 1)
        h = F.relu(self.conv1(h))  # 5x5
        h = F.relu(self.conv2(h))  # 13x13
        h = F.relu(self.conv3(h))  # 30x30
        h = self.conv4(h) + bak    # 64x64
        return torch.clamp(h, 0, 1)