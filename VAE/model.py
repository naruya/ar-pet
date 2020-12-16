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

    def forward(self, feed_dict, 
                return_img=False, return_img_f=False, return_sample_img=False, train=True, 
                phi=None):

        img, bak = feed_dict["img"].to(self.device), feed_dict["bak"].to(self.device)
        yaw, (s, mu, logvar) = self.encoder(img)
        kl = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())

        if phi:
            rot = np.array([[np.cos(phi), -np.sin(phi)],
                            [np.sin(phi),  np.cos(phi)]], dtype=np.float32)
            rot = torch.from_numpy(rot).to("cuda")
            yaw = torch.matmul(rot, yaw.reshape([yaw.shape[0],2,1])).reshape([yaw.shape[0], 2])

        if train:
            _img = self.decoder(yaw, s, bak)
        else:
            _img = self.decoder(yaw, mu, bak)
        loss = torch.sum((_img - img)**2) / len(img)
        
        flip = torch.from_numpy(np.array([-1, 1])).to(self.device)
        yaw_f = yaw * flip
        img_f = img[:,:,:,torch.arange(63,-1,-1)]
        bak_f = bak[:,:,:,torch.arange(63,-1,-1)]

        if train:
            _img_f = self.decoder(yaw_f, s, bak)
        else:
            _img_f = self.decoder(yaw_f, mu, bak)
        loss_f = torch.sum((_img_f - img_f)**2) / len(img_f)

        if return_img:
            return _img
        elif return_img_f:
            return _img_f
        elif return_sample_img:
            yaw = [2 * np.pi * y / 16 for y in range(16)]
            yaw = [[np.cos(y), np.sin(y)] for y in yaw]
            yaw = torch.from_numpy(np.array(yaw).astype(np.float32)).to(self.device)
            aaa = bak[:1]
            aaa[0,0] = aaa[0,0] * 0.
            aaa[0,1] = aaa[0,1] * 0.
            aaa[0,2] = aaa[0,2] * 0. # + 0.1
            _img = self.decoder(yaw, mu[:1].expand(16, -1), aaa.expand(16, -1, -1, -1))
            return _img
        
        return loss + kl + loss_f

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
        return self.forward(feed_dict, return_img=True, train=False)
    
    def reconst_f(self, feed_dict):
        return self.forward(feed_dict, return_img_f=True, train=False)
    
    def reconst_phi(self, feed_dict, phi):
        return self.forward(feed_dict, return_img_f=True, train=False, phi=phi)
    
    def sample_img(self, feed_dict):
        return self.forward(feed_dict, return_sample_img=True, train=False)

    def sample_vid(self, feed_dict):
        vid = []
        for i in range(4):
            _img = self.sample_img(feed_dict)
            vid.append(_img)
        vid = torch.stack(vid)
        return vid


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc1 = nn.Linear(1024, 2)
        self.fc21 = nn.Linear(1024, 8)
        self.fc22 = nn.Linear(1024, 8)

    def forward(self, x):
        h = F.relu(self.conv1(x))  # 31x31
        h = F.relu(self.conv2(h))  # 14x14
        h = F.relu(self.conv3(h))  # 6x6
        h = F.relu(self.conv4(h))  # 2x2
        h = h.view(x.size(0), 1024)
        yaw = self.fc1(h)  # 1024 -> 2
        print(torch.norm(yaw))
        yaw = yaw / torch.norm(yaw)
        mu = self.fc21(h)
        logvar = self.fc22(h)
        s = self.reparameterize(mu, logvar)
        return yaw, (s, mu, logvar)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(.5).exp_()
        eps = std.clone().detach().new(std.size()).normal_()
        return eps.mul(std).add_(mu)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(2 + 8, 1024)
        self.conv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, yaw, s, bak):

        aaa = bak[:1]
        aaa[0,0] = aaa[0,0] * 0#  + 0.1
        aaa[0,1] = aaa[0,1] * 0#  + 0.1
        aaa[0,2] = aaa[0,2] * 0#  + 0.1
        bak = aaa
        
        h = torch.cat([yaw, s], 1)
        h = self.fc1(h)  # linear
        h = h.view(yaw.size(0), 1024, 1, 1)
        h = F.relu(self.conv1(h))  # 5x5
        h = F.relu(self.conv2(h))  # 13x13
        h = F.relu(self.conv3(h))  # 30x30
        h = self.conv4(h) + bak    # 64x64
        return torch.clamp(h, 0, 1)