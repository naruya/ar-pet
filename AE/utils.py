import os
import torch
from datetime import datetime
from torch import nn


def save_model(model, name):
    time = datetime.now().strftime("%b%d_%H-%M-%S")
    path = os.path.join("model", name, time)
    os.makedirs(path, exist_ok=True)
    for i, dist in enumerate(model.distributions):
        torch.save(dist.state_dict(), os.path.join(path, "dist" + str(i) + ".pt"))
    torch.save(model.optimizer.state_dict(), os.path.join(path, "opt.pt"))
    return time


def load_model(model, name, time):
    path = os.path.join("model", name, time)
    for i, dist in enumerate(model.distributions):
        dist.load_state_dict(torch.load(os.path.join(path, "dist" + str(i) + ".pt")))
    model.optimizer.load_state_dict(torch.load(os.path.join(path, "opt.pt")))


# https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
def init_weights(model):
    print("---- init weights ----")
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.normal_(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.RNN, nn.RNNCell, nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.orthogonal_(param.data)
                else:
                    nn.init.normal_(param.data)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        else:
            print("  ", type(m))
            continue
        print("ok", type(m))