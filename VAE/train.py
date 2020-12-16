## TODO
## auxの数がおかしい（img, bakと合ってない）
## auxをdataloadする
## data augumentation(imgとbakに同じ変換をかます)


from model import AE
from data_loader import PetDataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from tqdm import tqdm
from utils import *


device=0


import os
from datetime import datetime
import subprocess

cmd = "git rev-parse --short HEAD"
ghash = subprocess.check_output(cmd.split()).strip().decode('utf-8')

log_dir = os.path.join(
    "./runs",
    datetime.now().strftime("%b%d_%H-%M-%S")
    + "_" + ghash
)

model = AE(device)
dataloader = PetDataLoader(device)
writer = SummaryWriter()

resume_epoch = 1  # if not args.resume else args.resume_epoch

itr = 0
for epoch in range(resume_epoch, 10 + 1):
    print(epoch)
    for batch in tqdm(dataloader):
        loss = model.train_(batch)
        print(loss.item())
        writer.add_scalar("loss", loss.item(), itr)
        itr += 1

    recons = model.reconst(batch)
    recons = torch.cat([batch["img"][:32].to(device), recons[:32]])
    recons = torchvision.utils.make_grid(recons)
    
    recons_f = model.reconst_f(batch)
    recons_f = torch.cat([batch["img"][:32].to(device), recons_f[:32]])
    recons_f = torchvision.utils.make_grid(recons_f)
    
    sample = model.sample_img(batch)
    sample = torchvision.utils.make_grid(sample)

    video = model.sample_vid(batch)

    writer.add_image('recons', recons, itr)
    writer.add_image('recons_f', recons_f, itr)
    writer.add_image('sample', sample, itr)
    writer.add_video('video', video, itr)
    writer.add_scalar("loss", loss.item(), itr)
    
    time = save_model(model, log_dir.split("/")[-1])
    load_model(model, log_dir.split("/")[-1], time)