from model import AE
from data_loader import PetDataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from tqdm import tqdm


device=0

model = AE(device)
dataloader = PetDataLoader(device)
writer = SummaryWriter()

resume_epoch = 1  # if not args.resume else args.resume_epoch

itr = 0
for epoch in range(resume_epoch, 5 + 1):
    print(epoch)
    for batch in tqdm(dataloader):
        loss = model.train_(batch)
        print(loss.item())
        writer.add_scalar("loss", loss.item(), itr)
        itr += 1
    recons = model.reconst(batch)
    recons = torch.cat([batch["img"][:32].to(device), recons[:32]])
    recons = torchvision.utils.make_grid(recons)
    sample = model.sample_img(batch)
    sample = torchvision.utils.make_grid(sample)
    video = model.sample_vid(batch)
    writer.add_image('recons', recons, itr)
    writer.add_image('sample', sample, itr)
    writer.add_video('video', video, itr)
    writer.add_scalar("loss", loss.item(), itr)