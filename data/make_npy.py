from utils import vread
import glob
from itertools import chain
import numpy as np

img_files = sorted(glob.glob("dataset/img/*.gif"))
bak_files = sorted(glob.glob("dataset/bak/*.gif"))

imgs = [vread(file) for file in img_files]
baks = [vread(file) for file in bak_files]

imgs = np.array(list(chain.from_iterable(imgs)))
baks = np.array(list(chain.from_iterable(baks)))

print(imgs.shape, baks.shape)

np.save("dataset/imgs.npy", imgs)
np.save("dataset/baks.npy", baks)