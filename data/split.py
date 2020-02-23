# python split.py --path raw/GH010005.mp4 --min 18
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=None)
parser.add_argument("--min", type=int, default=None)

args = parser.parse_args()

inp = args.path

for i in range(args.min):
    out = (inp[:-4] + "-{:02}.mp4".format(i)).replace("raw", "split")
    cmd = "ffmpeg -ss {} -i {} -t 60 {}".format(i * 60, inp, out)
    print(cmd)
    subprocess.run(cmd.split())