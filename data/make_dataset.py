import glob
from utils import vread
import numpy as np
import cv2
from copy import deepcopy
import os
import moviepy.editor as mpy
import pickle
from multiprocessing import Pool

# file: "ar-pet/data/split/hoge.mp4"


def make_dataset(file, skip=None):
    vid = vread(file)

    back_file = file.replace("split/", "back/").replace("mp4", "png")
    if os.path.exists(back_file):
        back = cv2.imread(back_file)
    else:
        back = np.array(vid).mean(0).astype(np.uint8)
        cv2.imwrite(back_file, back)
    H, W, C = back.shape

    # threshs = []
    # for t in range(2, len(vid)-2):
    #     if not t % 15 == 0:
    #         continue
    #     img = np.array(vid[t-2:t+3]).mean(0).astype(np.uint8)  # PARAM
    #     diff_img = cv2.absdiff(back, img)
    #     a = np.sum(diff_img, 2).reshape(-1)
    #     threshs.append(int(np.percentile(a, 99)))
    # thresh = int(np.percentile(threshs, 50))
    thresh = 18

    imgs = []
    backs = []
    auxs = []
    for t in range(2, len(vid)-2):
        if skip and not t % skip == 0:
            continue
        if t % 30 == 0:
            print(t)
        img = np.array(vid[t-2:t+3]).mean(0).astype(np.uint8)  # PARAM
        diff_img = cv2.absdiff(back, img)
        diff_all = np.sum(diff_img)

        img = np.sum(diff_img, 2)
        img = cv2.blur(img,(16,16))  # PARAM
        img = np.where(img < thresh, 0, 1).astype(np.uint8)  # PARAM  # 12 ~ 21(しっぽが見えるギリギリライン)

        contours, hierarchy = cv2.findContours(deepcopy(img), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)
        img = cv2.drawContours(deepcopy(vid[t]), contours, -1, (0,255,0), 10)

        best_i = None
        best_diff_mean = 0
        best_s = 0
        for j, c in enumerate(contours[:5]):  # PARAM
            mask = cv2.fillConvexPoly(deepcopy(diff_img), c, (0,0,0))
            s = cv2.contourArea(c)
            if s <= 5000:
                continue
            diff_mean = (diff_all - np.sum(mask)) / s
            if best_diff_mean < diff_mean:
                best_diff_mean = diff_mean
                best_i = j
                best_s = s
        if best_i == None or best_diff_mean * best_s < 500000 or best_diff_mean <= thresh*2:
            continue
        mask = cv2.fillConvexPoly(deepcopy(vid[t]), contours[best_i], (0,0,0))  # TODO きれいじゃない

        x,y,w,h = cv2.boundingRect(contours[best_i])
        img = cv2.rectangle(deepcopy(vid[t]),(x,y),(x+w,y+h),(0,255,0),10)

        margin = 128
        if w > h:
            margin_x = margin
            margin_y = margin + int((w-h)/2)
        else:
            margin_y = margin
            margin_x = margin + int((h-w)/2)

        if 0 <= y-margin_y and y+h+margin_y < H and 0 <= x-margin_x and x+w+margin_x < W:
            img_t = deepcopy(vid[t])[y-margin_y:y+h+margin_y, x-margin_x:x+w+margin_x]
            img_t = cv2.resize(img_t, (64, 64))
            imgs.append(img_t)
            back_t = back[y-margin_y:y+h+margin_y, x-margin_x:x+w+margin_x]
            back_t = cv2.resize(back_t, (64, 64))
            backs.append(back_t)
            auxs.append([t, x, y, w, h])

    if len(imgs) > 0:
        imgs_path = file.replace("split/", "dataset/img/").replace("mp4", "gif")
        clip = mpy.ImageSequenceClip(imgs, fps=30*2)
        clip.write_gif(imgs_path, fps=30*2)
        backs_path = file.replace("split/", "dataset/back/").replace("mp4", "gif")
        clip = mpy.ImageSequenceClip(backs, fps=30*2)
        clip.write_gif(backs_path, fps=30*2)
        pickle_path = file.replace("split/", "dataset/aux/").replace("mp4", "pkl")
        with open(pickle_path, mode='wb') as f:
            pickle.dump(auxs, f)


if __name__ == "__main__":
    files = sorted(glob.glob("split/*.mp4"))
    with Pool(processes=16) as pool:  # use 230GB RAM
        pool.map(make_dataset, files)