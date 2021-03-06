import cv2

def vread(path, T=9999, mute=True):
    vid = []
    cap = cv2.VideoCapture(path)
    i = 1
    while(cap.isOpened()):
        if i > T:
            break
        ret, frame = cap.read()
        if ret:
            vid.append(frame[:,:,::-1])
            i += 1
            if i % 300 == 0 and not mute:
                print(i, end=" ")
        else:
            cap.release()
    if not mute: print()
    return vid