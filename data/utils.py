import cv2

def vread(path, T=9999):
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
            if i % 300 == 0:
                print(i, end=" ")
        else:
            cap.release()
    print()
    return vid