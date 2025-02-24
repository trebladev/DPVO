import os
import cv2
import numpy as np
from multiprocessing import Process, Queue
from pathlib import Path

def image_stream(queue, imagedir, calib, stride, skip=0, fisheye=False):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(Path(imagedir).glob('*.png'))[skip::stride]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(str(imfile))
        if fisheye:
            h,w = image.shape[:2]
            DIM = (h,w)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, calib[4:], np.eye(3), K, DIM, cv2.CV_16SC2)
            undistorted_img = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            image = undistorted_img
            
            intrinsics = np.array([fx, fy, cx, cy])
        else:
            if len(calib) > 4:
                image = cv2.undistort(image, K, calib[4:])
                # show_image(image)
    
            if 0:
                image = cv2.resize(image, None, fx=0.5, fy=0.5)
                intrinsics = np.array([fx / 2, fy / 2, cx / 2, cy / 2])
    
            else:
                intrinsics = np.array([fx, fy, cx, cy])
                
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]
    
        queue.put((t, image, intrinsics))
    
    queue.put((-1, image, intrinsics))


def video_stream(queue, imagedir, calib, stride, skip=0):
    """ video generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    cap = cv2.VideoCapture(imagedir)

    t = 0

    for _ in range(skip):
        ret, image = cap.read()

    while True:
        # Capture frame-by-frame
        for _ in range(stride):
            ret, image = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break

        if not ret:
            break

        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w, _ = image.shape
        image = image[:h-h%16, :w-w%16]

        intrinsics = np.array([fx*.5, fy*.5, cx*.5, cy*.5])
        queue.put((t, image, intrinsics))

        t += 1

    queue.put((-1, image, intrinsics))
    cap.release()

