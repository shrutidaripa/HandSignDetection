import time

import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector

cap=cv2.VideoCapture(0)
detector=HandDetector(maxHands=1)
offset=17
img_size=300
counter=0

folder="Data/B"

while True:

    success,img=cap.read()
    hands,img=detector.findHands(img)
    if hands:
        hand=hands[0] # only one hand
        x,y,w,h=hand['bbox'] # for the bounding box

        img_white= np.ones((img_size,img_size,3),np.uint8)*255

        img_crop=img[y-offset:y+h+offset,x-offset:x+w+offset]

        # img cropped on img white
        img_crop_shape=img_crop.shape

        # fix the empty white space getting wasted
        aspect_ratio=h/w
        if aspect_ratio>1:
            # for variable height
            k=img_size/h
            w_cal=math.ceil(k*w)
            img_resize=cv2.resize(img_crop,(w_cal,img_size))
            img_resize_shape = img_resize.shape
            # center the cropped img on white img
            w_gap=math.ceil((img_size-w_cal)/2)
            img_white[:, w_gap:w_cal+w_gap] = img_resize
        else:
            # fix the width
            k = img_size / w
            h_cal = math.ceil(k * h)
            img_resize = cv2.resize(img_crop, (img_size,h_cal))
            img_resize_shape = img_resize.shape
            # center the cropped img on white img
            h_gap = math.ceil((img_size - h_cal) / 2)
            img_white[ h_gap:h_cal + h_gap,:] = img_resize

        cv2.imshow("Image_Cropped",img_crop)
        cv2.imshow("Image_White",img_white)
    cv2.imshow("Image",img)

    key=cv2.waitKey(1)
    if key==ord("s"):
        counter+=1
        cv2.imwrite(f'{folder}/image_{time.time()}.jpg',img_white)
        print(counter)
    # exit part
    # if cv2.waitKey(10) & 0xFF==ord('q'):
    #     break
cap.release()
cv2.destroyAllWindows()

