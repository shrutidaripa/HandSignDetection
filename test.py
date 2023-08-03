import time

import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


cap=cv2.VideoCapture(0)

detector=HandDetector(maxHands=1)
classifier=Classifier("model/keras_model.h5","model/labels.txt")
# Add more labels !
labels=["A","B","C"]

offset=17
img_size=300
counter=0



while cap.isOpened():

    success,img=cap.read()
    img_to_show=img.copy()
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
            prediction,index=classifier.getPrediction(img_white,draw=False)
            print(prediction,index)

        else:
            # fix the width
            k = img_size / w
            h_cal = math.ceil(k * h)
            img_resize = cv2.resize(img_crop, (img_size,h_cal))
            img_resize_shape = img_resize.shape
            # center the cropped img on white img
            h_gap = math.ceil((img_size - h_cal) / 2)
            img_white[ h_gap:h_cal + h_gap,:] = img_resize
            prediction, index = classifier.getPrediction(img_white,draw=False)

        cv2.putText(img_to_show,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)
        cv2.rectangle(img_to_show,(x-offset,y-offset),(x+w+offset,y+h+offset),(0,0,255),2)

        cv2.imshow("Image_Cropped",img_crop)
        cv2.imshow("Image_White",img_white)
    cv2.imshow("Image",img_to_show)


    # exit part
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

=