"""videoOperations.py

Contains functions that operate on video or stream of images
"""

import cv2
import numpy as np
import matcher as mt
import os
# path = './Images'

cap = cv2.VideoCapture("VID_20190516_235550.mp4")
# cap= cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,50)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,50)

distinct_frames=[]
comparison_frame =None
i=0
a= None
b= None


ret, frame = cap.read()
a = frame
cv2.imwrite('image' + str(i) + '.jpg', b)

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame', frame)
    
    b = frame
    image_fraction_matched = mt.ORB_match(a,b,2500,0.7)       
    if (image_fraction_matched < 0.1):
        cv2.imwrite('image' + str(i) + '.jpg', a)
        a = b

    i = i + 1

    if cv2.waitKey(1) & 0xFF== ord('q'):
        break

print(i)
cap.release()
cv2.destroyAllWindows()