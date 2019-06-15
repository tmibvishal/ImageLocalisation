import urllib
import cv2
import numpy as np
import urllib.request

url = 'http://10.194.43.76:8080/shot.jpg'

while True:
    imgResp = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)

    # all the opencv processing is done here
    cv2.imshow('test', img)
    if ord('q') == cv2.waitKey(10):
        exit(0)
