import cv2
import numpy as np

cap= cv2.VideoCapture(0)
distinct_frames=[]
comparison_frame =None
i=0
a= None
b= None

def matching(img1 , img2):
    if img1 is None or img2 is None:
        print('Could not open or find the images!')
        exit(0)

    minHessian = 400
    detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

    a1 = len(keypoints1)
    b1 = len(keypoints2)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)

    ratio_thresh = 0.7
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    c1 = len(good_matches)

    if (c1 / a1 > 0.2 and c1 / b1 > 0.2):
        return True
    else:
        return False



while(True):
    ret, frame =cap.read()
    cv2.imshow('frame', frame)
    if i==0:
        a=frame
        cv2.imwrite('image'+str(i)+'.jpg', a)
    elif i==1:
        b=frame
    else:
        a,b= b, frame
        image_matched= matching(a,b)
        if (image_matched== False):
            cv2.imwrite('image' + str(i) + '.jpg', a)
    i = i + 1

    if cv2.waitKey(1) & 0xFF== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

