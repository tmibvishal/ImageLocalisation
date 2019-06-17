import cv2
import numpy as np



img1 = cv2.imread("testData/testing1.jpg", cv2.IMREAD_COLOR)
img2 = cv2.imread("testData/testing2.jpg", cv2.IMREAD_COLOR)

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create()
orb = cv2.ORB_create(5000)

kp1, des1 = surf.detectAndCompute(img1, None)
# second parameter: None is for mask, a image is colorful and if grayscale at some mask then that it the mask
kp2, des2 = surf.detectAndCompute(img2, None)

#Note: tried using orb above but it didn't work

# feature matching
FLANN_INDEX_LSH = 0
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
matches = matcher.knnMatch(des1, des2, k=2)
# matches = flann.knnMatch(des1,des2,k=2)



#only good matches
good_matches = []
for m,n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

matchingresult = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=0)

# img = cv2.drawKeypoints(img1, kp1, None)

cv2.imshow("Image", matchingresult)
cv2.waitKey(1)
cv2.destroyAllWindows()