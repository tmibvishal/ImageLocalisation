"""matcher.py

Functions in this file allows the user to match 2 images
and get the fraction match between 2 images

Accepts only Mat (The Basic Image Container) format images
"""

import cv2
import numpy as np
import scipy
import general
import time

def cos_cdist(self, des1, des2):
    # getting cosine distance between search image and images database
    v = des1.reshape(1, -1)
    return scipy.spatial.distance.cdist(np.array([des2]), v, 'cosine').reshape(-1)


def KAZE_match(des1, des2):
    # reference: https://medium.com/machine-learning-world/feature-extraction-and-similar-image-search-with-opencv-for-newbies-3c59796bf774
    img_distances = cos_cdist(des1, des2)
    not_match = img_distances[0]
    print('Match', (1 - not_match))
    return 1 - not_match


def SURF_match_2(key_des_1, key_des_2, hessianThreshold: int = 400, ratio_thresh: float = 0.7,
                 symmetry_match: bool = True):
    """Give fraction match between 2 images descriptors using SURF and FLANN

    Parameters
    ----------
    key_des_1 : (length of keypoints, description) pair of image 1,
    key_des_2 : (length of keypoints, description) pair of image 2,
    hessianThreshold: Number of SURF points to consider in a image,
    ratio_thresh: (b/w 0 to 1) lower the number more serious the matching,
    symmetry_match: if symmetry_match then order of key_des_1 and 2 does not matter but slow

    Returns
    -------
    float,
        returns a number from 0 to 1 depending on percentage match and returns -1 if any of the parameter is None
    """
    if key_des_1 is None or key_des_2 is None:
        raise Exception("key_des_1 or key_des_1 can't be none")
        return -1

    if ratio_thresh > 1 or ratio_thresh < 0:
        raise Exception("ratio_thresh not between 0 to 1")
        return -1

    len_keypoints1, descriptors1 = key_des_1
    len_keypoints2, descriptors2 = key_des_2

    a1 = len_keypoints1
    b1 = len_keypoints2

    if a1 < 2 or b1 < 2:
        return 0

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    c1 = len(good_matches)

    if (symmetry_match):
        knn_matches = matcher.knnMatch(descriptors2, descriptors1, 2)
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        c2 = len(good_matches)
        fraction = (c1 + c2) / (a1 + b1)
        return fraction,(c1+c2)/2

    fraction = (2.0 * c1) / (a1 + b1)
    if (fraction > 1): fraction = 1
    # fraction can be greater than one in blur images because we are multiplying fraction with 2
    return fraction, c1


def SURF_match(img1, img2, hessianThreshold: int = 400, ratio_thresh: float = 0.7, symmetry_match: bool = True):
    """Give fraction match between 2 images using SURF and FLANN

    Parameters
    ----------
    img1 : Open CV image format,
    img2 : Open CV image format,
    hessianThreshold: Number of ORB points to consider in a image,
    ratio_thresh: (b/w 0 to 1) lower the number more serious the matching
    symmetry_match:

    Returns
    -------
    float,
        returns a number from 0 to 1 depending on percentage match and returns -1 if any of the parameter is None
    """
    if img1 is None or img2 is None:
        raise Exception("img1 or img2 can't be none")

    if ratio_thresh > 1 or ratio_thresh < 0:
        raise Exception("ratio_thresh not between 0 to 1")

    detector = cv2.xfeatures2d_SURF.create(hessianThreshold)
    keypoints1, descriptors1 = detector.detectAndCompute(img1, None)
    keypoints2, descriptors2 = detector.detectAndCompute(img2, None)

    a1 = len(keypoints1)
    b1 = len(keypoints2)

    if a1 < 2 or b1 < 2:
        return 0

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)
    c1 = len(good_matches)

    img3 = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
    cv2.drawMatches(
        img1, keypoints1, img2, keypoints2, good_matches, outImg=img3, matchColor=None, flags=2)
    if img3 is not None:
        cv2.namedWindow("matches", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("matches", 1600, 1600)
        cv2.imshow("matches", img3)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if symmetry_match:
        knn_matches = matcher.knnMatch(descriptors2, descriptors1, 2)
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        c2 = len(good_matches)

        img3 = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
        cv2.drawMatches(
            img2, keypoints2, img1, keypoints1, good_matches, outImg=img3, matchColor=None, flags=2)
        if img3 is not None:
            cv2.namedWindow("matches", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("matches", 1600, 1600)
            cv2.imshow("matches", img3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print(a1,"a1", b1,"b1", c1, "c1", c2, "c2")
        fraction = (c1 + c2) / (a1 + b1)
        print(fraction, "fraction between images matched")
        return fraction

    fraction = (2.0 * c1) / (a1 + b1)
    if fraction > 1:
        fraction = 1
    # fraction can be greater than one in blur images because we are multiplying fraction with 2
    print(fraction, "fraction between images matched")
    return fraction


def ORB_match(img1, img2, hessianThreshold: int = 400, ratio_thresh: float = 0.7):
    """Give fraction match between 2 images using ORB and Brute Force Matching

    Parameters
    ----------
    img1 : Open CV image format,
    img2 : Open CV image format,
    hessianThreshold: Number of ORB points to consider in a image,
    ratio_thresh: (b/w 0 to 1) lower the number more serious the matching

    Returns
    -------
    float,
        returns a number from 0 to 1 depending on percentage match and returns -1 if any of the parameter is None
    """
    if img1 is None or img2 is None:
        raise Exception("img1 or img2 can't be none")
        return -1

    if ratio_thresh > 1 or ratio_thresh < 0:
        raise Exception("ratio_thresh not between 0 to 1")
        return -1

    orb = cv2.ORB_create(hessianThreshold)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    a1 = len(keypoints1)
    b1 = len(keypoints2)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descriptors1, trainDescriptors=descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append([m])

    c1 = len(good_matches)

    return (2.0 * c1) / (a1 + b1)


def SURF_returns(kp_des_1, kp_des_2, hessianThreshold: int = 400, ratio_thresh: float = 0.7,
                 symmetry_match: bool = True,
                 max_slope=0.2, check_c1_c2: bool = True):
    """Give fraction match between 2 images using SURF and FLANN

    Parameters
    ----------
    img1 : Open CV image format,
    img2 : Open CV image format,
    hessianThreshold: Number of ORB points to consider in a image,
    ratio_thresh: (b/w 0 to 1) lower the number more serious the matching
    symmetry_match
    max_slope : Ensures |slope| of line connecting matching pts is less than max_slope
    check_c1_c2 : Ensures c1, c2 are less than 50% deviated from each other in symmetry_match

    Returns
    -------
    float,
        returns a number from 0 to 1 depending on percentage match and returns -1 if any of the parameter is None
    """
    # if img1 is None or img2 is None:
    #     raise Exception("img1 or img2 can't be none")
    #
    # if ratio_thresh > 1 or ratio_thresh < 0:
    #     raise Exception("ratio_thresh not between 0 to 1")
    #
    # detector = cv2.xfeatures2d_SURF.create(hessianThreshold)
    # keypoints1, descriptors1,  = detector.detectAndCompute(img1, None)
    # keypoints2, descriptors2 = detector.detectAndCompute(img2, None)
    #
    # import video_operations_3 as vo3
    # kp1 = vo3.serialize_keypoints(keypoints1)
    # kp2 = vo3.serialize_keypoints(keypoints2)
    #
    # shape1 = img1.shape
    # shape2 = img2.shape
    #
    # a1 = len(kp1)
    # b1 = len(kp2)

    a1, descriptors1, kp1, shape1 = kp_des_1
    b1, descriptors2, kp2, shape2 = kp_des_2

    if a1 < 2 or b1 < 2:
        return -1, None

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
    # matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
    good_matches=[]
    for m, n in knn_matches:
        if  m.distance < ratio_thresh * n.distance:

            # Calculation of slope
            img2_idx = m.trainIdx
            img1_idx = m.queryIdx
            (x1, y1) = kp1[img1_idx][0]
            (x2_rel, y2_rel) = kp2[img2_idx][0]
            (x2, y2) = (x2_rel + shape1[1], y2_rel)
            if x2 < x1:
                raise Exception("x1 somehow greater than x2")
            elif x2 == x1:
                continue
            else:
                slope = (y1 - y2) / (x2 - x1)  # Since y is measured from upper edge of frame

            if abs(slope) > max_slope:
                continue
            # Appending to good_matches
            good_matches.append(m)

            # Testing ( one match )

            # print(slope)
            # img3 = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
            # cv2.drawMatches(
            #     img1, keypoints1, img2, keypoints2, [m], outImg=img3, matchColor=None, flags=2)
            # if img3 is not None:
            #     cv2.imshow("matches", img3)
            #     cv2.waitKey(0)
            #     cv2.destroyAllWindows()

    c1 = len(good_matches)
    #print("no of matches ", c1)

   # Testing ( all matches )
   #  img3 = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
   #  cv2.drawMatches(
   #      img1, keypoints1, img2, keypoints2, good_matches, outImg=img3, matchColor=None, flags=2)
   #  if img3 is not None:
   #      cv2.imshow("matches", img3)
   #      cv2.waitKey(0)
   #      cv2.destroyAllWindows()

    if symmetry_match:
        knn_matches = matcher.knnMatch(descriptors2, descriptors1,2)
        good_matches = []
        for m,n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                # Calculation of slope
                img1_idx = m.trainIdx
                img2_idx = m.queryIdx
                (x2, y2) = kp2[img2_idx][0]
                (x1_rel, y1_rel) = kp1[img1_idx][0]
                (x1, y1) = (x1_rel + shape2[1], y1_rel)
                if x1 < x2:
                    raise Exception("x2 somehow greater than x1")
                elif x2 == x1:
                    continue
                else:
                    slope = (y1 - y2) / (x2 - x1)  # Since y is measured from upper edge of frame

                if abs(slope) > max_slope:
                    continue
                # Appending to good_matches
                good_matches.append(m)

        c2 = len(good_matches)
        # print("no of matches ",c2)
        #
        # img3 = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
        # cv2.drawMatches(
        #     img2, keypoints2, img1, keypoints1, good_matches, outImg=img3, matchColor=None, flags=2)
        # if img3 is not None:
        #     cv2.imshow("matches", img3)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        if check_c1_c2:
            if c2 == 0 or not 0.5 <= c1 / c2 <= 2:
                # print("******\nDiff btw c1 and c2!\n******")
                fraction = 2*min(c1, c2)/(a1+b1)
                return fraction, min(c1,c2)
        fraction = (c1 + c2) / (a1 + b1)
        return fraction, min(c1,c2)

    if c1 > b1:
        print("******\nc1 greater than b1, so returning zero\n*********")
        return -1, c1
    fraction = (2.0 * c1) / (a1 + b1)
    return fraction, c1

# img2 = cv2.imread("edge_data/edge_1_2/jpg/image104.jpg")
# b = SURF_match(img1, img2)
# a = SURF_returns(img1, img2)
# print(a)
# print(b)
# img1 = cv2.imread("testData/night sit 0 june 18/graph obj vishal/edge_data/edge_0_1/jpg/image106.jpg")
# gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# # img2= cv2.imread("edge_data/edge_0_1/jpg/image55.jpg", 0)
# # SURF_match(img1, img2, 2500)
# detector = cv2.xfeatures2d_SURF.create(2500)
# keypoints1, descriptors1 = detector.detectAndCompute(gray, None)
# print(len(keypoints1))
# img = cv2.drawKeypoints(gray, keypoints1, None)
#
# cv2.imshow("Image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# b, _, _ = cv2.split(img1)
# a =cv2.Laplacian(b, cv2.CV_64F).var()
# print(a)

# imgobj1 = general.load_from_memory("node_data/node_0/image52.pkl")
# imgobj2 = general.load_from_memory("edge_data/edge_0_1/image52.pkl")
# param1 = imgobj1.get_elements()
# param2 = imgobj2.get_elements()
# start = time.time()
# i=0
# while True:
#     fraction = SURF_returns(param1, param2)
#     elapsed = time.time() - start
#     i = i+1
#     if elapsed >= 1:
#         break
# print("Matched: "+str(i))
# print(elapsed)
