"""matcher.py

Functions in this file allows the user to match 2 images 
and get the fraction match between 2 images

Accepts only Mat (The Basic Image Container) format images
"""

import cv2
import numpy as np
import scipy


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


def SURF_match_2(key_des_1,key_des_2, hessianThreshold: int = 400, ratio_thresh: float = 0.7, symmetry_match: bool = True):
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

    if(symmetry_match):
        knn_matches = matcher.knnMatch(descriptors2, descriptors1, 2)
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        c2 = len(good_matches)
        fraction = (c1 + c2) / (a1 + b1)
        return fraction

    fraction = (2.0 * c1) / (a1 + b1)
    if(fraction > 1): fraction = 1
    # fraction can be greater than one in blur images because we are multiplying fraction with 2
    return fraction

def SURF_match(img1, img2, hessianThreshold: int = 400, ratio_thresh: float = 0.7, symmetry_match: bool = True):
    """Give fraction match between 2 images using SURF and FLANN

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

    if(symmetry_match):
        knn_matches = matcher.knnMatch(descriptors2, descriptors1, 2)
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        c2 = len(good_matches)
        fraction = (c1 + c2) / (a1 + b1)
        return fraction

    fraction = (2.0 * c1) / (a1 + b1)
    if(fraction > 1): fraction = 1
    # fraction can be greater than one in blur images because we are multiplying fraction with 2
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

