# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

# built-in modules
import itertools as it
from multiprocessing.pool import ThreadPool

# local modules
# from common import Timer
from find_obj import init_feature, filter_matches

def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv.warpAffine(img, A, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv.warpAffine(mask, A, (w, h), flags=cv.INTER_NEAREST)
    Ai = cv.invertAffineTransform(A)
    return img, mask, Ai


def affine_detect(detector, img, mask=None, pool=None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transformations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''
    params = [(1.0, 0.0)]
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print('affine sampling: %d / %d\r' % (i+1, len(params)), end='')
        keypoints.extend(k)
        descrs.extend(d)

    print()
    return keypoints, np.array(descrs)



def matcher(img1 , img2, feature_name= "surf-flann",symmetry_match: bool = True):

    if img1 is None or img2 is None:
        raise Exception("img1 or img2 can't be none")
        return -1

    img1 = cv.imread(img1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(img2, cv.IMREAD_GRAYSCALE)
    detector, matcher = init_feature(feature_name)
    if detector is None:
        print('unknown feature:', feature_name)
        return -1

    print('using', feature_name)

    pool = ThreadPool(processes=cv.getNumberOfCPUs())
    kp1, desc1 = affine_detect(detector, img1, pool=pool)
    kp2, desc2 = affine_detect(detector, img2, pool=pool)
    print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))

    k1 = len(kp1)
    k2 = len(kp2)

    if k1 < 2 or k1 < 2:
        return 0

    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
    a1 = len(kp_pairs)

    if len(p1) >= 4:
        H, status = cv.findHomography(p1, p2, cv.RANSAC, 5.0)
        # print('%d / %d  inliers/matched' % (np.sum(status1), len(status1)))
        # do not draw outliers (there will be a lot of them)
        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
        b1 = len(kp_pairs)
    else:
        H, status = None, None
        print('%d matches found, not enough for homography estimation' % len(p1))

    if (symmetry_match):
        raw_matches1 = matcher.knnMatch(desc2, trainDescriptors=desc1, k=2)
        p3, p4, kp_pairs1 = filter_matches(kp2, kp1, raw_matches1)
        a2 = len(kp_pairs1)

        if len(p3) >= 4:
            H1, status1 = cv.findHomography(p3, p4, cv.RANSAC, 5.0)
            # print('%d / %d  inliers/matched' % (np.sum(status1), len(status1)))
            # do not draw outliers (there will be a lot of them)
            kp_pairs1 = [kpp for kpp, flag in zip(kp_pairs1, status1) if flag]
            b2= len(kp_pairs1)
        else:
            H1, status1 = None, None
            print('%d matches found, not enough for homography estimation' % len(p3))

        fraction =(b1+b2)/(a1+a2)
        return fraction

    fraction = (b1) / (a1)
    return fraction

percentage = matcher("testData/testing/image0 (1).jpg","testData/testing/image0.jpg", "brisk-flann")
print(percentage)
