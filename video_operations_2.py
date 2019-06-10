"""video_operations.py
Contains functions that operate on video or stream of images
"""

import cv2
import numpy as np
import matcher as mt
import os
import time
import pickle
from general import *


class ImgObj:
    def __init__(self, no_of_keypoints, descriptors, time_stamp):
        self.no_of_keypoints = no_of_keypoints
        self.descriptors = descriptors
        self.time_stamp = time_stamp

    def get_elements(self):
        return (self.no_of_keypoints, self.descriptors)

    def get_time(self):
        return self.time_stamp


class DistinctFrames:
    def __init__(self):
        self.img_objects = []
        self.time_of_path = None

    def add_img_obj(self, img_obj):
        if not isinstance(img_obj, ImgObj):
            raise Exception("Param is not an img object")
        self.img_objects.append(img_obj)

    def add_all(self, list_of_img_objects):
        if isinstance(list_of_img_objects, list):
            if (len(list_of_img_objects) != 0):
                if isinstance(list_of_img_objects[0], ImgObj):
                    self.img_objects = list_of_img_objects
                    return
            else:
                self.img_objects = list_of_img_objects
                return
        raise Exception("Param is not a list of img objects")

    def calculate_time(self):
        if len(self.img_objects) != 0:
            start_time = self.img_objects[0].time_stamp
            end_time = self.img_objects[-1].time_stamp
            if isinstance(start_time, int) and isinstance(end_time, int):
                self.time_of_path = end_time - start_time
                return
        raise Exception("Error in calculating time of path")

    def get_time(self):
        if self.time_of_path is None:
            self.calculate_time()
        return self.time_of_path

    def no_of_frames(self):
        return len(self.img_objects)

    def get_objects(self, start_index=0, end_index=-1):
        if (start_index == 0 and end_index == -1):
            return self.img_objects[start_index:end_index]
        if (start_index or end_index) not in range(0, self.no_of_frames()):
            raise Exception("Invalid start / end indexes")
        if start_index > end_index:
            raise Exception("Start index should be less than or equal to end index")
        return self.img_objects[start_index:end_index]

    def get_object(self, index):
        if index not in range(0, self.no_of_frames()):
            raise Exception("Invalid index")
        return self.img_objects[index]


def variance_of_laplacian(image):
    """Compute the Laplacian of the image and then return the focus measure, 
    which is simply the variance of the Laplacian

    Parameters
    ----------
    image : image object (mat)

    Returns
    -------
    int,
        returns higher value if image is not blurry otherwise returns lower value

    Referenece
    -------
    https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()


def is_blurry_colorful(image):
    """Check if the image passed is blurry or not

    Parameters
    ----------
    image : image object (mat)

    Returns
    -------
    bool,
        returns True if image is blurry otherwise returns False
    """
    b, _, _ = cv2.split(image)
    a = variance_of_laplacian(b)
    return (variance_of_laplacian(b) < 100)


def is_blurry_grayscale(gray_image):
    a = variance_of_laplacian(gray_image)
    return (variance_of_laplacian(gray_image) < 100)


def save_distinct_ImgObj(video_str, folder, frames_skipped: int = 0, check_blurry: bool = False,
                         hessian_threshold: int = 2500):
    """Saves non redundent and distinct frames of a video in folder
    Parameters
    ----------
    video_str : is video_str = "webcam" then loads webcam. O.W. loads video at video_str location,
    folder : folder where non redundant images are to be saved,
    frames_skipped: Number of frames to skip and just not consider,
    check_blurry: If True then only considers non blurry frames but is slow

    Returns
    -------
    array,
        returns array contaning non redundant frames(mat format)
    """

    ensure_path(folder + "/jpg")

    frames_skipped += 1

    if video_str == "webcam":
        video_str = 0
    cap = cv2.VideoCapture(video_str)
    # cap= cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)

    distinct_frames = DistinctFrames()
    i = 0
    a = None
    b = None
    check_next_frame = False

    detector = cv2.xfeatures2d_SURF.create(hessian_threshold)

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    a = (len(keypoints), descriptors)
    img_obj = ImgObj(a[0], a[1], i)
    save_to_memory(img_obj, 'image' + str(i) + '.pkl', folder)
    cv2.imwrite(folder + '/jpg/image' + str(i) + '.jpg', gray)
    distinct_frames.add_img_obj(img_obj)

    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if i % frames_skipped != 0 and not check_next_frame:
                i = i + 1
                continue

            cv2.imshow('frame', gray)
            print(i)

            if check_blurry:
                if is_blurry_grayscale(gray):
                    check_next_frame = True
                    i = i + 1
                    continue
                check_next_frame = False

            keypoints, descriptors = detector.detectAndCompute(gray, None)
            b = (len(keypoints), descriptors)
            image_fraction_matched = mt.SURF_match_2((a[0], a[1]), (b[0], b[1]), 2500, 0.7, False)
            if image_fraction_matched < 0.1:
                img_obj2 = ImgObj(b[0], b[1], i)
                save_to_memory(img_obj2, 'image' + str(i) + '.pkl', folder)
                cv2.imwrite(folder + '/jpg/image' + str(i) + '.jpg', gray)
                distinct_frames.add_img_obj(img_obj2)
                a = b

            i = i + 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    distinct_frames.calculate_time()
    return distinct_frames


def read_images(folder):
    """Reads images of the form "image<int>.pkl" from folder(passed as string containing 
    relative path of the specific folder)

    Parameters
    ----------
    folder: name of the folder

    Returns
    -------
    array,
        distinct_frames : a list containing tuples of the form
        (time_stamp, frame, len_keypoints, descriptors) where time_stamp is the <int> part of
        image<int>.pkl and frame is object of the image created using imread
    """
    distinct_frames = DistinctFrames()

    for file in sorted(sorted(os.listdir(folder)),
                       key=len):  # sorting files on basis of 1) length and 2) numerical order
        '''
            Sorting is done 2 times because
            if files in the folder are
                1. image100.pkl
                2. image22.pkl
                3. image21.pkl
            firstly sort them to image100.pkl,image21.pkl,image22.pkl then according to length to image21.pkl,image22.pkl,image100.pkl
        '''
        try:
            img_obj = load_from_memory(file, folder)
            time_stamp = img_obj.get_time()
            distinct_frames.add_img_obj(img_obj)
            print("Reading image .." + str(time_stamp) + " from " + folder)  # for debug purpose
        except:
            # exception will occur for files like .DS_Store and jpg directory
            continue

    if distinct_frames.no_of_frames() != 0:
        distinct_frames.calculate_time()

    return distinct_frames


def read_images_jpg(folder, hessian_threshold: int = 2500, for_nodes=False, folder_to_save:str=None):
    """Reads images of the form "image<int>.jpg" from folder(passed as string containing
    relative path of the specific folder)

    Parameters
    ----------
    folder
    hessian_threshold
    for_nodes
    folder_to_save

    Returns
    -------
    array,
        distinct_frames : a list containing tuples of the form (time_stamp, frame)
    where time_stamp is the <int> part of image<int>.jpg and frame is object of the
    image created using imread
    """
    distinct_frames = DistinctFrames()
    detector = cv2.xfeatures2d_SURF.create(hessian_threshold)

    if not for_nodes:
        for file in sorted(sorted(os.listdir(folder)), key=len):  # sorting files on basis of
            # 1) length and 2) numerical order
            """
                Sorting is done 2 times because
                if files in the folder are
                    1. image100.jpg
                    2. image22.jpg
                    3. image21.jpg
                firstly sort them to image100.jpg,image21.jpg,image22.jpg then according to length to 
                image21.jpg,image22.jpg,image100.jpg
            """
            try:
                grey = cv2.imread(folder + "/" + file, 0)
                time_stamp = int(file.replace('image', '').replace('.jpg', ''), 10)
                keypoints, descriptors = detector.detectAndCompute(grey, None)
                img_obj = ImgObj(len(keypoints), descriptors, time_stamp)
                distinct_frames.add_img_obj(img_obj)
                print("Reading image .." + str(time_stamp) + " from " + folder)  # for dev phase
            except:
                continue
    else:
        ensure_path(folder_to_save+'/jpg')
        i = 0
        for file in os.listdir(folder):
            try:
                gray = cv2.imread(folder + "/" + file, 0)
                time_stamp = None
                keypoints, descriptors = detector.detectAndCompute(gray, None)
                img_obj = ImgObj(len(keypoints), descriptors, time_stamp)
                save_to_memory(img_obj, 'image' + str(i) + '.pkl', folder_to_save)
                cv2.imwrite(folder_to_save+'/jpg/image' + str(i) + '.jpg', gray)
                distinct_frames.add_img_obj(img_obj)
                print("Reading image .." + str(i) + " from " + folder)  # for dev phase
                i = i + 1
            except:
                continue

    return distinct_frames


def edge_from_specific_pt(i_init, j_init, frames1, frames2):
    """
    Called when frames1[i_init][1] matches best with frames2[j_init][1]. This function checks
    subsequent frames of frames1 and frames2 to see if edge is detected.

    Parameters
    ----------
    i_init: index of the frame in frames1 list , which matches with the corresponding frame
            in frame2 list
    j_init: index of the frame in frames2 list , which matches with the corresponding frame
            in frame1 list
    frames1:
    frames2: are lists containing tuples of the form
            (time_stamp, frame, len_keypoints, descriptors) along path1 and path2

    Returns
    -------
    ( status, i_last_matched, j_last_matched ),
        status: if edge is found or not (starting from i_init and j_init)
        i_last_matched: index of last matched frame of frames1
        j_last_matched: index of last matched frame of frames2

    """
    confidence = 5
    """
    No edge is declared when confidence is zero.

    Confidence is decreased by 1 whenever match is not found for (i)th frame of frame1 among
    the next 4 frames after j_last_matched(inclusive)

    If match is found for (i)th frame, i_last_matched is changed to i, j_last_matched is changed to
    the corresponding match; and confidence is restored to initial value(5)
    """
    match, maxmatch, i, i_last_matched, j_last_matched = None, 0, i_init + 1, i_init, j_init
    """
    INV:
    i = index of current frame (in frames1) being checked for matches; i_last_matched<i<len(frames1)
    i_last_matched = index of last frame (in frames1 ) matched; i_init<=i_last_matched<len(frames1)
    j_last_matched = index of last frame (in frames2 ) matched(with i_last_matched);
                        j_init<=j_last_matched<len(frames2)
    match = index of best matched frame (in frames2) with (i)th frame in frames1. j_last_matched<=match<=j
    maxmatch = fraction matching between (i)th and (match) frames
    """
    while True:
        for j in range(j_last_matched + 1, j_last_matched + 5):
            if j >= frames2.no_of_frames():
                break
            image_fraction_matched = mt.SURF_match_2(frames1.get_object(i).get_elements(),
                                                     frames2.get_object(j).get_elements(),
                                                     2500, 0.7)
            if image_fraction_matched > 0.15:
                if image_fraction_matched > maxmatch:
                    match, maxmatch = j, image_fraction_matched
        if match is None:
            confidence = confidence - 1
            if confidence == 0:
                break
        else:
            confidence = 5
            j_last_matched = match
            i_last_matched = i
        i = i + 1
        if i >= frames1.no_of_frames():
            break
        match, maxmatch = None, 0

    if i_last_matched > i_init and j_last_matched > j_init:
        print("Edge found from :")
        print(str(frames1.get_object(i_init).get_time()) + "to" + str(
            frames1.get_object(i_last_matched).get_time()) + "of video 1")
        print(str(frames2.get_object(j_init).get_time()) + "to" + str(
            frames2.get_object(j_last_matched).get_time()) + "of video 2")
        return True, i_last_matched, j_last_matched
    else:
        return False, i_init, j_init


def compare_videos(frames1: DistinctFrames, frames2: DistinctFrames):
    """
    :param frames1:
    :param frames2: are lists containing tuples of the form (time_stamp, frame) along path1 and path2

    (i)th frame in frames1 is compared with all frames in frames2[lower_j ... (len2)-1].
    If match is found then edge_from_specific_pt is called from indexes i and match
    if edge found then i is incremented to i_last_matched (returned from edge_from_specific_pt) and
    lower_j is incremented to j_last_matched
    """

    len1, len2 = frames1.no_of_frames(), frames2.no_of_frames()
    lower_j = 0
    i = 0
    while (i < len1):
        match, maxmatch = None, 0
        for j in range(lower_j, len2):
            image_fraction_matched = mt.SURF_match_2(frames1.get_object(i).get_elements(),
                                                     frames2.get_object(j).get_elements(),
                                                     2500, 0.7)
            if image_fraction_matched > 0.15:
                if image_fraction_matched > maxmatch:
                    match, maxmatch = j, image_fraction_matched
        if match is not None:
            if i >= len1 or lower_j >= len2:
                break
            status, i, j = edge_from_specific_pt(i, match, frames1, frames2)
            lower_j = j
        i = i + 1


def compare_videos_and_print(frames1, frames2):
    len1, len2 = frames1.no_of_frames(), frames2.no_of_frames()
    lower_j = 0
    for i in range(len1):
        print("")
        print(str(frames1.get_object(i).get_time()) + "->")
        for j in range(lower_j, len2):
            image_fraction_matched = mt.SURF_match_2(frames1.get_object(i).get_elements(),
                                                     frames2.get_object(j).get_elements(),
                                                     2500, 0.7)
            if image_fraction_matched > 0.2:
                print(str(frames2.get_object(j).get_time()) + " : confidence is " + str(image_fraction_matched))


# FRAMES1 = save_distinct_ImgObj("testData/new things/6_2.MP4", "v3", 4, True)
# FRAMES2 = save_distinct_ImgObj("testData/sushant_mc/20190518_155931.mp4", "v2", 4)

# img_obj = FRAMES1.get_object(0)
# img_obj.get_time()
# FRAMES2 = read_images("v2")

# compare_videos_and_print(FRAMES1, FRAMES2)
# compare_videos(FRAMES2, FRAMES1)

'''
fFRAMES1 = cv2.imread("v1/image295.pkl", 0)
FRAMES2 = cv2.imread("v2/image1002.pkl", 0)
image_fraction_matched = mt.SURF_match(FRAMES1, FRAMES2, 2500, 0.7)
print(image_fraction_matched)



cap = cv2.VideoCapture("testData/sushant_mc/20190518_155931.mp4")
ret, frame = cap.read()
cv2.imshow("frame", frame)
print(is_blurry_colorful(frame))


frame = cv2.imread("v2/jpg/image207.jpg", 0)
print(is_blurry_grayscale(frame))
'''
