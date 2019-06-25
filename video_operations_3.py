"""video_operations.py
Contains functions that operate on video or stream of images
"""

import cv2
import numpy as np
import os
import time
import pickle
import matcher as mt
from general import *


class ImgObj:
    def __init__(self, no_of_keypoints, descriptors, time_stamp, serialized_keypoints, shape):
        self.no_of_keypoints = no_of_keypoints
        self.descriptors = descriptors
        self.time_stamp = time_stamp
        self.serialized_keypoints = serialized_keypoints
        self.shape = shape

    def get_elements(self):
        return self.no_of_keypoints, self.descriptors, self.serialized_keypoints, self.shape

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
        if start_index == 0 and end_index == -1:
            return self.img_objects[start_index:end_index]
        if (start_index not in range(0, self.no_of_frames())) or (end_index not in range(0, self.no_of_frames())):
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


def serialize_keypoints(keypoints):
    index = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        index.append(temp)
    return index

def deserialize_keypoints(index):
    kp = []
    for point in index:
        temp = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                            _octave=point[4], _class_id=point[5])
        kp.append(temp)
    return kp


def save_distinct_ImgObj(video_str, folder, frames_skipped: int = 0, check_blurry: bool = True,
                         hessian_threshold: int = 2500, ensure_min=True):
    """Saves non redundent and distinct frames of a video in folder
    Parameters
    ----------
    video_str : is video_str = "webcam" then loads webcam. O.W. loads video at video_str location,
    folder : folder where non redundant images are to be saved,
    frames_skipped: Number of frames to skip and just not consider,
    check_blurry: If True then only considers non blurry frames but is slow
    hessian_threshold
    ensure_min: whether a minimum no of frames (at least one per 50) is to be kept irrespective of
        whether they are distinct or not

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
    i_prev = 0  # the last i which was stored

    detector = cv2.xfeatures2d_SURF.create(hessian_threshold)

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    a = (len(keypoints), descriptors, serialize_keypoints(keypoints), gray.shape)
    img_obj = ImgObj(a[0], a[1], i, a[2], a[3])
    save_to_memory(img_obj, 'image' + str(i) + '.pkl', folder)
    cv2.imwrite(folder + '/jpg/image' + str(i) + '.jpg', gray)
    distinct_frames.add_img_obj(img_obj)
    i_of_a=0
    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if i % frames_skipped != 0 and not check_next_frame:
                i = i + 1
                continue

            cv2.imshow('frame', gray)
            # print(i)

            if check_blurry:
                if is_blurry_grayscale(gray):
                    check_next_frame = True
                    print("frame " + str(i) + " skipped as blurry")
                    i = i + 1
                    continue
                check_next_frame = False

            keypoints, descriptors = detector.detectAndCompute(gray, None)
            b = (len(keypoints), descriptors, serialize_keypoints(keypoints), gray.shape)
            if len(keypoints)<100:
                print("frame "+str(i)+ " skipped as "+str(len(keypoints))+" <100")
                i = i+1
                continue
            import matcher as mt
            image_fraction_matched, min_good_matches = mt.SURF_returns(a, b, 2500, 0.7, True)
            if image_fraction_matched == -1:
                check_next_frame = True
                i=i+1
                continue
            check_next_frame = False
            if 0< image_fraction_matched < 0.1 or min_good_matches<50 or (ensure_min and i - i_prev > 50):
                img_obj2 = ImgObj(b[0], b[1], i, b[2], b[3])
                print(str(image_fraction_matched)+ " fraction match between "+str(i_of_a)+" and "+ str(i))
                save_to_memory(img_obj2, 'image' + str(i) + '.pkl', folder)
                cv2.imwrite(folder + '/jpg/image' + str(i) + '.jpg', gray)
                distinct_frames.add_img_obj(img_obj2)
                a = b
                i_of_a=i
                i_prev = i

            i = i + 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    print("Created distinct frames object")
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


def read_images_jpg(folder, hessian_threshold: int = 2500):
    """Reads images of the form "image<int>.jpg" from folder(passed as string containing
    relative path of the specific folder)

    Parameters
    ----------
    folder

    Returns
    -------
    array,
        distinct_frames : a list containing tuples of the form (time_stamp, frame)
    where time_stamp is the <int> part of image<int>.jpg and frame is object of the
    image created using imread
    """
    distinct_frames = DistinctFrames()
    detector = cv2.xfeatures2d_SURF.create(hessian_threshold)

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
            img_obj = ImgObj(len(keypoints), descriptors, time_stamp, serialize_keypoints(keypoints))
            distinct_frames.add_img_obj(img_obj)
            print("Reading image .." + str(time_stamp) + " from " + folder)  # for dev phase
        except:
            continue

    return distinct_frames




# FRAMES1 = save_distinct_ImgObj("testData/afternoon_sit0 15june/NodeData/7.mp4", "v2", 4, True)
# FRAMES2 = save_distinct_ImgObj("testData/sushant_mc/20190518_155931.mp4", "v2", 4)

# img_obj = FRAMES1.get_object(0)
# img_obj.get_time()
# FRAMES2 = read_images("v2")

# compare_videos_and_print(FRAMES1, FRAMES2)
# compare_videos(FRAMES2, FRAMES1)

# FRAMES1 = cv2.imread("query_distinct_frame/case1/jpg/image244.jpg", 0)
# FRAMES2 = cv2.imread("edge_data/edge_0_1/jpg/image285.jpg", 0)
# image_fraction_matched = mt.SURF_match(FRAMES1, FRAMES2, 2500, 0.7)
# print(image_fraction_matched)

'''v2/image1002.pkl
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

# frames1 = save_distinct_ImgObj("testData/sit_morning_14_june/edges/0_1.webm", "query_distinct_frame/case7", 14, True, ensure_min=True)
#save_distinct_ImgObj("testData/night sit 0 june 18/query video/VID_20190618_202826.webm",
 #                    "query_distinct_frame/night", 3)