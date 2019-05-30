"""video_operations.py
Contains functions that operate on folder of images adn create folder of edges
"""

import cv2
import numpy as np
import matcher as mt
import os
import time
from imutils import paths
import shutil

# path = './Images'

def read_images(folder):
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
    distinct_frames = []

    for file in sorted(sorted(os.listdir(folder)), key=len):  # sorting files on basis of
        # 1) length and 2) numerical order
        '''
            Sorting is done 2 times because
            if files in the folder are
                1. image100.jpg
                2. image22.jpg
                3. image21.jpg
            firstly sort them to image100.jpg,image21.jpg,image22.jpg then according to length to image21.jpg,image22.jpg,image100.jpg
        '''
        frame = cv2.imread(folder + "/" + file)
        time_stamp = int(file.replace('image', '').replace('.jpg', ''), 10)
        distinct_frames.append((time_stamp, frame))
        print("Reading image .." + str(time_stamp) + " from " + folder)  # for dev phase
    return distinct_frames

def video_to_edges(video_number: str,path :str):
    if video_number=="1":
        shutil.copytree(path, "storage/1")
    else:
        frames_new = read_images(path)
        for folder in sorted(os.listdir("storage"))[:]:
            folder = str(folder)
            frames_of_edges =read_images('storage/'+folder)
            frames_being_used = frames_new
            # compare_videos_and_print(frames_being_used, frames_of_edges)
            compare_videos(frames_of_edges,frames_being_used)

        # print (len([folder for folder in sorted(os.listdir("storage"))]))
        # if os.listdir("storage").len():
        #     print("hi")


def edge_from_specific_pt1(i_init, j_init, frames1, frames2, k, folder, video_number):
    """
    Called when frames1[i_init][1] matches best with frames2[j_init][1]. This function checks
    subsequent frames of frames1 and frames2 to see if edge is detected.

    Parameters
    ----------
    i_init: index of the frame in frames1 list , which matches with the
    corresponding frame in frame2 list
    j_init: index of the frame in frames2 list , which matches with the
    corresponding frame in frame1 list
    frames1:
    frames2: are lists containing tuples of the form (time_stamp, frame) along path1 and path2

    Returns
    -------
    status, i_last_matched, j_last_matched,
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
    i = index of current frame (in frames1) being checked for matches; i_last_matched<i<len(frames1)
    i_last_matched = index of last frame (in frames1 ) matched; i_init<=i_last_matched<len(frames1)
    j_last_matched = index of last frame (in frames2 ) matched(with i_last_matched);
                        j_init<=j_last_matched<len(frames2)
    match = index of best matched frame (in frames2) with (i)th frame in frames1. j_last_matched<=match<=j
    maxmatch = fraction matching between (i)th and (match) frames
    """
    while True:
        for j in range(j_last_matched, j_last_matched + 4):
            if j >= len(frames2):
                break
            image_fraction_matched = mt.SURF_match(frames1[i][1], frames2[j][1], 2500, 0.7)
            if image_fraction_matched > 0.1:
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
        if i >= len(frames1):
            break
        match, maxmatch = None, 0

    if i_last_matched > i_init and j_last_matched > j_init:
        print("Edge found from :")
        print(str(frames1[i_init][0]) + "to" + str(frames1[i_last_matched][0]) + "of video 1")
        print(str(frames2[j_init][0]) + "to" + str(frames2[j_last_matched][0]) + "of video 2")
        # for l in range(i_init, i_last_matched+1):
        #     frames1[l]=(0,0)
        # for l in range(j_init, j_last_matched + 1):
        #     create_folder(folder+"_"+video_number)
        #     shutil.copy("storage/"+str(folder)+"/image"+str(frames2[l][0])+".jpg", "storage/"+str(folder)+"_"+video_number )
        #     os.remove("storage/"+str(folder)+"/image"+str(frames2[l][0])+".jpg")
        #     frames2[l] = (0, 0)

        return True, i_last_matched, j_last_matched
    else:
        return False, i_init, j_init


def compare_videos1(frames1, frames2, folder, video_number):
    """
    :param frames1:
    :param frames2: are lists containing tuples of the form (time_stamp, frame) along path1 and path2

    (i)th frame in frames1 is compared with all frames in frames2[lower_j ... (len2)-1].
    If match is found then edge_from_specific_pt is called from indexes i and match
    if edge found then i is incremented to i_last_matched (returned from edge_from_specific_pt) and
    lower_j is incremented to j_last_matched
    """

    len1, len2 = len(frames1), len(frames2)
    lower_j = 0
    k=1
    i=0
    while(i<len1):
        # if frames1[i] !=(0,0):
        match, maxmatch = None, 0
        for j in range(lower_j, len2):
            # if frames2[j] !=(0,0):
            image_fraction_matched = mt.SURF_match(frames1[i][1], frames2[j][1], 2500, 0.7)
            if image_fraction_matched > 0.1:
                if image_fraction_matched > maxmatch:
                    match, maxmatch = j, image_fraction_matched
                # else:
                #     continue
            if match is not None:
                print(i)
                print(j)
                status, i, j = edge_from_specific_pt(i, match, frames1, frames2,k, folder, video_number)
                lower_j = j
                k=k+1
                if i >= len1 or lower_j >= len2:
                    break
                i += 1
        # else:
        #     continue


def compare_videos_and_print(frames1, frames2):
    len1, len2 = len(frames1), len(frames2)
    # best_matches = []
    lower_j = 0
    for i in range(len1):
        print("")
        print(str(frames1[i][0]) + "->")
        # best_matches_for_i = []
        for j in range(lower_j, len2):
            image_fraction_matched = mt.SURF_match(frames1[i][1], frames2[j][1], 2500, 0.7)
            if image_fraction_matched > 0.2:
                print(str(frames2[j][0]) + " : confidence is " + str(image_fraction_matched))
                # best_matches_for_i.append((frames2[j][0], image_fraction_matched))
                # best_matches.append((frames1[i][0], best_matches_for_i))

def edge_from_specific_pt(i_init, j_init, frames1, frames2):
    """
    Called when frames1[i_init][1] matches best with frames2[j_init][1]. This function checks
    subsequent frames of frames1 and frames2 to see if edge is detected.

    Parameters
    ----------
    i_init: index of the frame in frames1 list , which matches with the
    corresponding frame in frame2 list
    j_init: index of the frame in frames2 list , which matches with the
    corresponding frame in frame1 list
    frames1:
    frames2: are lists containing tuples of the form (time_stamp, frame) along path1 and path2

    Returns
    -------
    status, i_last_matched, j_last_matched,
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
    i = index of current frame (in frames1) being checked for matches; i_last_matched<i<len(frames1)
    i_last_matched = index of last frame (in frames1 ) matched; i_init<=i_last_matched<len(frames1)
    j_last_matched = index of last frame (in frames2 ) matched(with i_last_matched);
                        j_init<=j_last_matched<len(frames2)
    match = index of best matched frame (in frames2) with (i)th frame in frames1. j_last_matched<=match<=j
    maxmatch = fraction matching between (i)th and (match) frames
    """
    while True:
        for j in range(j_last_matched, j_last_matched + 4):
            if j >= len(frames2):
                break
            image_fraction_matched = mt.SURF_match(frames1[i][1], frames2[j][1], 2500, 0.7)
            if image_fraction_matched > 0.1:
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
        if i >= len(frames1):
            break
        match, maxmatch = None, 0

    if i_last_matched > i_init and j_last_matched > j_init:
        print("Edge found from :")
        print(str(frames1[i_init][0]) + "to" + str(frames1[i_last_matched][0]) + "of video 1")
        print(str(frames2[j_init][0]) + "to" + str(frames2[j_last_matched][0]) + "of video 2")
        return True, i_last_matched, j_last_matched
    else:
        return False, i_init, j_init


def compare_videos(frames1, frames2):
    """
    :param frames1:
    :param frames2: are lists containing tuples of the form (time_stamp, frame) along path1 and path2

    (i)th frame in frames1 is compared with all frames in frames2[lower_j ... (len2)-1].
    If match is found then edge_from_specific_pt is called from indexes i and match
    if edge found then i is incremented to i_last_matched (returned from edge_from_specific_pt) and
    lower_j is incremented to j_last_matched
    """

    len1, len2 = len(frames1), len(frames2)
    lower_j = 0
    i = 0
    while(i < len1):
        match, maxmatch = None, 0
        for j in range(lower_j, len2):
            image_fraction_matched = mt.SURF_match(frames1[i][1], frames2[j][1], 2500, 0.7)
            if image_fraction_matched > 0.1:
                if image_fraction_matched > maxmatch:
                    match, maxmatch = j, image_fraction_matched
        if match is not None:
            status, i, j = edge_from_specific_pt(i, match, frames1, frames2)
            lower_j = j
            if i >= len1 or lower_j >= len2:
                break
        i = i + 1

# frames1 = read_images("v1")
# frames2 = read_images("v2")
# frames1 = save_distinct_frames("testData/sushant_mc/20190518_155651.mp4", "v1", 4,True)
# frames2 = save_distinct_frames("testData/sushant_mc/20190518_155820.mp4", "v2", 4, True)
#
# compare_videos_and_print(frames1, frames2)
# compare_videos(frames1, frames2)


"""
1. take first folder and convert it to single edge and name it 1_a
2. take nest video and match it with  the existing edges , if match found create new edge i
where i!= existing i 
and let the unmatche go to folder 1_a and i_a repectively

"""

def create_folder(directory):
    try:
        if not os.path.exists('./storage/'+directory+'/'):
            os.makedirs('./storage/'+directory+'/')
    except OSError:
        print('Error :creating directory' + directory)

# path = os.getcwd()
# new_path =os.path.join(path,"testing")
# print (new_path)
#path="v1"
#video_to_edges(video_number,path)

video_to_edges("1", "v1_n")
# video_to_edges("2", "v2")

# shutil.copytree("v1", "storage/1")