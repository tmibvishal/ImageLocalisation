"""videoOperations.py
Contains functions that operate on video or stream of images
"""

import cv2
import numpy as np
import matcher as mt
import os
import time

# path = './Images'

def save_distinct_frames(video_str, folder, frames_skipped=0):

    """Saves non redundent and distinct frames of a video in folder

    Parameters
    ----------
    video_str : is video_str = "webcam" then loadswebcam O.W. loads video at video_str location,
    folder : folder where non redundant images are to be saved,
    frames_skipped: Number of frames to skip and just not consider

    Returns
    -------
    array,
        returns array contaning non redundant frames(mat format)
    """

    frames_skipped += 1

    if video_str == "webcam":
        video_str = 0
    cap = cv2.VideoCapture(video_str)
    # cap= cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)

    distinct_frames = []
    comparison_frame = None
    i = 0
    a = None
    b = None

    ret, frame = cap.read()
    a = frame
    cv2.imwrite('image' + str(i) + '.jpg', a)

    while True:
        ret, frame = cap.read()
        if ret:
            if(i % frames_skipped != 0):
                i = i + 1
                continue
            cv2.imshow('frame', frame)
            b = frame
            image_fraction_matched = mt.SURF_match(a, b, 2500, 0.7)
            if image_fraction_matched < 0.1:
                cv2.imwrite(folder + '/image' + str(i) + '.jpg', a)
                distinct_frames.append((i, a))
                a = b

            i = i + 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    return distinct_frames


def compare_videos():
    frames1 = save_distinct_frames("testData/sushant_mc/20190517_220001.mp4", "v1", 4)
    frames2 = save_distinct_frames("testData/sushant_mc/20190517_220439.mp4", "v2", 4)

    len1, len2 = len(frames1), len(frames2)
    best_matches = []
    for i in range(len1):
        print("")
        print(str(frames1[i][0])+":")
        best_matches_for_i = []
        for j in range(len2):
            image_fraction_matched = mt.SURF_match(frames1[i][1], frames2[j][1], 2500, 0.7)
            if image_fraction_matched > 0.2:
                print(str(frames2[j][0])+" : confidence is "+str(image_fraction_matched))
                best_matches_for_i.append((frames2[j][0], image_fraction_matched))
        best_matches.append((frames1[i][0], best_matches_for_i))

compare_videos()