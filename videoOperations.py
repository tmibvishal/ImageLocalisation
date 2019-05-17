"""videoOperations.py

Contains functions that operate on video or stream of images
"""

import cv2
import numpy as np
import matcher as mt
import os
import time


# path = './Images'


def save_frames(video_str, folder):
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
    cv2.imwrite('image' + str(i) + '.jpg', b)

    while True:
        ret, frame = cap.read()
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

    cap.release()
    cv2.destroyAllWindows()
    return distinct_frames


def compare_videos():
    frames1 = save_frames("testData/sushant_mc/20190517_220001.mp4", "v1")
    frames2 = save_frames("testData/sushant_mc/20190517_220439.mp4", "v2")

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







