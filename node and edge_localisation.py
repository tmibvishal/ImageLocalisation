import cv2
import numpy as np
import os
import shutil
import time
from imutils import paths
import video_operations_2 as vo
import matcher at mt

def convert_query_video_to_objects(path, destination_folder):
    vo.save_distinct_ImgObj(path, "storage/"+destination_folder)

""" Assume that a person starts from a specific node.
Query on all nodes.
Store the nodes with maximum match"""

def locate_node(destination_folder):
