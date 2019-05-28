import cv2
import numpy as np
import os
import shutil
import time
from imutils import paths
import video_operations_2 as vo2
import matcher as mt

matched_nodes=[]
matched_edges=[]

def convert_query_video_to_objects(path, destination_folder):
    return vo2.save_distinct_ImgObj(path, "storage/"+destination_folder)

""" Assume that a person starts from a specific node.
Query on all nodes.
Store the nodes with maximum match"""

def locate_node(destination_folder, query_video_frames, no_of_frames_of_query_video_to_be_matched:int=2):
    global matched_nodes
    if len(matched_nodes)!=0:
        matched_nodes=[]
    for i in #querry in graph for nodes:
        confidence=0
        for j in range(no_of_frames_of_query_video_to_be_matched):
            for k in #pass the objects in node folder
           # image_fraction_matched = mt.SURF_match_2((frames1[i][1], frames1[i][2]), (frames2[j][1], frames2[j][2]),
                                                     2500, 0.7)
           if image_fraction_matched> 0.15:
               confidence= confidence+1
        if (confidence/no_of_frames_of_query_video_to_be_matched >0.32)
               matched_nodes.append(i)


def locate_edge(query_video_frames, confidence_level:int=4):
    global matched_nodes
    global matched_edges
    for i in matched_nodes:
        for j in #linkd edges of matched nodes:
            matched_edges.append((#edge, 0, 0)) #(edge, confidence, frame_position_matched)

    for i in query_video_frames:
        for j in matched_edges:
            match, maxmatch= None, 0
            for k in #starting from matched_edges[j][2] till end #edge folder:
                # image_fraction_matched = mt.SURF_match_2((frames1[i][1], frames1[i][2]), (frames2[j][1], frames2[j][2]),
                2500, 0.7)
                if image_fraction_matched >0.15:
                    if image_fraction_matched > maxmatch:
                        match, maxmatch = j, image_fraction_matched
            if match is not None:
                matched_edges[j]=(matched_edges[j][0],matched_edges[j][1]+1, match)
            else:
                matched_edges[j]=(matched_edges[j][0],matched_edges[j][1]-1, matched_edges[j][2])
            if matched_edges[j][1]< (-1)*confidence_level:
                matched_edges.pop(j)
            if matched_edges[j][1]> confidence_level and len(matched_edges)==1:
                print("edge found")
                break















query_video_frames= convert_query_video_to_objects(path, destination_folder)