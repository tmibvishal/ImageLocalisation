import cv2
import numpy as np
import os
import shutil
import time
from imutils import paths
import video_operations_2 as vo2
import matcher as mt
import graph
from graph import Graph

graph = graph.Graph()

class node_and_image_maching:
    def __init__(self):
        self.matched_nodes=[]
        self.matched_edges=[]

    def convert_query_video_to_objects(path, destination_folder):
        return vo2.save_distinct_ImgObj(path,destination_folder)

    """ Assume that a person starts from a specific node.
    Query on all nodes.
    Store the nodes with maximum match"""

    def locate_node(self,nodes_list, query_video_frames, no_of_frames_of_query_video_to_be_matched:int=2):
        if len(self.matched_nodes)!=0:
            self.matched_nodes=[]
        for node in nodes_list:
            confidence=0
            node_images= node.node_images
            for j in range(no_of_frames_of_query_video_to_be_matched):
                for k in range(len(node_images)):
                    image_fraction_matched = mt.SURF_match_2(query_video_frames.get_object(j).get_elements(), node_images[k].get_elements(),
                                                         2500, 0.7)
                    if image_fraction_matched> 0.15:
                       confidence= confidence+1
            if confidence/no_of_frames_of_query_video_to_be_matched >0.32:
                self.matched_nodes.append(node)
        print(self.matched_nodes)


    # def locate_edge(self,query_video_frames, confidence_level:int=4):
    #     for node in self.matched_nodes:
    #         for edge in node.links:
    #             self.matched_edges.append((edge, 0, 0)) #(edge, confidence, frame_position_matched)
    #
    #     for i in query_video_frames:
    #         for j in self.matched_edges:
    #             match, maximum_match= None, 0
    #             for k in range(int(j[2]),len(j[0].distinct_frames)):#starting from matched_edges[j][2] till end #edge folder
    #                 image_fraction_matched = mt.SURF_match_2((j[0].distinct_frames[k][1], j[0].distinct_frames[k][2]), (i[1], i[2]), 2500, 0.7))
    #                 if image_fraction_matched >0.15:
    #                     if image_fraction_matched > maximum_matchmatch:
    #                         match, maxmatch = k, image_fraction_matched
    #             if match is not None:
    #                 self.matched_edges[j]=(self.matched_edges[j][0],self.matched_edges[j][1]+1, match)
    #             else:
    #                 self.matched_edges[j]=(self.matched_edges[j][0],self.matched_edges[j][1]-1, self.matched_edges[j][2])
    #             if self.matched_edges[j][1]< (-1)*confidence_level:
    #                 self.matched_edges.pop(j)
    #             if self.matched_edges[j][1]> confidence_level and len(self.matched_edges)==1:
    #                 print("edge found")
    #                 break















query_video_frames= vo2.read_images("query_distinct_frame")
graph1=graph.load_graph()
node_and_image_maching.locate_node(graph1.Nodes, query_video_frames)
print(graph[2])