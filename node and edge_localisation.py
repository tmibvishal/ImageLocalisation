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

class node_and_image_matching:
    def __init__(self):
        self.matched_nodes = []
        self.matched_edges = []
        self.final_path = []
        self.last_frame_matched=0

    def convert_query_video_to_objects(path, destination_folder):
        return vo2.save_distinct_ImgObj(path, destination_folder)

    """ Assume that a person startas frpom a specific node.
    Query on all nodes.
    Store the nodes with maximum match"""

    def locate_initial_node(self, nodes_list: list, query_video_frames: vo2.DistinctFrames,
                    no_of_frames_of_query_video_to_be_matched: int = 2):
        """

        :param nodes_list: consist of array of nodes stored in graph
        :param query_video_frames: distinct image frames of query video
        :param no_of_frames_of_query_video_to_be_matched: the initial no of frames that has to be matched with each node to locate the starting node
        :return: the array which consist of object "Node" that have matched with query video

        a node is considered matched if the ratio of the number of frames that of node that match with query video frame to the number of frames of query video queriec
        is greater than 0.32 or 32%
        """
        if len(self.matched_nodes) != 0:
            self.matched_nodes = []
        for node in nodes_list:
            confidence = 0
            node_images = node.node_images
            if node_images is None:
                continue
            no_of_node_images = node_images.no_of_frames()
            for j in range(no_of_frames_of_query_video_to_be_matched):
                for k in range(no_of_node_images):
                    image_fraction_matched = mt.SURF_match_2(query_video_frames.get_object(j).get_elements(),
                                                             node_images.get_object(k).get_elements(),
                                                             2500, 0.7)
                    if image_fraction_matched > 0.15:
                        print(image_fraction_matched)
                        print(node.name)
                        print("query video frame"+str(j))
                        print("node image no"+str(k))
                        print()
                        confidence = confidence + 1
            if confidence / no_of_frames_of_query_video_to_be_matched > 0.32:
                self.matched_nodes.append(node)
        for nd in self.matched_nodes:
            print(nd.name)
        return self.matched_nodes

    def match_next_node(self, nodes_list: list, query_video_frames: vo2.DistinctFrames,
                        last_frame_object: vo2.ImgObj,
                        no_of_frames_of_query_video_to_be_matched: int = 2):
        if len(self.matched_nodes) != 0:
            self.matched_nodes = []
        new_src_node = last_frame_object[1].dest
        for node in nodes_list:
            if node.identity == new_src_node:
                print("matching new node" + node.name)
                confidence = 0
                node_images = node.node_images
                if node_images is None:
                    continue
                no_of_node_images = node_images.no_of_frames()
                for j in range(query_video_frames.no_of_frames()):
                    for k in range(no_of_node_images):
                        image_fraction_matched = mt.SURF_match_2(query_video_frames.get_object(j).get_elements(),
                                                                 node_images.get_object(k).get_elements(),
                                                                 2500, 0.7)

                        if image_fraction_matched > 0.10:
                            confidence = confidence + 1
                            print("query video frame " + str(j))
                            print("node frame" + str(k) + " of " + node.name)
                            print(image_fraction_matched)
                if confidence / no_of_frames_of_query_video_to_be_matched > 0.32:
                    self.matched_nodes.append(node)
                    self.locate_edge(nodes_list, query_video_frames)
                    break
                else:
                    print("not high confidence but still brute force")
                    self.matched_nodes.append(node)
                    self.locate_edge(nodes_list, query_video_frames)
                    break

    def locate_edge(self, nodes_list:list, query_video_frames: vo2.DistinctFrames,
                    query_video_frames_begin: int = 0, confidence_level: int = 1):
        query_video_frames_begin=self.last_frame_matched
        for node in self.matched_nodes:
            for edge in node.links:
                self.matched_edges.append([edge, 0, 0])  # (edge, confidence, frame_position_matched)
                print("edges added" + str(edge.src) + "_" + str(edge.dest))
                print()

        last_frame_matched_with_edge = query_video_frames_begin

        for i in range(query_video_frames_begin, query_video_frames.no_of_frames()):
        edge_list = self.matched_edges
            j = 0
            while j < len(edge_list):



                match, maximum_match = None, 0
                for k in range(int(edge_list[j][2]), edge_list[j][0].distinct_frames.no_of_frames()):
                    # starting from matched_edges[j][2] till end #edge folder
                    image_fraction_matched = mt.SURF_match_2(
                        edge_list[j][0].distinct_frames.get_object(k).get_elements(),
                        query_video_frames.get_object(i).get_elements(), 2500, 0.7)
                    print("query frame "+ str(i))
                    print("edge frame "+str(edge_list[j][0].src) + "_" + str(edge_list[j][0].dest)+" " + str(k))
                    print(image_fraction_matched)
                    if image_fraction_matched > 0.15:
                        if image_fraction_matched > maximum_match:
                            last_frame_matched_with_edge = i
                            print(image_fraction_matched)
                            print(i)
                            print(str(edge_list[j][0].src) + "_" + str(edge_list[j][0].dest))
                            print(k)
                            match, maximum_match = k, image_fraction_matched
                if match is not None:
                    edge_list[j][1] = edge_list[j][1] + 1
                    edge_list[j][2] = match
                    print(str(edge_list[j][0].src) + "_" + str(
                        edge_list[j][0].dest) + " has increased confidence = " + str(edge_list[j][1]))
                else:
                    # edge_list[j][1] = edge_list[j][1] - 1
                    print(str(edge_list[j][0].src) + "_" + str(
                        edge_list[j][0].dest) + " has no change in confidence = " + str(edge_list[j][1]))
                # if edge_list[j][1] < (-1) * confidence_level:
                #     print(str(edge_list[j][0].src) + "_" + str(edge_list[j][0].dest) + "deleted")
                #     del edge_list[j]
                # elif edge_list[j][1] > confidence_level or len(edge_list) == 1:
                #     print("edge found")
                #     break
                if edge_list[j][1]>= confidence_level:
                    print("edge found")
                    found_edge= edge_list[j]
                    edge_list=[]
                    edge_list.append(found_edge)
                    break
                else:
                    j += 1
            if len(edge_list) == 1:
                print("edge found finally")
                last_frame_object = (last_frame_matched_with_edge, edge_list[0][0])
                print(str(edge_list[0][0].src) + "_" + str(edge_list[0][0].dest))
                source_node = edge_list[0][0].src

                for node in self.matched_nodes:
                    if str(node.identity) == str(source_node):
                        self.final_path.append(node)
                        self.matched_nodes = []
                        self.final_path.append(edge_list[0][0])
                        self.matched_edges = []
                self.match_next_node(nodes_list, query_video_frames, last_frame_object)
                break

    def print_final_path(self):
        print("path is: ")
        for element in self.final_path:
            if isinstance(element, Node):
                print(element.name)
                print()
            elif isinstance(element, Edge):
                print(str(element.src) + "_" + str(element.dest))
                print()
            else:
                raise Exception("Path not right")


# graph=Graph()
# graph.add_floor_map(0, "graph/images/map0.jpg")
# graph.mark_nodes(0)
# graph.make_connections(0)
# graph.read_nodes("testData/Morning_sit/nodes",4)
# graph.read_edges("testData/Morning_sit/edges",4)
# graph.print_graph(0)
# graph.save_graph()


# query_video_frames1 = vo2.save_distinct_ImgObj("testData/query_sit0/20190528_160046.mp4","query_distinct_frame",3,True)
# # graph1=graph.load_graph()
# graph =load_graph("graph_1.pkl")
# node_and_image_matching_obj = node_and_image_matching()
# node_and_image_matching_obj.locate_initial_node(graph.Nodes, query_video_frames1)
# node_and_image_matching_obj.locate_edge(graph.Nodes, query_video_frames1)
# node_and_image_matching_obj.print_final_path()

# FRAMES1 = vo2.read_images_jpg("testData/node 2 - 6")
# FRAMES2 = vo2.read_images_jpg("testData/Photo frames sit 0/3")
# FRAMES3 = vo2.read_images_jpg("testData/Photo frames sit 0/6")
# graph1 = load_graph()
# graph1._add_edge_images(2, 6, FRAMES1)
# graph1._add_node_images(3, FRAMES2)
# graph1._add_node_images(6, FRAMES3)
# graph1.save_graph()

def run(code: int):
    # Create new graph
    if code == 0:
        graph = Graph()
        graph.add_floor_map(0, "graph/images/map0.jpg")
        graph.mark_nodes(0)
        graph.make_connections(0)
        graph.print_graph(0)
        graph.save_graph()

    # Print graph
    if code == 1:
        graph = load_graph()
        graph.print_graph(0)

    # Add nodes and edges
    if code == 2:
        graph = load_graph()
        graph.read_nodes("testData/Evening Sit/nodes", 4)
        graph.read_edges("testData/Evening Sit/edges", 4)
        graph.save_graph()

    # Query video
    if code == 3:
        query_video_frames1 = vo2.save_distinct_ImgObj("testData/Evening Sit/VID_20190610_203834.webm",
                                                       "query_distinct_frame",3, True)
        # query_video_frames1 = vo2.read_images("query_distinct_frame")
        graph = load_graph()
        node_and_image_matching_obj = node_and_image_matching()
        node_and_image_matching_obj.locate_initial_node(graph.Nodes, query_video_frames1)
        node_and_image_matching_obj.locate_edge(graph.Nodes, query_video_frames1)
        node_and_image_matching_obj.print_final_path()

    # Add specific node/edge data manually
    if code == 4:
        FRAMES1 = vo2.read_images_jpg("testData/node 2 - 6")
        FRAMES2 = vo2.read_images_jpg("testData/Photo frames sit 0/3")
        graph1 = load_graph("graph.pkl")
        graph1._add_edge_images(2, 6, FRAMES1)
        graph1._add_node_images(3, FRAMES2)
        graph1.save_graph()

    # Add node images
    if code == 5:
        graph = load_graph()
        graph.read_nodes_directly("testData/Node-direct-images")
        graph.save_graph()



