import cv2
import video_operations_3 as vo2
import os
import general
import copy
import matcher as mt
import shutil
import time
import numpy as np


class Node:
    def __init__(self, identity: int, name: str, x: int, y: int, z: int):
        self.identity = identity
        self.name = name
        self.coordinates = (x, y, z)
        self.links = []
        self.node_images = None


class Edge:
    def __init__(self, is_connected: bool, src: int, dest: int, distinct_frames=None, video_length: int = None):
        self.src = src
        self.dest = dest
        self.distinct_frames = distinct_frames
        self.video_length = video_length


class FloorMap:
    def __init__(self, floor_no: int = None, img=None):
        self.floor_no = floor_no
        self.pure = img
        self.impure = copy.deepcopy(img)


class Graph:

    def __init__(self):
        self.new_node_index = 0
        self.Nodes = []  # list of list of nodes Nodes[0] will be list of all nodes of floor0
        self.no_of_floors = 0
        self.Floor_map = []
        self.path_traversed = []

    # private functions
    def get_node(self, identity, z=None):
        if z is not None:
            for Nd in self.Nodes[z]:
                if identity == Nd.identity:
                    return Nd
        else:
            for floor_nodes in self.Nodes:
                for Nd in floor_nodes:
                    if identity == Nd.identity:
                        return Nd
        return None

    def get_edge(self, src, dest, z_src=None, z_dest=None):
        nd1 = self.get_node(src, z_src)
        nd2 = self.get_node(dest, z_dest)
        if nd1 is not None and nd2 is not None:
            for edge in nd1.links:
                if edge.dest == dest:
                    return edge
        return None

    def get_edges(self, identity: int, z=None):
        Nd = self.get_node(identity, z)
        if Nd is not None:
            return Nd.links
        return None

    def _create_node(self, name, x, y, z):
        identity = self.new_node_index
        Nd = Node(identity, name, x, y, z)
        self._add_node(Nd)

    def _add_node(self, Nd):
        z = Nd.coordinates[2]
        if len(self.Nodes) <= z:
            for i in range(0, z + 1 - len(self.Nodes)):
                self.Nodes.append([])
        if isinstance(Nd, Node):
            if Nd not in self.Nodes[z]:
                if isinstance(Nd.links, list):
                    if len(Nd.links) == 0 or isinstance(Nd.links[0], Edge):
                        self.Nodes[z].append(Nd)
                        self.new_node_index = self.new_node_index + 1
                else:
                    raise Exception("Nd.links is not a list of Edge")
            else:
                raise Exception("Nd is already present")
        else:
            raise Exception("Nd format is not of Node")

    def _nearest_node(self, x, y, z):
        def distance(nd):
            delx = abs(nd.coordinates[0] - x)
            dely = abs(nd.coordinates[1] - y)
            return delx ** 2 + dely ** 2

        minimum, nearest_node = -1, None
        for Nd in self.Nodes[z]:
            if abs(Nd.coordinates[0] - x) < 50 and abs(Nd.coordinates[1] - y) < 50:
                if minimum == -1 or distance(Nd) < minimum:
                    nearest_node = Nd
                    minimum = distance(Nd)
        return nearest_node

    def _connect(self, nd1, nd2):
        if isinstance(nd1, Node) and isinstance(nd2, Node):
            if nd2.identity < self.new_node_index and nd1.identity < self.new_node_index:
                edge = Edge(True, nd1.identity, nd2.identity)
                nd1.links.append(edge)
            else:
                raise Exception("Wrong identities of Nodes")
        else:
            raise Exception("Nd format is not of Node, or is already present")

    def _delete_node(self, nd: Node):
        z = nd.coordinates[2]
        if nd in self.Nodes[z]:
            self.Nodes[z].remove(nd)
            for floor_nodes in self.Nodes:
                for nd2 in floor_nodes:
                    for edge in nd2.links:
                        if nd.identity == edge.dest:
                            nd2.links.remove(edge)
        else:
            raise Exception("Nd does not exists in Nodes")

    def _add_edge_images(self, id1: int, id2: int, distinct_frames: vo2.DistinctFrames, z1=None, z2=None):
        if id1 > self.new_node_index or id2 > self.new_node_index:
            raise Exception("Wrong id's passed")
        if not isinstance(distinct_frames, vo2.DistinctFrames):
            raise Exception("Invalid param for distinct_frames")
        edge = self.get_edge(id1, id2, z1, z2)
        if edge is not None:
            edge.distinct_frames = distinct_frames
            edge.video_length = distinct_frames.get_time()
            return
        raise Exception("Edge from " + str(id1) + " to " + str(id2) + " not found")

    def _add_node_images(self, identity, node_images, z=None):
        if not isinstance(node_images, vo2.DistinctFrames):
            raise Exception("node_images is not DistinctFrames object")

        Nd = self.get_node(identity, z)
        if Nd is not None:
            Nd.node_images = node_images
            return
        raise Exception("Node " + str(identity) + " not found!")

    def _add_node_data(self, identity: int, path_of_video: str, folder_to_save: str = None,
                       frames_skipped: int = 0, check_blurry: bool = True, hessian_threshold: int = 2500,
                       z_node=None):
        distinct_frames = vo2.save_distinct_ImgObj(path_of_video, folder_to_save, frames_skipped, check_blurry,
                                                   hessian_threshold)
        self._add_node_images(identity, distinct_frames, z_node)

    def _add_edge_data(self, id1: int, id2: int, path_of_video: str, folder_to_save: str = None,
                       frames_skipped: int = 0, check_blurry: bool = True, hessian_threshold: int = 2500,
                       z1=None, z2=None):
        distinct_frames = vo2.save_distinct_ImgObj(path_of_video, folder_to_save, frames_skipped, check_blurry,
                                                   hessian_threshold, ensure_min=True)
        self._add_edge_images(id1, id2, distinct_frames, z1, z2)

    def _get_floor_img(self, z, params):
        for floor in self.Floor_map:
            if floor.floor_no == z:
                if params == "pure":
                    return floor.pure
                elif params == "impure":
                    return floor.impure
                else:
                    raise Exception("Invalid params passed")
        raise Exception("Couldn't find floor")

    def _set_floor_img(self, z, params, img):
        for floor in self.Floor_map:
            if floor.floor_no == z:
                if params == "pure":
                    floor.pure = img
                    return
                elif params == "impure":
                    floor.impure = img
                    return
                else:
                    raise Exception("Invalid params")
        raise Exception("Couldn't find floor")

    # public functions

    def print_graph(self, z):
        # Implementation 1 ( building from pure image)
        pure = self._get_floor_img(z, "pure")
        img = copy.deepcopy(pure)

        for Nd in self.Nodes[z]:
            img = cv2.circle(
                img, (Nd.coordinates[0], Nd.coordinates[1]), 8, (66, 126, 255), -1, cv2.LINE_AA)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(Nd.identity), (Nd.coordinates[0] + 10, Nd.coordinates[1] + 10), font, 1,
                        (66, 126, 255), 2, cv2.LINE_AA)
            for edge in Nd.links:
                Nd2 = self.get_node(edge.dest, z)
                if Nd2 is not None:
                    img = cv2.arrowedLine(img, (Nd.coordinates[0], Nd.coordinates[1]),
                                          (Nd2.coordinates[0], Nd2.coordinates[1]), (66, 126, 255), 1,
                                          cv2.LINE_AA)
                else:
                    raise Exception("linkId does not exists")

        # Implementation 2 ( directly taking impure image )
        # impure = self._get_floor_img(z, "impure")
        # img = impure
        cv2.imshow('Node graph for floor ' + str(z), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def print_graph_and_return(self, z):
        pure = self._get_floor_img(z, "pure")
        img = copy.deepcopy(pure)

        for Nd in self.Nodes[z]:
            img = cv2.circle(
                img, (Nd.coordinates[0], Nd.coordinates[1]), 8, (66, 126, 255), -1, cv2.LINE_AA)
            for edge in Nd.links:
                Nd2 = self.get_node(edge.dest, z)
                if Nd2 is not None:
                    img = cv2.line(img, (Nd.coordinates[0], Nd.coordinates[1]),
                                   (Nd2.coordinates[0], Nd2.coordinates[1]), (66, 126, 255), 1,
                                   cv2.LINE_AA)
                else:
                    raise Exception("linkId does not exists")
        return img

    def mark_nodes(self, z):
        if len(self.Nodes) <= z:
            for i in range(z + 1 - len(self.Nodes)):
                self.Nodes.append([])
        window_text = 'Mark Nodes for floor ' + str(z)

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                identity = self.new_node_index
                if self._nearest_node(x, y, z) is None:
                    self._create_node('Node-' + str(identity), x, y, z)
                    cv2.circle(img, (x, y), 8, (66, 126, 255), -1, cv2.LINE_AA)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, str(identity), (x + 10, y + 10), font, 1, (66, 126, 255), 2, cv2.LINE_AA)
                    cv2.imshow(window_text, img)

        impure = self._get_floor_img(z, "impure")
        img = impure
        cv2.imshow(window_text, img)
        cv2.setMouseCallback(window_text, click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def make_connections(self, z):
        nd = None
        window_text = "Make connections for floor " + str(z)

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                global nd
                nd = self._nearest_node(x, y, z)
            elif event == cv2.EVENT_LBUTTONUP:
                if nd is not None:
                    ndcur = self._nearest_node(x, y, z)
                    self._connect(nd, ndcur)
                    cv2.arrowedLine(img, (nd.coordinates[0], nd.coordinates[1]),
                                    (ndcur.coordinates[0], ndcur.coordinates[1]), (66, 126, 255), 1, cv2.LINE_AA)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    meanx = (nd.coordinates[0] + ndcur.coordinates[0]) // 2
                    meany = (nd.coordinates[1] + ndcur.coordinates[1]) // 2
                    cv2.putText(img, str(nd.identity) + "_" + str(ndcur.identity), (meanx + 5, meany + 5), font, 0.5,
                                (100, 126, 255), 2, cv2.LINE_AA)
                    cv2.imshow(window_text, img)

        impure = self._get_floor_img(z, "impure")
        img = impure
        if img is None:
            return
        cv2.imshow(window_text, img)
        cv2.setMouseCallback(window_text, click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def delete_nodes(self, z):
        window_text = 'Delete Nodes for floor ' + str(z)

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if self._nearest_node(x, y, z) is not None:
                    nd = self._nearest_node(x, y, z)
                    self._delete_node(nd)
                    img = self.print_graph_and_return(z)
                    cv2.imshow(window_text, img)

        # impure = self._get_floor_img(z, "impure")
        img = self.print_graph_and_return(z)
        cv2.imshow(window_text, img)
        cv2.setMouseCallback(window_text, click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def delete_connections(self, z):
        nd = None

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                global nd
                nd = self._nearest_node(x, y, z)
            elif event == cv2.EVENT_LBUTTONUP:
                if nd is not None:
                    ndcur = self._nearest_node(x, y, z)
                    for edge in nd.links:
                        if edge.dest == ndcur.identity:
                            nd.links.remove(edge)
                    for edge in ndcur.links:
                        if edge.dest == nd.identity:
                            ndcur.links.remove(edge)
                    img = self.print_graph_and_return(z)
                    cv2.imshow('Delete connections', img)

        img = self.print_graph_and_return(z)
        cv2.imshow('Delete connections', img)
        cv2.setMouseCallback('Delete connections', click_event)
        cv2.waitKey(0)
        cv2.imwrite('nodegraph.jpg', img)
        cv2.destroyAllWindows()

    def read_edges(self, folder, frames_skipped=0, check_blurry=True):
        if os.path.isdir(folder):
            for vid in os.listdir(folder):
                name, type = vid.split(".")
                src, dest = name.split("_")
                self._add_edge_data(int(src), int(dest), folder + "/" + vid, "edge_data/edge_" + str(name),
                                    frames_skipped, check_blurry)

    def read_nodes(self, folder, frames_skipped=0, check_blurry=True):
        if os.path.isdir(folder):
            for vid in os.listdir(folder):
                identity, type = vid.split(".")
                self._add_node_data(int(identity), folder + "/" + vid, "node_data/node_" + str(identity),
                                    frames_skipped, check_blurry)

    def add_floor_map(self, floor_no, path):
        if floor_no > self.no_of_floors:
            raise Exception("Add floor " + str(self.no_of_floors) + " first!!")
        img = cv2.imread(path)
        if img is not None:
            floor_map = FloorMap(floor_no, img)
            self.Floor_map.append(floor_map)
            self.no_of_floors = self.no_of_floors + 1
        else:
            raise Exception("Cannot read image path")

    def save_graph(self):
        general.save_to_memory(self, "graph.pkl")

    def on_node(self, identity):
        if len(self.path_traversed) > 0:
            if type(self.path_traversed[-1]) == int:
                prev_node = self.path_traversed[-1]
                edge = self.get_edge(identity, prev_node)
                if edge is not None:
                    self.path_traversed.append((prev_node, identity, 1))
        self.path_traversed.append(identity)

    def on_edge(self, src: int, dest: int, fraction_traversed: float = 0.5):
        if len(self.path_traversed) > 0:
            prev = self.path_traversed[-1]
            if type(prev) == tuple:
                if prev[0] == src:
                    self.path_traversed[-1] = (src, dest, fraction_traversed)
                    return
                else:
                    self.path_traversed[-1] = (prev[0], prev[1], 1)
                    self.path_traversed.append(prev[1])
            if self.path_traversed[-1] != src:
                self.path_traversed.append(src)
        else:
            self.path_traversed.append(src)
        self.path_traversed.append((src, dest, fraction_traversed))

    def display_path(self, z):
        img = self.print_graph_and_return(0)
        for item in self.path_traversed:
            if type(item) == int:
                nd = self.get_node(item)
                img = cv2.circle(img, (nd.coordinates[0], nd.coordinates[1]), 10, (150, 0, 0), -1, cv2.LINE_AA)
            if type(item) == tuple:
                src_nd = self.get_node(item[0])
                dest_nd = self.get_node(item[1])
                start_coordinates = (src_nd.coordinates[0], src_nd.coordinates[1])
                end_coordinates = (int(src_nd.coordinates[0] + item[2] * (dest_nd.coordinates[0] -
                                                                          src_nd.coordinates[0])),
                                   int(src_nd.coordinates[1] + item[2] * (dest_nd.coordinates[1] -
                                                                          src_nd.coordinates[1])))
                img = cv2.line(img, start_coordinates, end_coordinates, (150, 0, 0), 6,
                               cv2.LINE_AA)
        if len(self.path_traversed) > 0:
            last = self.path_traversed[-1]
            if type(last) == int:
                nd = self.get_node(last)
                img = cv2.circle(img, (nd.coordinates[0], nd.coordinates[1]), 15, (0, 200, 0), -1, cv2.LINE_AA)
            if type(last) == tuple:
                src_nd = self.get_node(last[0])
                dest_nd = self.get_node(last[1])
                end_coordinates = (int(src_nd.coordinates[0] + last[2] * (dest_nd.coordinates[0] -
                                                                          src_nd.coordinates[0])),
                                   int(src_nd.coordinates[1] + last[2] * (dest_nd.coordinates[1] -
                                                                          src_nd.coordinates[1])))
                img = cv2.circle(img, end_coordinates, 15, (0, 200, 0), -1, cv2.LINE_AA)

        cv2.imshow("Current location", img)
        cv2.waitKey(1)

    @staticmethod
    def load_graph():
        return general.load_from_memory("graph.pkl")


def load_graph():
    return general.load_from_memory("graph.pkl")


class node_and_image_matching:
    def __init__(self):
        self.matched_nodes = []
        self.matched_edges = []
        self.final_path = []

    def convert_query_video_to_objects(path, destination_folder):
        return vo2.save_distinct_ImgObj(path, destination_folder)

    """ Assume that a person starts from a specific node.
    Query on all nodes.
    Store the nodes with maximum match"""

    def locate_node(self, nodes_list: vo2.DistinctFrames, query_video_frames: vo2.DistinctFrames,
                    no_of_frames_of_query_video_to_be_matched: int = 2):
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
                        print(j)
                        print(k)
                        print()
                        confidence = confidence + 1
            if confidence / no_of_frames_of_query_video_to_be_matched > 0.32:
                self.matched_nodes.append(node)
        for nd in self.matched_nodes:
            print(nd.name)

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
                    self.locate_edge(nodes_list, query_video_frames, last_frame_object[0])
                    break

    def locate_edge(self, nodes_list: vo2.DistinctFrames, query_video_frames: vo2.DistinctFrames,
                    query_video_frames_begin: int = 0, confidence_level: int = 2):
        last_frame_matched_with_edge = None
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
                    # iterating over a kth edge in edge list
                    image_fraction_matched = mt.SURF_match_2(
                        edge_list[j][0].distinct_frames.get_object(k).get_elements(),
                        query_video_frames.get_object(i).get_elements(), 2500, 0.7)
                    print("query frame " + str(i))
                    print("query frame" + str(k))
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
                    edge_list[j][1] = edge_list[j][1] - 1
                    print(str(edge_list[j][0].src) + "_" + str(
                        edge_list[j][0].dest) + " has decreased confidence = " + str(edge_list[j][1]))
                if edge_list[j][1] < (-1) * confidence_level:
                    print(str(edge_list[j][0].src) + "_" + str(edge_list[j][0].dest) + "deleted")
                    del edge_list[j]
                elif edge_list[j][1] > confidence_level or len(edge_list) == 1:
                    print("edge found")
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


def run(code: int):
    # Create new graph
    if code == 0:
        graph = load_graph()
        #graph.add_floor_map(0, "graph/maps/map0.jpg")
        #graph.mark_nodes(0)
        graph.make_connections(0)
        graph.print_graph(0)
        graph.save_graph()

    # Print graph
    if code == 1:
        graph = load_graph()
        graph.print_graph(0)

    # Add nodes and edges
    if code == 2:
        graph: Graph = load_graph()
        #graph.read_nodes("testData/night sit 0 june 18/node data", 4)
        graph.read_edges("testData/night sit 0 june 18/edge data", 4)
        graph.save_graph()

    # Query video
    if code == 3:
        query_video_frames1 = vo2.save_distinct_ImgObj("testData/sit-june3/VID_20190603_110640.mp4",
                                                       "query_distinct_frame", 0, True)
        # query_video_frames1 = vo2.read_images("query_distinct_frame")
        graph = load_graph()
        node_and_image_matching_obj = node_and_image_matching()
        node_and_image_matching_obj.locate_node(graph.Nodes, query_video_frames1)
        node_and_image_matching_obj.locate_edge(graph.Nodes, query_video_frames1)
        node_and_image_matching_obj.print_final_path()

    # Add specific node/edge data manually
    if code == 4:
        FRAMES1 = vo2.read_images_jpg("testData/node 2 - 6")
        FRAMES2 = vo2.read_images_jpg("testData/Photo frames sit 0/3")
        graph1 = load_graph()
        graph1._add_edge_images(2, 6, FRAMES1)
        graph1._add_node_images(3, FRAMES2)
        graph1.save_graph()

    # Add node images
    if code == 5:
        graph = load_graph()
        graph.read_nodes_directly("testData/Node-direct-images")
        graph.save_graph()

# image = cv2.imread('graph/maps/map0.jpg')
# image = cv2.resize(image, (0, 0), None, .5, 0.5)
#
# grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
#
# numpy_horizontal = np.hstack((image, grey_3_channel))
# numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)
#
# cv2.imshow('Numpy Horizontal', numpy_horizontal)
# cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)
#
# cv2.waitKey()
# run(1)
#run(2)
# run(2)
