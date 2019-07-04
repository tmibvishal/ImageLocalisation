import cv2
import video_operations_3 as vo2
import os
import general
import copy
import matcher as mt
import shutil
import time
import numpy as np
import math
import image_in_one_frame as one_frame


class Node:
    def __init__(self, identity: int, name: str, x: int, y: int, z: int):
        self.identity = identity
        self.name = name
        self.coordinates = (x, y, z)
        self.links = []
        self.node_images = None

    def __str__(self):
        return str(self.identity)


class Edge:
    def __init__(self, is_connected: bool, src: int, dest: int, distinct_frames=None, video_length: int = None,
                 angles=None):
        self.src = src
        self.dest = dest
        self.distinct_frames = distinct_frames
        self.video_length = video_length
        self.name = str(src) + "_" + str(dest)
        self.angles = angles  # list of form (edge_name, angle)

    def __str__(self):
        return self.name


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

    def _get_edge_slope(self, edge: Edge, floor: int = 0):
        src = edge.src
        dest = edge.dest
        src_node = None
        dest_node = None
        for nd in self.Nodes[floor]:
            if nd.identity == src:
                src_node = nd
            if nd.identity == dest:
                dest_node = nd
        src1 = src_node.coordinates[0]
        src2 = src_node.coordinates[1]
        dest1 = dest_node.coordinates[0]
        dest2 = dest_node.coordinates[1]
        slope_in_degree = None
        if (dest1 - src1) == 0:
            slope_in_degree = 90
        elif (-1) * (dest2 - src2) > 0 and (dest1 - src1) > 0:
            slope = (-1) * (dest_node.coordinates[1] - src_node.coordinates[1]) / (
                        dest_node.coordinates[0] - src_node.coordinates[0])
            slope_in_degree = math.degrees(math.atan(slope))
        elif (-1) * (dest2 - src2) > 0 and (dest1 - src1) < 0:
            slope = (-1) * (dest_node.coordinates[1] - src_node.coordinates[1]) / (
                        dest_node.coordinates[0] - src_node.coordinates[0])
            slope_in_degree = math.degrees(math.atan(slope))
            slope_in_degree = 180 + slope_in_degree
        elif (-1) * (dest2 - src2) < 0 and (dest1 - src1) < 0:
            slope = (-1) * (dest_node.coordinates[1] - src_node.coordinates[1]) / (
                        dest_node.coordinates[0] - src_node.coordinates[0])
            slope_in_degree = math.degrees(math.atan(slope))
            slope_in_degree = 180 + slope_in_degree - 360
        elif (-1) * (dest2 - src2) < 0 and (dest1 - src1) > 0:
            slope = (-1) * (dest_node.coordinates[1] - src_node.coordinates[1]) / (
                        dest_node.coordinates[0] - src_node.coordinates[0])
            slope_in_degree = math.degrees(math.atan(slope))
            slope_in_degree = 360 + slope_in_degree - 360
        else:
            print("no such cased exists")

        return slope_in_degree

    def _get_angle_between_two_edges(self, edge1: Edge, edge2: Edge, floor: int = 0):
        slope1 = self._get_edge_slope(edge1, floor)
        print(edge1.name + str(slope1))
        slope2 = self._get_edge_slope(edge2, floor)
        print(edge2.name + str(slope2))
        slope_diff = slope2 - slope1
        if slope_diff > 180:
            slope_diff = slope_diff - 360
        if slope_diff < (-180):
            slope_diff = slope_diff + 360

        return slope_diff

    def _set_specific_edge_angles(self, cur_edge: Edge):
        cur_edge.angles = []
        nd = self.get_node(cur_edge.dest)
        for next_edge in nd.links:
            if next_edge.dest == cur_edge.src:
                ang = 180
            else:
                ang = self._get_angle_between_two_edges(cur_edge, next_edge)
            cur_edge.angles.append((next_edge.name, ang))
        print(cur_edge.name)
        print(cur_edge.angles)
        print()

    def _set_all_angles(self, floor_no=0):
        for nd in self.Nodes[floor_no]:
            for edge in nd.links:
                self._set_specific_edge_angles(edge)

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
                                                   hessian_threshold, ensure_min=True)
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
        cv2.namedWindow('Node graph for floor ' + str(z), cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Node graph for floor ' + str(z), 1600, 1600)
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
                    cv2.namedWindow(window_text, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_text, 1600, 1600)
                    cv2.imshow(window_text, img)

        impure = self._get_floor_img(z, "impure")
        img = impure
        cv2.namedWindow(window_text, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_text, 1600, 1600)
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
                    cv2.namedWindow(window_text, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_text, 1600, 1600)
                    cv2.imshow(window_text, img)

        impure = self._get_floor_img(z, "impure")
        img = impure
        if img is None:
            return
        cv2.namedWindow(window_text, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_text, 1600, 1600)
        cv2.imshow(window_text, img)
        cv2.setMouseCallback(window_text, click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        self._set_all_angles(z)

    def delete_nodes(self, z):
        window_text = 'Delete Nodes for floor ' + str(z)

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if self._nearest_node(x, y, z) is not None:
                    nd = self._nearest_node(x, y, z)
                    self._delete_node(nd)
                    img = self.print_graph_and_return(z)
                    cv2.namedWindow(window_text, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_text, 1600, 1600)
                    cv2.imshow(window_text, img)

        # impure = self._get_floor_img(z, "impure")
        img = self.print_graph_and_return(z)
        cv2.namedWindow(window_text, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_text, 1600, 1600)
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
                    cv2.namedWindow('Delete connections', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Delete connections', 1600, 1600)
                    cv2.imshow('Delete connections', img)

        img = self.print_graph_and_return(z)
        cv2.namedWindow('Delete connections', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Delete connections', 1600, 1600)
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

    def save_graph(self, folder, filename):
        general.ensure_path(folder)
        # new_path = os.path.join(path)
        general.save_to_memory(self, filename, folder)

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
                    if prev[1] == dest and prev[2] > fraction_traversed:
                        return
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

    def display_path(self, z, current_location_str=""):
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

        # cv2.namedWindow("Current location", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("Current location", 1600, 1600)
        cv2.putText(img, current_location_str, (20, 32), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        one_frame.run_graph_frame(img)
        # cv2.imshow("Current location", img)
        # cv2.waitKey(1)

    @staticmethod
    def load_graph(graph_path):
        return general.load_from_memory(graph_path)


def load_graph(graph_path):
    return general.load_from_memory(graph_path)


def run(code: int):
    # Create new graph
    if code == 0:
        graph = Graph()
        graph.add_floor_map(0, "graph/maps/map0.jpg")
        graph.mark_nodes(0)
        graph.make_connections(0)
        graph.print_graph(0)
        graph.save_graph("testData/afternoon_sit0 15june", "graph.pkl")

    # Print graph
    if code == 1:
        graph = load_graph("testData/afternoon_sit0 15june/graph.pkl")
        graph.print_graph(0)

    # Add nodes and edges
    if code == 2:
        graph: Graph = load_graph("new_objects/graph.pkl")
        graph.read_nodes("testData/afternoon_sit0 15june/NodeData", 4)
        graph.read_edges("testData/night sit 0 june 18/Transfer returns", 4)
        graph.save_graph("new_objects", "graph.pkl")

    # # Add specific node/edge data manually
    # if code == 3:
    #     FRAMES1 = vo2.read_images_jpg("testData/node 2 - 6")
    #     FRAMES2 = vo2.read_images_jpg("testData/Photo frames sit 0/3")
    #     graph1 = load_graph()
    #     graph1._add_edge_images(2, 6, FRAMES1)
    #     graph1._add_node_images(3, FRAMES2)
    #     graph1.save_graph("new_objects", "graph.pkl")

