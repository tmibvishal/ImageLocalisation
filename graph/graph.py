import cv2
import video_operations_2 as vo2
import os
import general
import copy


class Node:
    def __init__(self, identity: int, name: str, x: int, y: int, z: int):
        self.identity = identity
        self.name = name
        self.coordinates = (x, y, z)
        self.links = []
        self.node_images = []


class Edge:
    def __init__(self, is_connected: bool, src: int, dest: int, distinct_frames=None, video_length: int = None):
        self.is_connected = is_connected
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
        self.Nodes = []
        self.no_of_floors = 0
        self.Floor_map = []

    # private functions

    def _create_node(self, name, x, y, z):
        identity = self.new_node_index
        Nd = Node(identity, name, x, y, z)
        self._add_node(Nd)

    def _add_node(self, Nd):
        if isinstance(Nd, Node):
            if Nd not in self.Nodes:
                # Check if Nd has the format of Node , and if it is already present
                if isinstance(Nd.links, list):
                    if len(Nd.links) == 0 or isinstance(Nd.links[0], Edge):
                        self.Nodes.append(Nd)
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
        for Nd in self.Nodes:
            if Nd.z == z:
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

    def _delete_node(self, nd):
        if nd in self.Nodes:
            self.Nodes.remove(nd)
            for nd2 in self.Nodes:
                for edge in nd2.links:
                    if nd.identity == edge.dest:
                        nd2.links.remove(edge)
        else:
            raise Exception("Nd does not exists in Nodes")

    def _add_edge_images(self, id1: int, id2: int, distinct_frames: vo2.DistinctFrames):
        if id1 > self.new_node_index or id2 > self.new_node_index:
            raise Exception("Wrong id's passed")
        if not isinstance(distinct_frames, vo2.DistinctFrames):
            raise Exception("Invalid param for distinct_frames")
        for Nd in self.Nodes:
            if Nd.identity == id1:
                for edge in Nd.links:
                    if edge.dest == id2:
                        edge.distinct_frames = distinct_frames
                        edge.video_length = distinct_frames.get_time()
                        return
        raise Exception("Edge from " + str(id1) + " to " + str(id2) + " not found")

    def _add_node_images(self, identity, node_images):
        if not isinstance(node_images, vo2.DistinctFrames):
            raise Exception("node_images is not DistinctFrames object")

        for Nd in self.Nodes:
            if Nd.identity == identity:
                Nd.node_images = node_images
                return
        raise Exception("Node " + str(identity) + " not found!")

    def _add_node_data(self, identity: int, path_of_video: str, folder_to_save: str = None,
                       frames_skipped: int = 0, check_blurry: bool = True, hessian_threshold: int = 2500):
        distinct_frames = vo2.save_distinct_ImgObj(path_of_video, folder_to_save, frames_skipped, check_blurry,
                                                   hessian_threshold)
        self._add_node_images(identity, distinct_frames)

    def _add_edge_data(self, id1: int, id2: int, path_of_video: str, folder_to_save: str = None,
                       frames_skipped: int = 0, check_blurry: bool = True, hessian_threshold: int = 2500):
        distinct_frames = vo2.save_distinct_ImgObj(path_of_video, folder_to_save, frames_skipped, check_blurry,
                                                   hessian_threshold)
        self._add_edge_images(id1, id2, distinct_frames)

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

    def get_node(self, identity):
        for Nd in self.Nodes:
            if identity == Nd.identity:
                return Nd
        return None

    def print_graph(self, z):
        # Implementation 1 ( building from pure image)
        pure = self._get_floor_img(z, "pure")
        img = copy.deepcopy(pure)

        for Nd in self.Nodes:
            if Nd.coordinates[2] == z:
                img = cv2.circle(
                    img, (Nd.coordinates[0], Nd.coordinates[1]), 8, (66, 126, 255), -1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(Nd.identity), (Nd.coordinates[0] + 10, Nd.coordinates[1] + 10), font, 1,
                            (66, 126, 255), 2, cv2.LINE_AA)
                for edge in Nd.links:
                    if edge.is_connected:
                        for Nd2 in self.Nodes:
                            if Nd2.identity == edge.dest:
                                img = cv2.arrowedLine(img, (Nd.coordinates[0], Nd.coordinates[1]),
                                                      (Nd2.coordinates[0], Nd2.coordinates[1]), (66, 126, 255), 1,
                                                      cv2.LINE_AA)
                    else:
                        raise Exception("linkId does not exists")

        # Implementation 2 ( directly taking impure image )
        # impure = self._get_floor_img(z, "impure")
        # img = impure
        return img
        # cv2.imshow('Node graph for floor ' + str(z), img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def mark_nodes(self, z):
        window_text = 'Mark Nodes for floor ' + str(z)

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                identity = self.new_node_index
                if self._nearest_node(x, y, z) is None:
                    self._create_node('Node-' + str(identity), x, y, z)
                    cv2.circle(img, (x, y), 8, (66, 126, 255), -1)
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
                    cv2.putText(img, str(nd.identity) + "_" + str(ndcur.identity), (meanx+5, meany+5), font, 0.5,
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
                    img = self.print_graph(z)
                    cv2.imshow(window_text, img)

        impure = self._get_floor_img(z, "impure")
        img = impure
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
                    img = self.print_graph(z)
                    cv2.imshow('Delete connections', img)

        img = cv2.imread('nodegraph.jpg')
        if img is None:  # No nodes present
            return
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

    @staticmethod
    def load_graph():
        return general.load_from_memory("graph.pkl")


graph = Graph()
graph.add_floor_map(0, "images/map0.jpg")
graph.mark_nodes(0)
graph.make_connections(0)
graph.delete_nodes(0)
graph.save_graph()
