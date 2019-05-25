import cv2
import video_operations_2 as vo2


class Node:
    def __init__(self, identity, name, x, y, z, links=set(), node_images={}):
        self.identity = identity
        self.name = name
        self.coordinates = (x, y, z)
        self.links = links
        self.node_images = node_images


class Edge:
    def __init__(self, is_connected: bool, distinct_frames, video_length: float):
        self.is_connected = is_connected
        self.distinct_frames = distinct_frames
        self.video_length = video_length


class Graph:
    Nodes = []
    Length = 0

    def __init__(self):
        return

    def create_node(self, name, x, y, z, links):
        identity = self.Length
        Nd = Node(identity, name, x, y, z, links)
        self.add_node(Nd)

    def add_node(self, Nd):
        if isinstance(Nd, Node) and Nd not in self.Nodes:
            # Check if Nd has the format of Node or not
            if isinstance(Nd.links, set):
                # Check if Nd.links is a set
                if len(Nd.links) != 0:
                    # AIM: also check that set is a set of integers or not
                    self.Nodes.append(Nd)
                    self.Length = self.Length + 1
                else:
                    self.Nodes.append(Nd)
                    self.Length = self.Length + 1
            else:
                raise Exception("Nd.links is not a set")
        else:
            raise Exception("Nd format is not of Node, or is already present")

    def get_node(self, identity):
        for Nd in self.Nodes:
            if identity == Nd.identity:
                return Nd
        return None

    def nearest_node(self, x, y):
        def distance(nd):
            delx = abs(nd.coordinates[0] - x)
            dely = abs(nd.coordinates[1] - y)
            return delx ** 2 + dely ** 2

        min, nearest_node = -1, None
        for nd in self.Nodes:
            if abs(nd.coordinates[0] - x) < 50 and abs(nd.coordinates[1] - y) < 50:
                if min == -1 or distance(nd) < min:
                    nearest_node = nd
                    min = distance(nd)
                elif distance(nd) == min:
                    return None
        return nearest_node

    def connect(self, nd1, nd2):
        if isinstance(nd1, Node) and isinstance(nd2, Node):
            if (nd2.identity < self.Length and nd1.identity < self.Length):
                nd1.links.add(nd2.identity)
                nd2.links.add(nd1.identity)
            else:
                raise Exception("Wrong identitys of Nodes")
        else:
            raise Exception("Nd format is not of Node, or is already present")

    def image_string(self, z, params):
        # pure = 1 : original (not scattered with nodes) image of map
        # pure = -1 : Graph ( with nodes ) of map only
        # In implementation, the original map image is assigned to img object named 'pure', and node one to
        # 'impure' img object
        if params == "pure":
            return 'images/map' + str(z) + '.jpg'
        elif params == "impure":
            return 'images/nodegraph' + str(z) + '.jpg'
        else:
            raise Exception("Wrong params passed")

    def print_graph(self, z):
        pure = self.image_string(z, "pure")
        impure = self.image_string(z, "impure")
        img = cv2.imread(pure)

        for Nd in self.Nodes:
            if Nd.coordinates[2] == z:
                img = cv2.circle(
                    img, (Nd.coordinates[0], Nd.coordinates[1]), 8, (66, 126, 255), -1)
                for linkId in Nd.links:
                    if linkId < self.Length:
                        Nd2 = self.Nodes[linkId]
                        img = cv2.line(img, (Nd.coordinates[0], Nd.coordinates[1]),
                                       (Nd2.coordinates[0], Nd2.coordinates[1]), (66, 126, 255), 1, cv2.LINE_AA)
                    else:
                        raise Exception("linkId does not exists")
        cv2.imshow('Node graph for floor ' + str(z), img)
        cv2.waitKey(0)
        cv2.imwrite(impure, img)
        cv2.destroyAllWindows()

    def mark_nodes(self, z):
        window_text = 'Mark Nodes for floor ' + str(z)

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                identity = self.Length
                if self.nearest_node(x, y) is None:
                    self.create_node('Node-' + str(identity), x, y, z, set())
                    cv2.circle(img, (x, y), 8, (66, 126, 255), -1)

                cv2.imshow(window_text, img)

        pure = self.image_string(z, "pure")
        impure = self.image_string(z, "impure")
        # img = cv2.imread(impure)
        # if img is None:
        img = cv2.imread(pure)
        cv2.imshow(window_text, img)
        cv2.setMouseCallback(window_text, click_event)
        cv2.waitKey(0)
        cv2.imwrite(impure, img)
        cv2.destroyAllWindows()

    def make_connections(self, z):
        nd = None
        window_text = "Make connections for floor " + str(z)

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                global nd
                nd = self.nearest_node(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                if nd is not None:
                    ndcur = self.nearest_node(x, y)
                    self.connect(nd, ndcur)
                    cv2.line(img, (nd.coordinates[0], nd.coordinates[1]),
                             (ndcur.coordinates[0], ndcur.coordinates[1]), (66, 126, 255), 1, cv2.LINE_AA)
                    cv2.imshow(window_text, img)

        impure = self.image_string(z, "impure")
        img = cv2.imread(impure)
        if img is None:
            return
        cv2.imshow(window_text, img)
        cv2.setMouseCallback(window_text, click_event)
        cv2.waitKey(0)
        cv2.imwrite(impure, img)
        cv2.destroyAllWindows()

    def delete_node(self, Nd):
        if Nd in self.Nodes:
            self.Nodes.remove(Nd)
            for i in range(len(self.Nodes)):
                if (Nd.identity in self.Nodes[i].links):
                    self.Nodes[i].links.remove(Nd.identity)
        else:
            raise Exception("Nd does not exists in Nodes")

    def delete_connections(self):
        nd = None

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                global nd
                nd = self.nearest_node(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                if nd is not None:
                    ndcur = self.nearest_node(x, y)
                    self.Nodes[nd.identity].links.remove(ndcur.identity)
                    self.Nodes[ndcur.identity].links.remove(nd.identity)
                    img = cv2.imread('map.jpg')
                    for Nd in self.Nodes:
                        img = cv2.circle(
                            img, (Nd.coordinates[0], Nd.coordinates[1]), 8, (66, 126, 255), -1)
                        for linkId in Nd.links:
                            if linkId < self.Length:
                                Nd2 = self.Nodes[linkId]
                                img = cv2.line(img, (Nd.coordinates[0], Nd.coordinates[1]),
                                               (Nd2.coordinates[0], Nd2.coordinates[1]), (66, 126, 255), 1, cv2.LINE_AA)
                            else:
                                raise Exception("linkId does not exists")
                    cv2.imshow('Delete connections', img)

        img = cv2.imread('nodegraph.jpg')
        if img is None:  # No nodes present
            return
        cv2.imshow('Delete connections', img)

        cv2.setMouseCallback('Delete connections', click_event)

        cv2.waitKey(0)
        cv2.imwrite('nodegraph.jpg', img)
        cv2.destroyAllWindows()

    def add_node_images(self, identity, node_images):
        def is_node_images(node_images):
            if (not isinstance(node_images, list) or not len(node_images) == 4):
                return False
            else:
                return True

        i = 0
        while (i < len(self.Nodes)):
            Nd = self.Nodes[i]
            if Nd.identity == identity:
                if(is_node_images(node_images)):
                    self.Nodes[i].node_images = node_images
                else:
                    raise Exception("node_data is not a size 4 array")
            i = i + 1

    def add_edge_data(self, node1: Node, node2: Node, path_of_video: str, folder_to_save: str = None,
                      frames_skipped: int = 0, check_blurry: bool = True, hessian_threshold: int = 2500):
        distict_frames = vo2.save_distinct_ImgObj(path_of_video, folder_to_save, frames_skipped, check_blurry,
                                                  hessian_threshold)
