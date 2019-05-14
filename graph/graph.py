import cv2
import os


class Node:
    def __init__(self, id, name, x, y, z, links):
        self.id = id
        self.name = name
        self.coordinates = (x, y, z)
        self.links = links


class Graph:
    Nodes = []
    Length = 0

    def __init__(self):
        return

    def create_node(self, name, x, y, z, links):
        id = self.Length
        Nd = Node(id, name, x, y, z, links)
        self.add_node(Nd)

    def add_node(self, Nd):
        if isinstance(Nd, Node) and Nd.name not in self.Nodes:
            # Check if Nd has the format of Node or not
            if isinstance(Nd.links, list):
                # Check if Nd.links is a list
                if len(Nd.links) != 0:
                    if isinstance(Nd.links[0], int):
                        # Check if Nd.list is a list of integers
                        self.Nodes.append(Nd)
                        self.Length = self.Length + 1
                    else:
                        raise Exception("Nd.list is not a list of ids (which are integers)")
                else:
                    self.Nodes.append(Nd)
                    self.Length = self.Length + 1
            else:
                raise Exception("Nd.list is not a list")
        else:
            raise Exception("Nd format is not of Node, or is already present")

    def get_node(self, id):
        if id < self.Length:
            return self.Nodes[id]
        else:
            raise Exception("id does not exist")

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
            if nd2.id not in nd1.links:
                nd1.links.append(nd2.id)
            if nd1.id not in nd2.links:
                nd2.links.append(nd1.id)

    def print_graph(self):
        img = cv2.imread('map.jpg')
        for Nd in self.Nodes:
            img = cv2.circle(img, (Nd.coordinates[0], Nd.coordinates[1]), 8, (66, 126, 255), -1)
            for linkId in Nd.links:
                if linkId < self.Length:
                    Nd2 = self.Nodes[linkId]
                    img = cv2.line(img, (Nd.coordinates[0], Nd.coordinates[1]),
                                   (Nd2.coordinates[0], Nd2.coordinates[1]), (66, 126, 255), 1, cv2.LINE_AA)
                else:
                    raise Exception("linkId does not exists")
        cv2.imshow('Node graph', img)
        cv2.waitKey(0)
        cv2.imwrite('nodegraph.jpg', img)
        cv2.destroyAllWindows()

    def mark_nodes(self):
        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                id = self.Length
                if self.nearest_node(x, y) is None:
                    self.create_node('Node-' + str(id), x, y, 0, [])
                    cv2.circle(img, (x, y), 8, (66, 126, 255), -1)
                cv2.imshow('Mark Nodes', img)

        img = cv2.imread('nodegraph.jpg')
        if img is None:
            img = cv2.imread('map.jpg')
        cv2.imshow('Mark Nodes', img)

        cv2.setMouseCallback('Mark Nodes', click_event)

        cv2.waitKey(0)
        cv2.imwrite('nodegraph.jpg', img)
        cv2.destroyAllWindows()

    def make_connections(self):
        nd = None

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
                    cv2.imshow('Make connections', img)

        img = cv2.imread('nodegraph.jpg')
        if img is None:  # No nodes present
            return
        cv2.imshow('Make connections', img)

        cv2.setMouseCallback('Make connections', click_event)

        cv2.waitKey(0)
        cv2.imwrite('nodegraph.jpg', img)
        cv2.destroyAllWindows()


# Deleting nodegraph.jpg ( for initial phase )
if os.path.exists("nodegraph.jpg"):
    os.remove("nodegraph.jpg")
graph = Graph()
graph.mark_nodes()
graph.make_connections()
graph.print_graph()

# To create nodes manually, use
# graph.create_node('entrance', 256, 256, 0, [1])
# graph.create_node('lift-grd', 50, 50, 0, [0])

# To check available events, use
# events = [i for i in dir(cv2) if 'EVENT' in i]
# print(events)

# To use a black image instead of map, use
# import numpy as np
# img = np.ones([512, 512, 3], np.uint8)
# instead of
# img = cv2.imread('map.jpg')

