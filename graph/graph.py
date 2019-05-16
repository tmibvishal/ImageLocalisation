import cv2


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

    def get_node(self, id):
        for Nd in self.Nodes:
            if id == Nd.id:
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
            nd1.links.add(nd2.id)
            nd2.links.add(nd1.id)
        else:
            raise Exception("Nd format is not of Node, or is already present")

    def image_string(self, z, pure):
        # pure = 1 : original (not scattered with nodes) image of map
        # pure = -1 : Graph ( with nodes ) of map only
        # In implementation, the original map image is assigned to img object named 'pure', and node one to
        # 'impure' img object
        if pure:
            return 'images/map'+str(z)+'.jpg'
        else:
            return 'images/nodegraph'+str(z)+'.jpg'

    def print_graph(self, z):
        pure = self.image_string(z, True)
        impure = self.image_string(z, False)
        img = cv2.imread(pure)

        for Nd in self.Nodes:
            if Nd.coordinates[2] == z:
                img = cv2.circle(img, (Nd.coordinates[0], Nd.coordinates[1]), 8, (66, 126, 255), -1)
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
                id = self.Length
                if self.nearest_node(x, y) is None:
                    self.create_node('Node-' + str(id), x, y, z, set())
                    cv2.circle(img, (x, y), 8, (66, 126, 255), -1)

                cv2.imshow(window_text, img)

        pure = self.image_string(z, True)
        impure = self.image_string(z, False)
        img = cv2.imread(impure)
        if img is None:
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

        impure = self.image_string(z, False)
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
                if(Nd.id in self.Nodes[i].links):
                    self.Nodes[i].links.remove(Nd.id)
        else:
            raise Exception("Nd does not exists in Nodes")
            
    def delete_connections(self):
        nd= None

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                global nd
                nd = self.nearest_node(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                if nd is not None:
                    ndcur = self.nearest_node(x,y)
                    self.Nodes[nd.id].links.remove(ndcur.id)
                    self.Nodes[ndcur.id].links.remove(nd.id)
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
                    cv2.imshow('Delete connections', img)


        img = cv2.imread('nodegraph.jpg')
        if img is None:  # No nodes present
            return
        cv2.imshow('Delete connections', img)

        cv2.setMouseCallback('Delete connections', click_event)

        cv2.waitKey(0)
        cv2.imwrite('nodegraph.jpg', img)
        cv2.destroyAllWindows()


# Deleting nodegraph.jpg ( for initial phase )
if os.path.exists("nodegraph.jpg"):
    os.remove("nodegraph.jpg")
graph = Graph()
graph.mark_nodes()
graph.make_connections()
graph.delete_connections()
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

