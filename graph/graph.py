import cv2

class Node:
    def __init__(self, id, name, x, y, z, links):
        self.id = id
        self.name = name
        self.coordinates = (x, y, z)
        self.links = links

class Edge:
    def __init__(self,isConnected:bool=False,seqOfImagesfromVideo=None, videoLength:float=None):
        self.isConnected = isConnected
        self.seqOfImagesfromVideo = seqOfImagesfromVideo
        self.videoLength = videoLength

class Graph:
    Nodes = []
    Length = 0
    edgesMatrix = []

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
                    for row in self.edgesMatrix:
                        row.append(Edge())
                    self.edgesMatrix.append([Edge()] * (len(self.edgesMatrix)+1))
                else:
                    self.Nodes.append(Nd)
                    self.Length = self.Length + 1
                    Ed = Edge()
                    for row in self.edgesMatrix:
                        row.append(Edge())
                    self.edgesMatrix.append([Edge()] * (len(self.edgesMatrix)+1))
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
            if(nd2.id<self.Length and nd1.id<self.Length):
                nd1.links.add(nd2.id)
                nd2.links.add(nd1.id)
                id1 = nd1.id
                id2 = nd2.id
                print(self.edgesMatrix[id1][id2])
                (self.edgesMatrix[id1][id2]).isConnected = True
                (self.edgesMatrix[id2][id1]).isConnected = True
            else:
                raise Exception("Wrong ids of Nodes")
        else:
            raise Exception("Nd format is not of Node, or is already present")

    def image_string(self, z, params):
        # pure = 1 : original (not scattered with nodes) image of map
        # pure = -1 : Graph ( with nodes ) of map only
        # In implementation, the original map image is assigned to img object named 'pure', and node one to
        # 'impure' img object
        if params=="pure":
            return 'images/map'+str(z)+'.jpg'
        elif params=="impure":
            return 'images/nodegraph'+str(z)+'.jpg'
        else:
            raise Exception("Wrong params passed")

    def print_graph(self, z):
        pure = self.image_string(z, "pure")
        impure = self.image_string(z, "impure")
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

