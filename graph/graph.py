import numpy as np
import cv2
class Node:
        def __init__(self, id, name, x__y__z, links):
            self.id = id
            self.name = name
            self.coordinates = (x,y,z)
            self.links = links

class Graph:

    Nodes = {}
    Length = 0

    def __init__(self):
        return

    def create_node(self, name, x__y__z, links):
        id = self.Length
        Nd = Node(id, name, (x,y,z), links)
        self.add_node(Nd)
        
    def add_node(self, Nd):
        if(isinstance(Nd, Node) and Nd.name not in self.Nodes):
            # Check if Nd has the format of Node or not
            if(isinstance(Nd.links, list)):
                # Check if Nd.links is a list
                if(len(Nd.links) != 0):
                    if(isinstance(Nd.links[0], int)):
                        # Check if Nd.list is a list of integers
                        self.Nodes[Nd.id] = Nd
                        self.Length += 1
                    else:
                        raise Exception("Nd.list is not a list of ids (which are integers)")
                else:
                    self.Nodes[Nd.id] = Nd
                    self.Length += 1
            else:
                raise Exception("Nd.list is not a list")
        else:
            raise Exception("Nd format is not of Node")

    def get_node(self, id):
        if(id<self.Length):
            return Nodes[id]
        else:
            raise Exception("id does not exists")

    def print_graph(self):
        img = np.ones([512,512,3], np.uint8)
        for Nd in self.Nodes:
            img = cv2.circle(img, (Nd.coordinates[0],Nd.coordinates[1]), 2, (0,0,0), -1)
            for linkId in Nd.links:
                if(linkId<self.Length):
                    img = cv2.arrowedLine(img, (Nd.coordinates[0],Nd.coordinates[1]) , (Nodes[linkId].coordinates[0], Nodes[linkId].coordinates[1]) , (0,0,0), 3, cv2.LINE_AA)
                else:
                    raise Exception("linkId does not exists")
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


                
