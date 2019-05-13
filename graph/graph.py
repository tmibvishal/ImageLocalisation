class Node:
        def __init__(self, id, name, x__y__z, link):
            self.id = id
            self.name = name
            self.coordinates = (x,y,z)
            self.link = link

class Graph:

    Nodes = {}
    Length = 0

    def __init__(self):
        return

    def create_node(self, name, x__y__z, link):
        id = self.Length
        Nd = Node(id, name, (x,y,z), link)
        self.add_node(Nd)
        
    def add_node(self, Nd):
        if(isinstance(Nd, Node) and Nd.name not in self.Nodes):
            # Check if Nd has the format of Node or not
            if(isinstance(Nd.list, list)):
                # Check if Nd.list is a list
                if(len(Nd.list) != 0):
                    if(isinstance(Nd.list[0], int)):
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

    
            