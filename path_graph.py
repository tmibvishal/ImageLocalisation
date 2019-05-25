
class Node:
    def __init__(self, no, ):
        self.no = no
        self.connections = []
        #self.node_data = node_data

class Edge:
    def __init__(self, src, dest, edge_data):
        self.src =src
        self.dest = dest
        self.edge_data= edge_data


class Graph:

    def __init__(self):
        self.length = 0
        self.graph = []
        self.edges =[]

    def add_node(self):
        node_id = self.length
        new_node = Node(node_id)
        self.graph.append(new_node)
        self.length = self.length + 1

    def add_edge(self, src, dest, edge_data):
        if (src or dest) >= self.length:
            raise Exception("invalid params")

        nd = self.graph[src]
        nd.connections.append((dest, edge_data))

        self.graph[src] = nd
        new_edge = Edge(src, dest, edge_data)
        self.edges.append(new_edge)
        return new_edge

    #def remove_edge(self, src, dest)
    def search_edge(self, src, dest=None):
        if dest is None:
            if src >= self.length:
                return False

            nd = self.graph[src]

            for edge in nd.connections:
                edge_data = edge[1]

                # do stuff here

            return True

        else:
            if(src or dest) >= self.length:
                return False

            nd = self.graph[src]

            for edge in nd.connections:
                if edge[0] is dest:

                    # do stuff here

                    return True

    def print_graph(self):
        for i in range(self.length):
            print("List of vertex {} ".format(i))
            nd = self.graph[i]
            for edge in nd.connections:
                print(str(edge[0])+" : "+str(edge[1]))
            print("")


# graph = Graph()
# for i in range(5):
#     graph.add_node()
# graph.add_edge(0, 1, "data_0_to_1")
# graph.add_edge(0, 3, "data_0_to_3")
# graph.add_edge(1, 2, "data_1_to_2")
# graph.add_edge(2, 4, "data_2_to_4")
# graph.add_edge(3, 5, "data_3_to_5")
# graph.print_graph()
