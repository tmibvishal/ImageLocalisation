import video_operations_2 as vd2
import re


class Node:
    def __init__(self, no):
        self.no = no
        self.connections = []


class Graph:

    def __init__(self):
        self.no_of_nodes = 0
        self.graph = []

    def add_node(self):
        node_id = self.no_of_nodes
        new_node = Node(node_id)
        self.graph.append(new_node)
        self.no_of_nodes = self.no_of_nodes + 1
        return node_id

    def add_edge(self, src, dest, edge_data):
        if (src or dest) >= self.no_of_nodes:
            raise Exception("invalid src / dest id")

        nd = self.graph[src]

        # edge_id = str(src)+"-"+str(dest)
        nd.connections.append((dest, edge_data))
        self.graph[src] = nd

        # self.no_of_edges = self.no_of_edges + 1

    def remove_edge(self, src, dest):
        if (src or dest) >= self.no_of_nodes:
            raise Exception("invalid src / dest id")

        for edge in self.graph[src].connections:
            if edge[0] == dest:
                self.graph[src].connections.remove(edge)
                break


    def get_nodes(self, edge_id):
        src, dest = re.split('-',edge_id)
        if (src or dest) >=self.no_of_nodes:
            raise Exception("invalid edge id") 
        return src,dest
    
    def get_edge_data(self, edge_id):
        src, dest = self.get_nodes(edge_id)
        for edge in self.graph[src].connections:
            if edge[0] == dest:
                return edge[1]


    def search_edge(self, src, dest=None):
        if dest is None:
            if src >= self.no_of_nodes:
                return False

            nd = self.graph[src]

            for edge in nd.connections:
                edge_data = edge[1]

                # do stuff here

            return True

        else:
            if(src or dest) >= self.no_of_nodes:
                return False

            nd = self.graph[src]

            for edge in nd.connections:
                if edge[0] is dest:

                    # do stuff here

                    return True

# accomodate_path : not tested yet
# edges updated only in graph and not in storage
# not complete for multiple edge matching
    def accomodate_path(self, frames1):
        frames1_start_time = frames1[0][0]
        frames1_end_time = frames1[-1][0]
        for nd in self.graph:
            src = nd.no
            for edge in nd.connections:
                dest = edge[0]
                frames2 = edge[1]
                frames2_start_time = frames2[0][0]
                frames2_end_time = frames2[-1][0]
                matches_found = vd2.compare_videos(frames1, frames2)
                if len(matches_found) != 0:
                    for match in matches_found:
                        edge_start_node, edge_end_node = None, None
                        if match[2] != 0:
                            if match[3] != len(frames2)-1:
                                new_node_1 = self.add_node()
                                new_node_2 = self.add_node()
                                self.remove_edge(src, dest)
                                self.add_edge(src, new_node_1, frames2[0:match[2]])
                                self.add_edge(new_node_1, new_node_2, frames2[match[2]:match[3]+1])
                                self.add_edge(new_node_2, dest, frames2[match[3]+1:])
                                src = new_node_2
                                edge_start_node, edge_end_node = new_node_1, new_node_2
                            else:
                                new_node_1 = self.add_node()
                                self.remove_edge(src, dest)
                                self.add_edge(src, new_node_1, frames2[0:match[2]])
                                self.add_edge(new_node_1, dest, frames2[match[2]:])
                                src = dest
                                edge_start_node, edge_end_node = new_node_1, dest
                        else:
                            if match[3] != len(frames2)-1:
                                new_node_1 = self.add_node()
                                self.remove_edge(src, dest)
                                self.add_edge(src, new_node_1, frames2[0:match[3]+1])
                                self.add_edge(new_node_1, dest, frames2[match[3]+1:])
                                edge_start_node, edge_end_node = src, new_node_1
                                src = new_node_1
                            else:
                                # do nothing
                                edge_start_node, edge_end_node = src, dest
                                src = dest

                        if match[0] != 0: # currently added for 1st match only, have to update for further matches
                            if match[1] != len(frames1)-1:
                                new_node_1 = self.add_node()
                                new_node_2 = self.add_node()
                                self.add_edge(new_node_1, edge_start_node, frames1[0:match[0]])
                                self.add_edge(edge_end_node, new_node_2, frames1[match[1]+1:])
                            else:
                                new_node_1 = self.add_node()
                                self.add_edge(new_node_1, edge_start_node, frames1[0:match[0]])
                        else:
                            if match[1] != len(frames1) - 1:
                                new_node_1 = self.add_node()
                                self.add_edge(edge_end_node, new_node_1, frames1[match[1]+1:])
                            # else:
                            # do nothing

    def print_graph(self):
        for i in range(self.no_of_nodes):
            print("List of vertex {} ".format(i))
            nd = self.graph[i]
            for edge in nd.connections:
                print(str(edge[0])+" : "+str(edge[1]))
            print("")


graph = Graph()
for i in range(5):
    a = graph.add_node()
graph.add_edge(0, 1, "data_0_to_1")
graph.add_edge(0, 3, "data_0_to_3")
graph.add_edge(1, 2, "data_1_to_2")
graph.add_edge(2, 4, "data_2_to_4")
graph.add_edge(3, 5, "data_3_to_5")
graph.print_graph()
