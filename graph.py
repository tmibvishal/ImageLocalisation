import cv2
import video_operations_2 as vo2
import os
import general
import copy
import matcher as mt
import shutil
import time


class Node:
    def __init__(self, identity:int, name:str, x:int, y:int, z:int):
        self.identity = identity
        self.name = name
        self.coordinates = (x, y, z)
        self.links = []
        self.node_images=None


class Edge:
    def __init__(self, is_connected: bool, src:int, dest:int, distinct_frames=None, video_length:int=None):
        self.is_connected = is_connected
        self.src = src
        self.dest = dest
        self.distinct_frames = distinct_frames
        self.video_length = video_length


class FloorMap:
    def __init__(self, floor_no:int=None, img=None):
        self.floor_no = floor_no
        self.pure = img
        self.impure = copy.deepcopy(img)


class Graph:

    def __init__(self):
        self.Length = 0
        self.Nodes = []
        self.no_of_floors = 0
        self.Floor_map = []

    # private functions

    def _create_node(self, name, x, y, z):
        identity = self.Length
        Nd = Node(identity, name, x, y, z)
        self._add_node(Nd)

    def _add_node(self, Nd):
        if isinstance(Nd, Node) and Nd not in self.Nodes:
            # Check if Nd has the format of Node , and if it is already present
            if isinstance(Nd.links, list):
                if len(Nd.links) == 0 or isinstance(Nd.links[0], Edge):
                    self.Nodes.append(Nd)
                    self.Length = self.Length + 1
            else:
                raise Exception("Nd.links is not a list")
        else:
            raise Exception("Nd format is not of Node, or is already present")

    def _nearest_node(self, x, y):
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

    def _connect(self, nd1, nd2):
        if isinstance(nd1, Node) and isinstance(nd2, Node):
            if (nd2.identity < self.Length and nd1.identity < self.Length):
                edge = Edge(True, nd1.identity, nd2.identity)
                nd1.links.append(edge)
            else:
                raise Exception("Wrong identities of Nodes")
        else:
            raise Exception("Nd format is not of Node, or is already present")

    def _delete_node(self, Nd):
        if Nd in self.Nodes:
            self.Nodes.remove(Nd)
            for nd2 in self.Nodes:
                for edge in nd2.links:
                    if (Nd.identity == edge.dest):
                        nd2.links.remove(edge)
        else:
            raise Exception("Nd does not exists in Nodes")

    def _add_edge_images(self, id1: int, id2: int, distinct_frames: vo2.DistinctFrames):
        if (not isinstance(id1, int)) or (not isinstance(id2, int)) or ((id1 or id2) >= self.Length):
            raise Exception("Wrong id's passed")
        if not isinstance(distinct_frames, vo2.DistinctFrames):
            raise Exception("Invalid param for distinct_frames")
        for nd in self.Nodes:
            if nd.identity == id1:
                for edge in nd.links:
                    if edge.dest == id2:
                        edge.distinct_frames = distinct_frames
                        edge.video_length = distinct_frames.get_time()
                        return
        raise Exception("Edge from "+str(id1)+" to "+str(id2)+" not found")

    def _add_node_images(self, identity, node_images):
        if not isinstance(node_images, vo2.DistinctFrames):
            raise Exception("node_images is not DistinctFrames object")

        for Nd in self.Nodes:
            if Nd.identity == identity:
                    Nd.node_images = node_images
                    return
        raise Exception("Node "+str(identity)+" not found!")


    def _add_node_data(self, identity:int, path_of_video: str, folder_to_save: str = None,
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
        #Implementation 1 ( building from pure image)
        pure = self._get_floor_img(z, "pure")
        img = copy.deepcopy(pure)

        for Nd in self.Nodes:
            if Nd.coordinates[2] == z:
                img = cv2.circle(
                    img, (Nd.coordinates[0], Nd.coordinates[1]), 8, (66, 126, 255), -1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(Nd.identity), (Nd.coordinates[0] + 10, Nd.coordinates[1] + 10), font, 1, (66, 126, 255), 2, cv2.LINE_AA)
                for edge in Nd.links:
                    if edge.is_connected:
                        for Nd2 in self.Nodes:
                            if Nd2.identity == edge.dest:
                                img = cv2.arrowedLine(img, (Nd.coordinates[0], Nd.coordinates[1]),
                                        (Nd2.coordinates[0], Nd2.coordinates[1]), (66, 126, 255), 1, cv2.LINE_AA)
                    else:
                        raise Exception("linkId does not exists")

        # Implementation 2 ( directly taking impure image )
        #impure = self._get_floor_img(z, "impure")
        #img = impure
        cv2.imshow('Node graph for floor ' + str(z), img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def mark_nodes(self, z):
        window_text = 'Mark Nodes for floor ' + str(z)

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                identity = self.Length
                if self._nearest_node(x, y) is None:
                    self._create_node('Node-' + str(identity), x, y, z)
                    cv2.circle(img, (x, y), 8, (66, 126, 255), -1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(img, str(identity),(x+10,y+10), font, 1,(66,126,255),2,cv2.LINE_AA)
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
                nd = self._nearest_node(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                if nd is not None:
                    ndcur = self._nearest_node(x, y)
                    self._connect(nd, ndcur)
                    cv2.arrowedLine(img, (nd.coordinates[0], nd.coordinates[1]),
                             (ndcur.coordinates[0], ndcur.coordinates[1]), (66, 126, 255), 1, cv2.LINE_AA)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    meanx = (nd.coordinates[0] + ndcur.coordinates[0])//2
                    meany = (nd.coordinates[1] + ndcur.coordinates[1])//2
                    cv2.putText(img, str(nd.identity)+"_"+str(ndcur.identity), ( meanx, meany), font, 1, (100, 126, 255), 2, cv2.LINE_AA)
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
                if self._nearest_node(x, y) is not None:
                    nd = self._nearest_node(x, y)
                    self._delete_node(nd)
                    img = self.print_graph(z)
                    cv2.imshow(window_text, img)

        impure = self._get_floor_img(z, "impure")
        img = impure
        cv2.imshow(window_text, img)
        cv2.setMouseCallback(window_text, click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def delete_connections(self):
        nd = None

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                global nd
                nd = self._nearest_node(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                if nd is not None:
                    ndcur = self._nearest_node(x, y)
                    for edge in nd.links:
                        if edge.dest == ndcur.identity:
                            nd.links.remove(edge)
                    for edge in ndcur.links:
                        if edge.dest == nd.identity:
                            ndcur.links.remove(edge)
                    img = cv2.imread('map.jpg')
                    for Nd in self.Nodes:
                        img = cv2.circle(
                            img, (Nd.coordinates[0], Nd.coordinates[1]), 8, (66, 126, 255), -1)
                        for edge in Nd.links:
                                Nd2 = self.get_node(edge.dest)
                                img = cv2.arrowedLine(img, (Nd.coordinates[0], Nd.coordinates[1]),
                                               (Nd2.coordinates[0], Nd2.coordinates[1]), (66, 126, 255), 1, cv2.LINE_AA)
                    cv2.imshow('Delete connections', img)

        img = cv2.imread('nodegraph.jpg')
        if img is None:  # No nodes present
            return
        cv2.imshow('Delete connections', img)

        cv2.setMouseCallback('Delete connections', click_event)

        cv2.waitKey(0)
        cv2.imwrite('nodegraph.jpg', img)
        cv2.destroyAllWindows()

    def read_edges(self, folder, frames_skipped = 0, check_blurry = True):
        if os.path.isdir(folder):
            for vid in os.listdir(folder):
                name, type = vid.split(".")
                src, dest = name.split("_")
                self._add_edge_data(int(src), int(dest), folder + "/" + vid, "edge_data/edge_"+str(name), frames_skipped, check_blurry)

    def read_nodes(self, folder, frames_skipped = 0, check_blurry = True):
        if os.path.isdir(folder):
            for vid in os.listdir(folder):
                identity,type = vid.split(".")
                self._add_node_data(int(identity), folder + "/" + vid, "node_data/node_"+str(identity), frames_skipped, check_blurry)

    def read_edges_from_images(self, folder):
        if os.path.isdir(folder):
            for edge_folder in os.listdir(folder):
                e, src, dest = edge_folder.split('_')
                distinct_frames_obj = vo2.read_images(folder+"/"+edge_folder)
                for nd in self.Nodes:
                    if nd.identity == src:
                        for edge in nd.links:
                            if edge.dest == dest:
                                edge.distinct_frames = distinct_frames_obj
                                break

    def read_nodes_from_images(self, folder):
        if os.path.isdir(folder):
            for node_folder in os.listdir(folder):
                n, identity = node_folder.split('_')
                distinct_frames_obj = vo2.read_images(folder+"/"+node_folder)
                for node in self.Nodes:
                    if node.identity == identity:
                        node.node_images = distinct_frames_obj
                        break

    def add_floor_map(self, floor_no, path):
        if floor_no > self.no_of_floors:
            raise Exception("Add floor "+str(self.no_of_floors)+" first!!")
        img = cv2.imread(path)
        if img is not None:
            floor_map = FloorMap(floor_no, img)
            self.Floor_map.append(floor_map)
            self.no_of_floors = self.no_of_floors + 1
        else:
            raise Exception("Cannot read image path")

    def save_graph(self):
        general.save_to_memory(self, "graph.pkl")

# @staticmethod
def load_graph():
    return general.load_from_memory("graph.pkl")




class node_and_image_matching:
    def __init__(self):
        self.matched_nodes=[]
        self.matched_edges=[]
        self.final_path=[]

    def convert_query_video_to_objects(path, destination_folder):
        return vo2.save_distinct_ImgObj(path,destination_folder)

    """ Assume that a person starts frpom a specific node.
    Query on all nodes.
    Store the nodes with maximum match"""

    def locate_node(self,nodes_list, query_video_frames, no_of_frames_of_query_video_to_be_matched:int=2):
        if len(self.matched_nodes)!=0:
            self.matched_nodes=[]
        for node in nodes_list:
            confidence=0
            node_images= node.node_images
            if(node_images == None):
                continue
            no_of_node_images= node_images.no_of_frames()
            for j in range(no_of_frames_of_query_video_to_be_matched):
                for k in range(no_of_node_images):
                    image_fraction_matched = mt.SURF_match_2(query_video_frames.get_object(j).get_elements(), node_images.get_object(k).get_elements(),
                                                         2500, 0.7)
                    if image_fraction_matched> 0.15:
                        print(image_fraction_matched)
                        print(j)
                        print(k)
                        print()
                        confidence= confidence+1
            if confidence/no_of_frames_of_query_video_to_be_matched >0.32:
                self.matched_nodes.append(node)
        for nd in self.matched_nodes:
            print(nd.name)


    def match_next_node(self, nodes_list, query_video_frames1, last_frame_matched_with_edge):
        if len(self.matched_nodes)!=0:
            self.matched_nodes=[]
        new_src_node= last_frame_matched_with_edge[1].dest
        for node in nodes_list:
            if node.identity== new_src_node:
                print("hi")




    def locate_edge(self, nodes_list, query_video_frames, confidence_level: int = 2):
        last_frame_matched_with_edge=None
        for node in self.matched_nodes:
            for edge in node.links:
                self.matched_edges.append([edge, 0, 0])  # (edge, confidence, frame_position_matched)
                print("edges added"+str(edge.src)+"_"+str(edge.dest))
                print()

        for i in range(query_video_frames.no_of_frames()):
            edge_list=self.matched_edges
            j=0
            while j <len(edge_list):
                match, maximum_match = None, 0
                for k in range(int(edge_list[j][2]), edge_list[j][0].distinct_frames.no_of_frames()):  # starting from matched_edges[j][2] till end #edge folder
                    image_fraction_matched = mt.SURF_match_2(edge_list[j][0].distinct_frames.get_object(k).get_elements(),
                        query_video_frames.get_object(i).get_elements(), 2500, 0.7)
                    if image_fraction_matched > 0.15:
                        if image_fraction_matched > maximum_match:
                            last_frame_matched_with_edge=(i,edge_list[j][0])
                            print(last_frame_matched_with_edge)
                            print(image_fraction_matched)
                            print(i)
                            print(str(edge_list[j][0].src)+"_"+str(edge_list[j][0].dest))
                            print(k)
                            match, maximum_match = k, image_fraction_matched
                if match is not None:
                    edge_list[j][1] = edge_list[j][1] + 1
                    edge_list[j][2] = match
                    print(str(edge_list[j][0].src)+"_"+str(edge_list[j][0].dest)+" has increased confidence = "+ str(edge_list[j][1]))
                else:
                    edge_list[j][1] = edge_list[j][1] - 1
                    print(str(edge_list[j][0].src) + "_" + str(edge_list[j][0].dest) + " has decreased confidence = " + str(edge_list[j][1]))
                if edge_list[j][1] < (-1) * confidence_level:
                    print(str(edge_list[j][0].src)+"_"+str(edge_list[j][0].dest)+"deleted")
                    del edge_list[j]
                elif edge_list[j][1] > confidence_level or len(edge_list) == 1:
                    print("edge found")
                    break
                else:
                    j += 1
            if len(edge_list) == 1:
                print("edge found finally")
                print(str(edge_list[0][0].src)+"_"+str(edge_list[0][0].dest))
                source_node= edge_list[0][0].src
                print(source_node)

                for node in self.matched_nodes:
                    if node.name== source_node:
                        self.final_path.append(node)
                        self.matched_nodes=[]
                        self.final_path.append(edge_list[0])
                        print(self.final_path)
                break

                # match_next_node(nodes_list, query_video_frames1, last_frame_matched_with_edge)

    def print_final_path(self):
        print("path is: ")
        for element in self.final_path:
            if isinstance(element, Node):
                print(element.name)
                print()
            elif isinstance(element, Edge):
                print(str(element.src)+"_"+str(element.dest))
                print()
            else:
                raise Exception("Path not right")





# graph=Graph()
# graph.add_floor_map(0, "graph/images/map0.jpg")
# graph.mark_nodes(0)
# graph.make_connections(0)
# graph.read_nodes("testData/Morning_sit/nodes",4)
# graph.read_edges("testData/Morning_sit/edges",4)
# graph.print_graph(0)
# graph.save_graph()


query_video_frames1 = vo2.read_images("query_distinct_frame")
# graph1=graph.load_graph()
graph =load_graph()
node_and_image_matching_obj = node_and_image_matching()
node_and_image_matching_obj.locate_node(graph.Nodes, query_video_frames1)
node_and_image_matching_obj.locate_edge(graph.Nodes, query_video_frames1)
node_and_image_matching_obj.print_final_path()
# graph.print_graph(0)
