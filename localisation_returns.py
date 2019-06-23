import os
import shutil
import general
import cv2
import video_operations_3 as vo
from graph2 import Graph, Edge, Node
import matcher as mt

video_path = "Query video path"

folder = "folder path to save distinct frames"
if os.path.exists(folder):
    print('---INPUT REQD----" ' + folder + " \"alongwith its contents will be deleted. Continue? (y/n)")
    if input() == "y":
        shutil.rmtree(folder)


class FoundMatch:
    def __init__(self, query_index, edge_index, edge_name, fraction_matched):
        self.query_index = query_index
        self.edge_index = edge_index
        self.edge_name = edge_name
        self.fraction_matched = fraction_matched


class PossibleEdge:
    def __init__(self, edge: Edge, confidence=0, ):
        self.name = edge.name
        self.edge = edge
        self.no_of_frames = len(edge.distinct_frames)
        self.matches_found = []  # list of FoundMatch objects
        self.indexes_matched = []
        self.confidence = confidence
        self.to_match_params = (0, min(self.no_of_frames, 3))

        def get_frame_params(self, frame_index):
            return edge.distinct_frames.get_object(frame_index).get_elements()


class RealTimeMatching:
    def __init__(self, graph_obj: Graph, ):
        self.confirmed_path = []
        self.probable_path = None
        self.possible_edges = []
        self.graph_obj = graph_obj
        self.query_objects = vo.DistinctFrames()

    def get_query_params(self, frame_index):
        return self.query_objects.get_object(frame_index).get_elements()

    def match_edges(self, query_index):
        # Assume all possible edge objects are there in possible_edges
        progress = False
        for possible_edge in self.possible_edges:
            for j in range(possible_edge.to_match_params[0], possible_edge.to_match_params[1]):
                fraction_matched = mt.SURF_returns(possible_edge.get_frame_params(j),
                    self.get_query_params(query_index))
                if fraction_matched>0.1:
                    foundMatch = FoundMatch(query_index,j, possible_edge.name, fraction_matched)
                    possible_edge.matches_found.append(foundMatch)
                    if j not in possible_edge.indexes_matched:
                        possible_edge.indexes_matched.append(j)
                    progress = True
        return progress

    def handle_edges(self):
        if len(self.possible_edges) == 0:
            if type(self.confirmed_path[-1])==int:
                identity = self.confirmed_path[-1]
                nd = self.graph_obj.get_node(identity)
                if nd is not None:
                    for edge in nd.links:
                        possible_edge = PossibleEdge(edge)
                        self.possible_edges.append(possible_edge)
            elif type(self.confirmed_path[-1])==tuple:
                src, dest = self.confirmed_path[-1][0], self.confirmed_path[-1][1]
                edge = self.graph_obj.get_edge(src, dest)
                possible_edge = PossibleEdge(edge)
                self.possible_edges.append(possible_edge)
                nd = self.graph_obj.get_node(dest)
                if nd is not None:
                    for edge in nd.links:
                        possible_edge = PossibleEdge(edge)
                        self.possible_edges.append(possible_edge)
        query_index = self.query_objects.no_of_frames() - 1
        progress = self.match_edges(query_index)
        while not progress and len(self.possible_edges)>0:
            for possible_edge in self.possible_edges:
                if possible_edge.to_match_params[1] == possible_edge.no_of_frames:
                    self.possible_edges.remove(possible_edge)
                    continue
                possible_edge.to_match_params[0] += 3
                possible_edge.to_match_params[1] +=3
                if possible_edge.to_match_params[1] > possible_edge.no_of_frames:
                    possible_edge.to_match_params[1] = possible_edge.no_of_frames
                if possible_edge.to_match_params[0]>=possible_edge.no_of_frames:
                    raise Exception("Wrong params for checking")
                progress = self.match_edges(query_index)
        if not progress:
            return
        # if progress found!!
        self.possible_edges.sort(key=lambda x: (len(x.indexes_matched), len(x.matches_found)), reverse=True)
        if self.probable_path != self.possible_edges.e
            self.[-1]



    def save_query_objects(self, video_path, folder="query_distinct_frame", livestream=False, write_to_disk=False):

        hessian_threshold = 2500

        if os.path.exists(folder):
            print('---INPUT REQD----" ' + folder + " \"alongwith its contents will be deleted. Continue? (y/n)")
            if input() == "y":
                shutil.rmtree(folder)
        general.ensure_path(folder + '/jpg')

        detector = cv2.xfeatures2d_SURF.create(hessian_threshold)

        cap = cv2.VideoCapture(video_path)
        i = 0
        while True:
            if livestream:
                cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if vo.is_blurry_grayscale(gray):
                continue

            cv2.imshow('Query Video!!', gray)
            keypoints, descriptors = detector.detectAndCompute(gray, None)

            a = (len(keypoints), descriptors, vo.serialize_keypoints(keypoints), gray.shape)
            img_obj = vo.ImgObj(a[0], a[1], i, a[2], a[3])

            self.query_objects.add_img_obj(img_obj)

            if write_to_disk:
                general.save_to_memory(img_obj, 'image' + str(i) + '.pkl', folder)
                cv2.imwrite(folder + '/jpg/image' + str(i) + '.jpg', gray)

            realTimeMatching.handle_edges()

            i = i + 1


realTimeMatching = RealTimeMatching()
