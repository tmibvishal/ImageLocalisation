import os
import shutil
import general
import cv2
import video_operations_3 as vo
from graph2 import Graph, Edge, Node, FloorMap
import matcher as mt


class PossibleEdge:
    def __init__(self, edge: Edge):
        self.name = edge.name
        self.edge = edge
        self.no_of_frames = edge.distinct_frames.no_of_frames()
        self.to_match_params = (0, self.no_of_frames)

    def __str__(self):
        return self.name

    def get_frame_params(self, frame_index):
        # return ( no_of_keypoints, descriptors, serialized_keypoints, shape ) of imgObj at frame_index
        return self.edge.distinct_frames.get_object(frame_index).get_elements()


class RealTimeMatching:
    def __init__(self, graph_obj: Graph, ):
        self.confirmed_path = []
        self.probable_path = None
        self.possible_edges = []
        self.next_possible_edges = []
        self.graph_obj = graph_obj
        self.query_objects = vo.DistinctFrames()
        self.last_5_matches = []
        self.max_confidence_edges = 0
        self.current_location_str = ""

    def get_query_params(self, frame_index):
        return self.query_objects.get_object(frame_index).get_elements()

    def match_edges(self, query_index):
        # Assume all possible edge objects are there in possible_edges
        progress = False
        match, maxmatch, maxedge = None, 0, None
        for i, possible_edge in enumerate(self.possible_edges):
            for j in range(possible_edge.to_match_params[0], possible_edge.to_match_params[1]):
                fraction_matched, features_matched = mt.SURF_returns(possible_edge.get_frame_params(j),
                                                                     self.get_query_params(query_index))
                if fraction_matched > 0.09 or features_matched > 200:
                    progress = True

                    if fraction_matched > maxmatch:
                        match, maxmatch, maxedge = j, fraction_matched, possible_edge.name

            if i == self.max_confidence_edges - 1 and match is not None:
                print("---Max match for " + str(query_index) + ": ", end="")
                print((match, maxedge))
                self.current_location_str = "---Max match for " + str(query_index) + ": (" + str(match) + " ," + str(
                    maxedge) + " )"
                self.last_5_matches.append((match, maxedge))
                if len(self.last_5_matches) > 5:
                    self.last_5_matches.remove(self.last_5_matches[0])
                return progress, match
        print("---Max match for " + str(query_index) + ": ", end="")
        print((match, maxedge))
        self.current_location_str = "---Max match for " + str(query_index) + ": (" + str(match) + " ," + str(
            maxedge) + " )"
        self.last_5_matches.append((match, maxedge))
        if len(self.last_5_matches) > 5:
            self.last_5_matches.remove(self.last_5_matches[0])
        return progress, match

    def handle_edges(self):
        if len(self.confirmed_path) == 0:
            for nd in self.graph_obj.Nodes[0]:
                for edge in nd.links:
                    possible_edge_node = PossibleEdge(edge)
                    possible_edge_node.to_match_params = (0, 1)
                    self.possible_edges.append(possible_edge_node)
            query_index = self.query_objects.no_of_frames() - 1
            progress, match = self.match_edges(query_index)
            if not progress or len(self.last_5_matches) < 2:
                return
            last_5_edges_matched = []
            for i in range(len(self.last_5_matches)):
                if self.last_5_matches[i][1] is not None:
                    last_5_edges_matched.append(self.last_5_matches[i][1])
            maxCount, most_occuring_edge, most_occuring_second = 0, None, None
            for edge in last_5_edges_matched:
                coun = last_5_edges_matched.count(edge)
                if coun > maxCount:
                    most_occuring_edge = edge
                    most_occuring_second = None
                elif coun == maxCount and edge != most_occuring_edge:
                    most_occuring_second = edge

            if most_occuring_edge is None or most_occuring_second is not None:
                return

            for possible_edge in self.possible_edges:
                if possible_edge.name == most_occuring_edge:
                    self.probable_path = possible_edge
                    self.probable_path.to_match_params = (0, possible_edge.no_of_frames)
                    self.max_confidence_edges = 1
                    src, dest = most_occuring_edge.split("_")
                    self.confirmed_path = [int(src)]
            self.next_possible_edges = [self.probable_path]
            nd = self.graph_obj.get_node(self.probable_path.edge.dest)

            for edge in nd.links:
                present = False
                for possible_edg in self.next_possible_edges:
                    if possible_edg.name == edge.name:
                        present = True
                        break
                if present: continue
                possibleEdge = PossibleEdge(edge)
                self.next_possible_edges.append(possibleEdge)
            nd = self.graph_obj.get_node(self.probable_path.edge.src)
            for edge in nd.links:
                if edge.dest == self.probable_path.edge.dest:
                    continue
                possibleEdge = PossibleEdge(edge)
                self.next_possible_edges.append(possibleEdge)

        if len(self.next_possible_edges) != 0:
            self.possible_edges = self.next_possible_edges

        elif len(self.possible_edges) == 0:
            if type(self.confirmed_path[-1]) == int:
                identity = self.confirmed_path[-1]
                nd = self.graph_obj.get_node(identity)
                if nd is not None:
                    for edge in nd.links:
                        possible_edge = PossibleEdge(edge)
                        self.possible_edges.append(possible_edge)

        query_index = self.query_objects.no_of_frames() - 1
        progress, match = self.match_edges(query_index)

        if not progress:
            return

        allow = True
        if (None, None) in self.last_5_matches:
            allow = False
            for i, match_tup in enumerate(self.last_5_matches):
                if match_tup is not (None, None):
                    counter = 0
                    for j in range(i + 1, 5):
                        if self.last_5_matches[j][1] is None:
                            continue
                        elif self.last_5_matches[j][1] == match_tup[1]:
                            counter += 1
                        else:
                            counter = -1
                            break
                    if counter == -1: break
                    if counter >= 2:
                        allow = True
                        break

        if len(self.last_5_matches) < 5 or not allow:
            self.next_possible_edges = self.possible_edges
            return
        last_5_edges_matched = []
        for i in range(len(self.last_5_matches)):
            if self.last_5_matches[i][1] is not None:
                last_5_edges_matched.append(self.last_5_matches[i][1])
        maxCount, most_occuring_edge, most_occuring_second = 0, None, None
        for edge in last_5_edges_matched:
            coun = last_5_edges_matched.count(edge)
            if coun > maxCount:
                most_occuring_edge = edge
                most_occuring_second = None
            elif coun == maxCount and edge != most_occuring_edge:
                most_occuring_second = edge

        if most_occuring_edge is None or most_occuring_second is not None:
            return
        for possible_edge in self.possible_edges:
            if possible_edge.name == most_occuring_edge:
                self.probable_path = possible_edge
                self.max_confidence_edges = 1

        edge_indexes = []
        for matches in self.last_5_matches:
            if matches[1] == most_occuring_edge:
                edge_indexes.append(matches[0])

        cur_edge_index = -1
        maxCount = 0
        for index in edge_indexes:
            coun = edge_indexes.count(index)
            if coun > maxCount or (coun == maxCount and index > cur_edge_index):
                cur_edge_index = index

        self.next_possible_edges = [self.probable_path]
        nd = self.graph_obj.get_node(self.probable_path.edge.dest)
        if cur_edge_index > self.probable_path.no_of_frames - 2:
            count_of_straight_edges, straightPossibleEdge = 0, None
            for tup in self.probable_path.edge.angles:
                if abs(tup[1]) < 20:
                    count_of_straight_edges += 1
                    src, dest = tup[0].split('_')
                    edg = self.graph_obj.get_edge(int(src), int(dest))
                    possible_edge = PossibleEdge(edg)
                    straightPossibleEdge = possible_edge
                    self.next_possible_edges.append(possible_edge)
                    self.max_confidence_edges += 1
            if count_of_straight_edges == 1:
                fraction_matched, features_matched = mt.SURF_returns(straightPossibleEdge.get_frame_params(0),
                                                                     self.get_query_params(query_index))
                if fraction_matched >= 0.1:  # 0.7 * self.probable_path.matches_found[-1].fraction_matched:
                    self.probable_path = straightPossibleEdge
                    cur_edge_index = 0
                    self.next_possible_edges = [self.probable_path]
                    nd = self.graph_obj.get_node(self.probable_path.edge.dest)

        for edge in nd.links:
            present = False
            for possible_edg in self.next_possible_edges:
                if possible_edg.name == edge.name:
                    present = True
                    break
            if present: continue
            possibleEdge = PossibleEdge(edge)
            self.next_possible_edges.append(possibleEdge)
        nd = self.graph_obj.get_node(self.probable_path.edge.src)
        for edge in nd.links:
            if edge.dest == self.probable_path.edge.dest:
                continue
            possibleEdge = PossibleEdge(edge)
            self.next_possible_edges.append(possibleEdge)

        last_jth_matched_img_obj = self.probable_path.edge.distinct_frames.get_object(cur_edge_index)
        time_stamp = last_jth_matched_img_obj.get_time()
        total_time = self.probable_path.edge.distinct_frames.get_time()
        fraction = time_stamp / total_time if total_time != 0 else 0
        self.graph_obj.on_edge(self.probable_path.edge.src, self.probable_path.edge.dest, fraction)
        self.graph_obj.display_path(0,self.current_location_str)
        return

    def save_query_objects(self, video_path, folder="query_distinct_frame", livestream=False, write_to_disk=False,
                           frames_skipped=0):

        frames_skipped += 1
        hessian_threshold = 2500

        if write_to_disk:
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

            if i % frames_skipped != 0:
                i = i + 1
                continue

            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if vo.is_blurry_grayscale(gray):
                continue

            cv2.imshow('Query Video!!', gray)
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            if len(keypoints) < 50:
                print("frame skipped as keypoints", len(keypoints), " less than 50")
                i = i + 1
                continue

            a = (len(keypoints), descriptors, vo.serialize_keypoints(keypoints), gray.shape)
            img_obj = vo.ImgObj(a[0], a[1], i, a[2], a[3])

            self.query_objects.add_img_obj(img_obj)

            if write_to_disk:
                general.save_to_memory(img_obj, 'image' + str(i) + '.pkl', folder)
                cv2.imwrite(folder + '/jpg/image' + str(i) + '.jpg', gray)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            self.handle_edges()

            i = i + 1

        cap.release()
        cv2.destroyAllWindows()


graph1: Graph = Graph.load_graph("new_objects/graph.pkl")
realTimeMatching = RealTimeMatching(graph1)
realTimeMatching.save_query_objects("testData/night sit 0 june 18/query video/VID_20190618_202826.webm",
                                    frames_skipped=1)
