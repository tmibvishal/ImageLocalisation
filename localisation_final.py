import os
import shutil
import general
import cv2
import video_operations_3 as vo
from graph2 import Graph, Edge, Node, FloorMap
import matcher as mt
import image_in_one_frame as one_frame


class PossibleEdge:
    def __init__(self, edge: Edge):
        self.name = edge.name
        self.edge = edge
        self.no_of_frames = edge.distinct_frames.no_of_frames()
        self.to_match_params = (0, self.no_of_frames) # Indexes to be queried in the edge

    def __str__(self):
        return self.name

    def get_frame_params(self, frame_index):
        """
        Returns params of particular imgObj of edge for SURF matching
        :param frame_index: Index of imgObj in the edge , int in range(0, no_of_frames)
        :return: tuple of the form
        ( no_of_keypoints, descriptors, serialized_keypoints, shape ) of imgObj at frame_index
        """
        return self.edge.distinct_frames.get_object(frame_index).get_elements()


class RealTimeMatching:
    def __init__(self, graph_obj: Graph, ):
        self.confirmed_path = [] # contains identity of first node
        self.probable_path = None # contains PossibleEdge object of current edge
        self.possible_edges = [] # list of PossibleEdge objects
        # It contains current 'src_dest' edge, edges with source as 'dest' and edges with destination as 'src'
        self.next_possible_edges = [] # possible_edges for the upcoming (next) query frame
        self.graph_obj = graph_obj
        self.query_objects = vo.DistinctFrames() # list, but can be changed later to just store the latest frame
        self.last_5_matches = [] # last 5 matches as (edge_index_matched, edge_name)
        self.max_confidence_edges = 0 # no of edges with max confidence, i.e. those which are to be checked first
        # generally it is equal to 1 and corresponds to the current edge (self.probable_path)
        # but can also be equal to 2 when the next edge is almost straight (<20 deg) from current edge and
        # the end of current edge is near
        # Also self.possible_edges is arranged in such a way that the edges corresponding to max confidence are appended
        # first in the list
        self.current_location_str = ""

    def get_query_params(self, frame_index):
        """
        Returns params of particular imgObj of query DistinctFrames object for SURF matching
        :param frame_index: Index of imgObj in the query object , int in range(0, no_of_frames)
        :return: tuple of the form
        ( no_of_keypoints, descriptors, serialized_keypoints, shape ) of imgObj at frame_index
        """
        return self.query_objects.get_object(frame_index).get_elements()

    def match_edges(self, query_index):
        """
        Finds matches of query frame with frames in possible edges and updates last 5 matches
        :param:
        query_index: current index (to be queried) of query frames
        :return:
        progress : bool -> if a match has been found or not
        """
        # Assume all possible edge objects are there in possible_edges
        progress = False
        match, maxmatch, maxedge = None, 0, None
        # These 3 variables correspond to the best match for the given query_index frame
        # match : edge_index (int), maxmatch: fraction_matched(float), maxedge: edge_name(str)
        for i, possible_edge in enumerate(self.possible_edges):
            for j in range(possible_edge.to_match_params[0], possible_edge.to_match_params[1]):
                fraction_matched, features_matched = mt.SURF_returns(possible_edge.get_frame_params(j),
                                                                     self.get_query_params(query_index))
                if fraction_matched > 0.09 or features_matched > 200:
                    progress = True

                    if fraction_matched > maxmatch:
                        match, maxmatch, maxedge = j, fraction_matched, possible_edge.name

            # First check best match in the max confidence edges. If yes, then no need to check others
            if i == self.max_confidence_edges - 1 and match is not None:
                print("---Max match for " + str(query_index) + ": ", end="")
                print((match, maxedge))
                if match is None:
                    self.current_location_str = "---Max match for " + str(query_index) + ": (None, None)"
                else:
                    self.current_location_str = "---Max match for " + str(query_index) + ": (" + str(
                        match) + " ," + str(
                        maxedge) + " )"
                self.graph_obj.display_path(0, self.current_location_str)
                # Update last_5_matches
                self.last_5_matches.append((match, maxedge))
                if len(self.last_5_matches) > 5:
                    self.last_5_matches.remove(self.last_5_matches[0])
                return progress

        print("---Max match for " + str(query_index) + ": ", end="")
        print((match, maxedge))
        if match is None:
            self.current_location_str = "---Max match for " + str(query_index) + ": (None, None)"
        else:
            self.current_location_str = "---Max match for " + str(query_index) + ": (" + str(match) + " ," + str(
                maxedge) + " )"
        self.graph_obj.display_path(0, self.current_location_str)
        # Update last_5_matches
        self.last_5_matches.append((match, maxedge))
        if len(self.last_5_matches) > 5:
            self.last_5_matches.remove(self.last_5_matches[0])
        return progress

    def handle_edges(self):
        """
        Updates possible_edges, next_possible_edges and
        decides most_occuring_edge and cur_edge_index ( which give the current location )
        based on last_5_matches
        :return: None
        """
        # if self.confirmed_path is empty then starting pt is not defined yet.
        if len(self.confirmed_path) == 0:

            # Append all edges in self.possible_edges with the to_match_params being only the first frame of each edge
            for nd in self.graph_obj.Nodes[0]:
                for edge in nd.links:
                    possible_edge_node = PossibleEdge(edge)
                    possible_edge_node.to_match_params = (0, 1) # <- Change this to include more frames of each edge
                                                                # in determination of initial node
                    self.possible_edges.append(possible_edge_node)

            # Pick up the last query index

            query_index = self.query_objects.no_of_frames() - 1
            progress = self.match_edges(query_index)

            # We need at least 2 matches to consider first node
            if not progress or len(self.last_5_matches) < 2: # <- Change this to set no of matches reqd for
                                                             # determination of first node
                return

            # To find the most occuring edge in last_5_matches
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

            # If most_occuring_second is not None it implies 2 edges are having max count
            if most_occuring_edge is None or most_occuring_second is not None:
                return

            # At this point we have the most occuring edge
            for possible_edge in self.possible_edges:
                if possible_edge.name == most_occuring_edge:

                    # Setting self.probable_path, self.confirmed_path
                    self.probable_path = possible_edge
                    self.probable_path.to_match_params = (0, possible_edge.no_of_frames)
                    self.max_confidence_edges = 1
                    src, dest = most_occuring_edge.split("_")
                    self.confirmed_path = [int(src)]

            # Setting self.next_possible_edges in this order:
            # 1. current edge
            # 2. nearby edges
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

        # If something is already there is self.next_possible_edges, use that
        elif len(self.next_possible_edges) != 0:
            self.possible_edges = self.next_possible_edges

        # Else use the node identity stored in self.confirmed_path
        # This should be deprecated i guess
        elif len(self.possible_edges) == 0:
            if type(self.confirmed_path[-1]) == int:
                identity = self.confirmed_path[-1]
                nd = self.graph_obj.get_node(identity)
                if nd is not None:
                    for edge in nd.links:
                        possible_edge = PossibleEdge(edge)
                        self.possible_edges.append(possible_edge)

        query_index = self.query_objects.no_of_frames() - 1
        progress = self.match_edges(query_index)

        if not progress:
            # print("err 0")
            return

        if len(self.last_5_matches) < 5:
            self.next_possible_edges = self.possible_edges
            # print("err 1")
            return

        # To find the most occuring edge in last_5_matches
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
                maxCount = coun
            elif coun == maxCount and edge != most_occuring_edge:
                most_occuring_second = edge

        # If most_occuring_second is not None it implies 2 edges are having max count
        if most_occuring_edge is None or most_occuring_second is not None:
            # print("err 2")
            return

        if (None,None) in self.last_5_matches and maxCount<3:
            # print("err 3")
            return

        # At this point we have the most occuring edge
        for possible_edge in self.possible_edges:
            if possible_edge.name == most_occuring_edge:
                # Setting self.probable_path
                self.probable_path = possible_edge
                self.max_confidence_edges = 1

        # Finding the most occuring edge index (in the last 5 matches) on the current edge
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
                maxCount = coun

        # cur_edge_index holds the most occuring edge index (in the last 5 matches) on the current edge

        # Setting self.next_possible_edges in this order:
        # 1. current edge
        # 2. Edge with src as dest of current edge , and with its angle being <20 deg deviated from current edge
        #        ( will be added only if cur_edge_index is the last index of current edge)
        # 3. Other nearby edges
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
            if count_of_straight_edges == 1:# Setting next_pos
                # If cur_edge_index is last index of current edge, and
                # If only one edge is straight ahead (angle < 20 deg) and its first frame matches, then the next edge
                # is set as self.probable_path (i.e., it is set as the current edge)
                fraction_matched, features_matched = mt.SURF_returns(straightPossibleEdge.get_frame_params(0),
                                                                     self.get_query_params(query_index))
                if fraction_matched >= 0.1:  # maybe changed to
                                             # 0.7 * self.probable_path.matches_found[-1].fraction_matched:
                                             # or something
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

        # Displaying current location on graph
        # print(str(most_occuring_edge)+", "+str(cur_edge_index))
        edgeObj, allow = None, True
        for nd in self.graph_obj.Nodes[0]:
            if not allow: break
            for edge in nd.links:
                if edge.name == most_occuring_edge:
                    edgeObj = edge
                    allow = False
                    break
        last_jth_matched_img_obj = edgeObj.distinct_frames.get_object(cur_edge_index)
        time_stamp = last_jth_matched_img_obj.get_time()
        total_time = edgeObj.distinct_frames.get_time()
        fraction = time_stamp / total_time if total_time != 0 else 0
        self.graph_obj.on_edge(edgeObj.src, edgeObj.dest, fraction)
        # print("graph called")
        self.graph_obj.display_path(0,self.current_location_str)
        return

    def save_query_objects(self, video_path, folder="query_distinct_frame", livestream=False, write_to_disk=False,
                           frames_skipped=0):

        """
        Receives and reads query video, generates non-blurry gray image frames, creates ImgObj and
        updates query_objects
        :param video_path: The address of query video , can be a path or a url, str format
        :param folder: Path of folder to save query frames, str format
        :param livestream: bool, If True: then video_path is a url , If False: video_path is a path on disk
        :param write_to_disk: bool, If True, then query frames will be saved
        to specified folder in .pkl and .jpg formats
        :param frames_skipped: int, No of frames to be skipped in query video
        :return: None
        """

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

            # cv2.imshow('Query Video!!', gray)
            break_video= one_frame.run_query_frame(gray)

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

            if (cv2.waitKey(1) & 0xFF == ord('q')) or break_video:
                break

            # Calling the localisation functions
            self.handle_edges()

            i = i + 1

        cap.release()
        cv2.destroyAllWindows()


graph1: Graph = Graph.load_graph("new_objects/graph.pkl")
realTimeMatching = RealTimeMatching(graph1)
url = "http://10.194.36.234:8080/shot.jpg"
realTimeMatching.save_query_objects(url, livestream=True,
                                    frames_skipped=0)
