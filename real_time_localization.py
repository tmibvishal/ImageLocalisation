import cv2
import numpy as np
import os
import shutil

import socket
import sys

# IP = "10.194.35.37"
# s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# try:
#     s.bind((IP, 1234))
# except socket.error as err:
#     print("Bind failed, Error Code" + str(err.args[0]) + ", message: " + err.args[1])
#     sys.exit()
# s.listen(5)

import video_operations_3 as vo2
import matcher as mt
from graph2 import Graph, Edge, Node, FloorMap, load_graph
from video_operations_3 import ensure_path, DistinctFrames, ImgObj, save_to_memory, is_blurry_grayscale

query_video_distinct_frames = DistinctFrames()
query_video_ended = False
a = 2

class NodeEdgeRealTimeMatching:
    matched_path: list = []  # This will contain the list of confirmed paths
    nodes_matched: list = []
    # This will contain the list of nodes matched, at initial node at may have many nodes
    # but after confirming one edge it has the destination of that edge
    i_at_matched_node: int = 0
    # This is to have a record of i in query video - this i will contain i at matched node (nodes_matched[0])
    # Initial value is 0 when there can be multiple nodes_matched
    possible_edges = []
    # possible_edges contains all the possible edges that originate from nodes in nodes_matched
    # possible_edges is array of dictionary of type { "node: Node", "edge: Edge", "confidence: float",
    # "last_matched_i_with_j: int", "last_matched_j: int", "no_of_frames_to_match: int", "edge_ended_probability: int"}
    max_confidence = 0

    '''
    # ----- A bit useless feature - but added after Sushant request -----
    previous_i: int = 0
    continuously_no_matches_for_i: int = 0
    # If for a current value of i, suppose same i is being queried again and again in different possible edges,
    # thus we use continuously_no_matches_for_i to limit the same value of i to 3
    # although i will eventually increase for a particular possible_edge no_of_continuous_no_match reaches 3
    # note i is different moved for different possible_edge
    # ---------- @ this feature was added after Sushant request ----------
    '''

    def __init__(self, graph_obj: Graph):
        # some_query_img_objects = (query_video_distinct_frames.get_objects(0, 2))
        # img_objects_list contains 3 elements
        # nodes_matched = self.match_node_with_frames(some_query_img_objects, graph_obj)
        print("atleast started")
        # nodes_matched = []
        # self.nodes_matched.append(graph_obj.get_node(2))
        self.nodes_matched.append(graph_obj.get_node(0))
        # self.find_edge_with_nodes(0)
        return

    @staticmethod
    def match_node_with_frames(some_query_img_objects: list, graph_obj: Graph):
        search_list = graph_obj.Nodes
        node_confidence = []
        # node_confidence is list of (node.identity:int , confidence:int , total_fraction_matched:float)
        for node in search_list:
            for img_obj in some_query_img_objects:
                node_images: vo2.DistinctFrames = node.node_images
                if node_images is not None:
                    for data_obj in node_images.get_objects():
                        image_fraction_matched, min_good_matches = mt.SURF_returns(img_obj.get_elements(), data_obj.get_elements(),
                                                                 2500, 0.7)
                        if min_good_matches>100 and image_fraction_matched != -1:
                            if image_fraction_matched > 0.05 or min_good_matches>225:
                                print("Match found btw" + str(img_obj.get_time()) + " of query video and " + str(
                                    data_obj.get_time()) + " of node data")
                                if len(node_confidence) > 0 and node_confidence[-1][0] == node.identity:
                                    entry = node_confidence[-1]
                                    node_confidence[-1] = (node.identity, entry[1] + 1, entry[2] + image_fraction_matched)
                                    # print(str(node.identity) + " matched by " + str(image_fraction_matched))
                                else:
                                    node_confidence.append((node.identity, 1, image_fraction_matched))
        node_confidence = sorted(node_confidence, key=lambda x: (x[1], x[2]), reverse=True)
        print(node_confidence)
        final_node_list = []
        for entry in node_confidence:
            final_node_list.append(graph_obj.get_node(entry[0]))
        return final_node_list

    @staticmethod
    def match_edge_with_frame(possible_edge, i: int, query_video_ith_frame: vo2.ImgObj):
        # Match a possible edge object with query_video_ith_frame
        # possible edge here is passed as reference.


        j = possible_edge["last_matched_j"]
        # if last_matched_j is 3rd frame, now j will start matching from 4th frame,
        # gave better and more real time results

        # max value upto which j should be iterated
        jmax = possible_edge["last_matched_j"] + possible_edge["no_of_frames_to_match"]

        # let pe = possible_edge["edge"]
        # query_video_ith_frame will be matched with pe[j...max(jmax, maximum elements in pe)]
        # frame in pe[j...max(jmax, maximum elements in pe)] with maxmatch will be stored in case of multiple matches
        match, maxmatch = None, 0
        while j < jmax and j < possible_edge["edge"].distinct_frames.no_of_frames():
            # print(j)
            edge = possible_edge["edge"]
            img_obj_from_edge: vo2.ImgObj = edge.distinct_frames.get_object(j)
            image_fraction_matched, min_good_matches = mt.SURF_returns(img_obj_from_edge.get_elements(),
                                                     query_video_ith_frame.get_elements(), 2500, 0.7)
            # print("query i: ", i, ", jth frame of " + str(possible_edge["edge"].src) + "to" +
            # str(possible_edge["edge"].dest) + " :", j, image_fraction_matched)
            if image_fraction_matched!= -1:
                if image_fraction_matched > 0.09:
                    # print("query i: ", i, ", jth frame of " + str(possible_edge["edge"].src) + "to" +
                    #   str(possible_edge["edge"].dest) + " :", j, image_fraction_matched)
                    if image_fraction_matched > maxmatch:
                        match, maxmatch = j, image_fraction_matched
            j = j + 1
        if match is None:
            # no match is found in the j to jmax interval
            possible_edge["last_matched_i_with_j"] = i
            possible_edge["confidence"] = possible_edge["confidence"] - 0.5  # decreasing confidence
            possible_edge["no_of_continuous_no_match"] = possible_edge["no_of_continuous_no_match"] + 1
            # if possible_edge["no_of_frames_to_match"] < 5:
            possible_edge["no_of_frames_to_match"] = possible_edge["no_of_frames_to_match"] + 1
            # else:
            #     possible_edge["last_matched_j"] = possible_edge["last_matched_j"] + 1
            #     possible_edge["no_of_frames_to_match"] = 3
            if possible_edge["no_of_continuous_no_match"] >= 3:
                # handling the case if the query frame is just not matching
                # possible_edge["last_matched_i_with_j"] = possible_edge["last_matched_i_with_j"] + 1
                # restoring some confidence
                possible_edge["confidence"] = possible_edge["confidence"] + 1
                # also little restoration in no_of_frames_to_match
                # possible_edge["no_of_frames_to_match"] = possible_edge["no_of_frames_to_match"] - 1
                possible_edge["no_of_continuous_no_match"] = 1
        else:
            # match is found in the j to jmax interval
            img_obj_from_edge: vo2.ImgObj = edge.distinct_frames.get_object(match)
            print("popo query i: ", i, ", jth frame", match, img_obj_from_edge.time_stamp, maxmatch)
            possible_edge["last_matched_j"] = match
            possible_edge["last_matched_i_with_j"] = i
            possible_edge["confidence"] = possible_edge["confidence"] + 1
            if possible_edge["no_of_frames_to_match"] > 3:
                possible_edge["no_of_frames_to_match"] = possible_edge["no_of_frames_to_match"] - 1
            possible_edge["no_of_continuous_no_match"] = 0

        if j == possible_edge["edge"].distinct_frames.no_of_frames():
            # in this case the edge is being ended
            # ---- improvement required in this, for possible_edge with low no of distinct frames
            # j will end up reaching the end and without any match there will be a inc in edge_ended_probability ----
            possible_edge["edge_ended_probability"] = possible_edge["edge_ended_probability"] + 0.4


    @staticmethod
    def match_edge_end_frames_with_frame(possible_edge, i: int, query_video_ith_frame: vo2.ImgObj,
                                         no_of_edge_end_frames_to_consider: int = 2):
        edge: Edge = possible_edge["edge"]
        j_max = possible_edge["edge"].distinct_frames.no_of_frames()
        j = j_max - no_of_edge_end_frames_to_consider
        match, maxmatch = None, 0
        while j < 0:
            j += 1
        while j < j_max:
            img_obj_from_edge: vo2.ImgObj = edge.distinct_frames.get_object(j)
            image_fraction_matched, min_good_matches = mt.SURF_returns(img_obj_from_edge.get_elements(),
                                                                       query_video_ith_frame.get_elements(), 2500, 0.7)
            if image_fraction_matched != -1:
                if image_fraction_matched > 0.09:
                    if image_fraction_matched > maxmatch:
                        match, maxmatch = j, image_fraction_matched
            j = j + 1
        if match:
            print("edge end has matched")
            possible_edge["edge_ended_probability"] = possible_edge["edge_ended_probability"] + 0.5
            return True
        else:
            return False

    def find_edge_with_nodes(self):
        # for a nodes_matched, appending all the originating nodes in possible_edges
        if len(self.possible_edges) == 0:
            # only appending the nodes if the possible_edges is empty, otherwise
            # our also calls this function in between also even when possible_edges is already filed
            for node in self.nodes_matched:
                for edge in node.links:
                    if edge.distinct_frames is not None:
                        self.possible_edges.append({
                            "node": node,
                            "edge": edge,
                            "confidence": 0,
                            "last_matched_i_with_j": self.i_at_matched_node - 1,
                            "last_matched_j": 0,
                            "no_of_frames_to_match": 3,
                            "no_of_continuous_no_match": 0,
                            "edge_ended_probability": 0
                        })

        is_edge_found = False  # in the beginning is_edge_found is False
        is_edge_partially_found = False  # in the beginning is_edge_partially_found is False
        found_edge = None  # in the beginning found_edge is None
        i = self.i_at_matched_node
        # i_at_matched_node is 0 in the beginning in case of multiple nodes in nodes_matched
        # after one edge match it is i_at_matched_node as the name suggest

        # self.max_confidence keeps the track of the self.max_confidence of the current possible_edges
        j = 0
        while True and j < 100:
            if is_edge_found or is_edge_partially_found:
                # if is_edge_found or is_edge_partially_found then break the loop
                break
            for possible_edge in self.possible_edges:
                # if running for each possible_edge in possible_edges
                if possible_edge["confidence"] < self.max_confidence:
                    # breaking the loop if possible_edge["confidence"] is less than the self.max_confidence
                    continue
                # changing i for a particular possible_edge
                i = possible_edge["last_matched_i_with_j"] + 1
                '''
                # ----------------- A bit useless feature - but added after Sushant request -----------------
                if self.previous_i == i:
                    # if last queried i was same then inc continuously_no_matches_for_i
                    self.continuously_no_matches_for_i += 1

                else:
                    # else set continuously_no_matches_for_i to 0
                    self.continuously_no_matches_for_i = 0
                    self.previous_i = i

                if self.continuously_no_matches_for_i >= 3:
                    # if continuously_no_matches_for_i becomes large then move forward in a particular case
                    # so that same i is not queried a lot of times
                    i += 1
                    self.previous_i = i
                # - @ this feature is same as above in the class definition was added after Sushant request --
                '''

                if i >= query_video_distinct_frames.no_of_frames():
                    # if i for a query_video_distinct_frames has reached an end
                    if query_video_ended:
                        # it means query_video_ended has ended first, it must be is_edge_partially_found
                        # or it can also be full edge_ended but in that case
                        # also i am showing is_edge_partially_found with the last frame matched
                        is_edge_partially_found = True
                        found_edge = possible_edge
                        break
                    else:
                        # no of frames have reached an end needs more frames thus again ends and gets back to save_distinct_realtime_modified_ImgObj
                        return

                self.match_edge_end_frames_with_frame(possible_edge, i, query_video_distinct_frames.get_object(i),
                                                      no_of_edge_end_frames_to_consider=2)

                j_max = possible_edge["edge"].distinct_frames.no_of_frames()
                j_min_for_probability = j_max - 3
                while j_min_for_probability < 0:
                    j_min_for_probability += 1
                if possible_edge["edge_ended_probability"] >= 0.5 and possible_edge["last_matched_j"]>j_min_for_probability:
                    # if possible_edge["confidence"] > 0:
                    # edge is found
                    is_edge_found = True
                    found_edge = possible_edge
                    break
                # print("yo are travelling on" + str(possible_edge["edge"].src) + "to" + str(possible_edge["edge"].dest))
                j = possible_edge["last_matched_j"]
                # print("i:", i, "j:", j)
                last_jth_matched_img_obj = possible_edge["edge"].distinct_frames.get_object(j)
                time_stamp = last_jth_matched_img_obj.get_time()
                total_time = possible_edge["edge"].distinct_frames.get_time()
                fraction = time_stamp / total_time if total_time != 0 else 0
                graph_obj.on_edge(possible_edge["edge"].src, possible_edge["edge"].dest, fraction)
                graph_obj.display_path(0)

                query_video_ith_frame = query_video_distinct_frames.get_object(i)
                self.match_edge_with_frame(possible_edge, i, query_video_ith_frame)
                self.max_confidence = possible_edge["confidence"]
            j = j + 1
        print("go")
        if is_edge_found:
            # gaining the last pushed compass value
            # clientsocket, address = s.accept()
            # print(f"Connection from {address} has been extablished")
            # angle = clientsocket.recv(64)
            # print(int(angle))
            '''
            if found_edge["confidence"] < 1:
                for possible_edge in self.possible_edges:
                    if found_edge["edge"].src == possible_edge["edge"].src:
                        if found_edge["edge"].dest == possible_edge["edge"].dest:
                            # i am resetting this possible_edge
                            possible_edge["last_matched_j"] = 0
                            possible_edge["no_of_frames_to_match"] = 3
                            possible_edge["no_of_continuous_no_match"] = 0
                            possible_edge["edge_ended_probability"] = 0
            else:
            '''
            self.matched_path.append(found_edge)  # appending the found_edge to self.matched_path
            next_node_identity = found_edge["edge"].dest
            next_matched_nodes = []
            next_matched_nodes.append(graph_obj.get_node(next_node_identity))
            self.nodes_matched = next_matched_nodes  # setting self.nodes_matched
            print("confirmed: you crossed edge" + str(found_edge["edge"].src) + "_" + str(found_edge["edge"].dest))
            # next_matched_nodes will only contain one node which is the the nest node
            self.possible_edges = []  # setting self.possible_edges to empty list for next possible_edges query
            self.max_confidence = 0  # resetting self.max_confidence
            self.i_at_matched_node = i  # setting self.i_at_matched_node to i
            self.find_edge_with_nodes()  # recursively calling the function
            # i = found_edge["last_matched_i_with_j"] + 1
        elif is_edge_partially_found:
            # if is_edge_partially_found
            self.matched_path.append(found_edge)  # appending the found_edge to self.matched_path
            j = found_edge["last_matched_j"]
            last_jth_matched_img_obj = found_edge["edge"].distinct_frames.get_object(j)
            print("edge" + str(found_edge["edge"].src) + "_" + str(found_edge["edge"].dest))
            print("This edge is partially found upto " + str(last_jth_matched_img_obj.time_stamp))

    def print_path(self):
        for found_edge in self.matched_path:
            edge: Edge = found_edge["edge"]
            print("edge" + str(edge.src) + "_" + str(edge.dest))


graph_obj: Graph = load_graph("testData/night sit 0 june 18/graph obj vishal/new_objects/graph.pkl")
node_and_edge_real_time_matching = NodeEdgeRealTimeMatching(graph_obj)


def save_distinct_realtime_modified_ImgObj(video_str: str, folder: str, frames_skipped: int = 0,
                                           check_blurry: bool = True,
                                           hessian_threshold: int = 2500, ensure_min=True,
                                           livestream=False):
    ensure_path(folder + "/jpg")

    frames_skipped += 1

    if video_str == "webcam":
        video_str = 0
    cap = cv2.VideoCapture(video_str)
    # cap= cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)

    i = 0
    a = None
    b = None
    check_next_frame = False
    i_prev = 0  # the last i which was stored

    detector = cv2.xfeatures2d_SURF.create(hessian_threshold)

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    keypoints, descriptors = detector.detectAndCompute(gray, None)

    a = (len(keypoints), descriptors, vo2.serialize_keypoints(keypoints), gray.shape)
    img_obj = ImgObj(a[0], a[1], i, a[2], a[3])
    save_to_memory(img_obj, 'image' + str(i) + '.pkl', folder)
    cv2.imwrite(folder + '/jpg/image' + str(i) + '.jpg', gray)
    query_video_distinct_frames.add_img_obj(img_obj)
    node_and_edge_real_time_matching.find_edge_with_nodes()
    while True:
        if livestream:
            cap = cv2.VideoCapture(video_str)
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if i % frames_skipped != 0 and not check_next_frame:
                i = i + 1
                continue

            cv2.imshow('frame', gray)
            # print(i)
            if check_blurry:
                if is_blurry_grayscale(gray):
                    check_next_frame = True
                    # print("frame " + str(i) + " skipped as blurry")
                    i = i + 1
                    continue
                check_next_frame = False
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            if len(keypoints)<50:
                print("frame "+str(i)+ " skipped as "+str(len(keypoints))+" <50")
                i = i+1
                continue
            b = (len(keypoints), descriptors, vo2.serialize_keypoints(keypoints), gray.shape)
            image_fraction_matched, min_good_matches = mt.SURF_returns(a, b, 2500, 0.7, True)
            if image_fraction_matched == -1:
                check_next_frame = True
                i=i+1
                continue
            check_next_frame = False
            if 0< image_fraction_matched < 0.10 or min_good_matches<50 or (ensure_min and i - i_prev > 50):
                img_obj2 = ImgObj(b[0], b[1], i, b[2], b[3])
                save_to_memory(img_obj2, 'image' + str(i) + '.pkl', folder)
                cv2.imwrite(folder + '/jpg/image' + str(i) + '.jpg', gray)
                query_video_distinct_frames.add_img_obj(img_obj2)
                node_and_edge_real_time_matching.find_edge_with_nodes()
                a = b
                i_prev = i

            i = i + 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    print("released")
    cap.release()
    cv2.destroyAllWindows()
    global query_video_ended
    query_video_ended = True
    query_video_distinct_frames.calculate_time()
    return query_video_distinct_frames


def locate_new_edge_using_angle(initial_edge: Edge, graph_obj: Graph, angle_turned, allowed_angle_error: int = 20):
    located_edge = None
    for new_edge in initial_edge.angles:
        if angle_turned < new_edge[1] + allowed_angle_error and angle_turned > new_edge[1] - allowed_angle_error:
            print("new edge located is " + new_edge[0] + " as stored angle is " + str(
                new_edge[1]) + " and query angle is " + str(angle_turned))
            located_edge = new_edge[0]
            locations = located_edge.split("_")
            located_edge_obj = graph_obj.get_edge(locations[0], locations[1])
            return located_edge_obj
        else:
            print(new_edge[0] + " is not matched as stored angle is " + str(new_edge[1]) + " and query angle is " + str(
                angle_turned))

    return "no edge found"



if __name__ == '__main__':
    url = "http://10.194.36.234:8080/shot.jpg"
    # url = "http://10.194.36.234:8080/shot.jpg"
    # save_distinct_realtime_modified_ImgObj(url,
    #                                        "query_distinct_frame/night", 0,
    #                                        check_blurry=True, ensure_min=True, livestream=True)
    save_distinct_realtime_modified_ImgObj("testData/night sit 0 june 18/query video/VID_20190618_202826.webm","query_distinct_frame", 2,
                                           check_blurry=True, ensure_min=True, livestream=False)
