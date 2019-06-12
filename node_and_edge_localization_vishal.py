import cv2
import numpy as np
import os
import shutil
import time

import video_operations_2 as vo2
import matcher as mt
from graph import Graph, Edge, Node, FloorMap, load_graph
from video_operations_2 import variance_of_laplacian, is_blurry_colorful, is_blurry_grayscale, ensure_path, save_to_memory, DistinctFrames



class NodeEdgeMatching:
    matched_path = []

    # matched is array of dictionary of type { "node: Node", "edge: Edge", "confidence: float",
    # "last_matched_i_with_j: int", "last_matched_j: int", "no_of_frames_to_match: int", "edge_ended: bool"}

    def __init__(self, graph_obj: Graph, query_video_distinct_frames: vo2.DistinctFrames):
        some_query_img_objects = (query_video_distinct_frames.get_objects(0, 2))
        # img_objects_list contains 3 elements
        nodes_matched = self.match_node_with_frames(some_query_img_objects, graph_obj)
        # nodes_matched = []
        # nodes_matched.append(graph_obj.get_node(0))
        # nodes_matched.append(graph_obj.get_node(6))
        self.find_edge_with_nodes(nodes_matched, query_video_distinct_frames, 0)
        return

    def match_node_with_frames(self, some_query_img_objects: list, graph_obj: Graph):
        """
        :param some_query_img_objects:
        :param graph_obj:

        :return:
        final_node_list = a list containing matched nodes in descending order of probability
        """
        search_list = graph_obj.Nodes
        node_confidence = []
        # node_confidence is list of (node.identity:int , confidence:int , total_fraction_matched:float)
        for node in search_list:
            for img_obj in some_query_img_objects:
                node_images: vo2.DistinctFrames = node.node_images
                if node_images is not None:
                    for data_obj in node_images.get_objects():
                        image_fraction_matched = mt.SURF_match_2(img_obj.get_elements(), data_obj.get_elements(),
                                                                 2500, 0.7)
                        if image_fraction_matched > 0.1:
                            print("Match found btw"+str(img_obj.get_time())+" of query video and "+str(data_obj.get_time())+" of node data")
                            if len(node_confidence) > 0 and node_confidence[-1][0] == node.identity:
                                entry = node_confidence[-1]
                                node_confidence[-1] = (node.identity, entry[1] + 1, entry[2] + image_fraction_matched)
                                # print(str(node.identity) + " matched by " + str(image_fraction_matched))
                            else:
                                node_confidence.append((node.identity, 1, image_fraction_matched))
        sorted(node_confidence, key=lambda x: (x[1], x[2]), reverse=True)
        print(node_confidence)
        final_node_list = []
        for entry in node_confidence:
            final_node_list.append(graph_obj.get_node(entry[0]))
        return final_node_list

    def match_edge_with_frame(self, possible_edge, i: int, query_video_ith_frame: vo2.ImgObj):
        # possible edge here is passed as reference

        j = possible_edge["last_matched_j"]
        jmax = possible_edge["last_matched_j"] + possible_edge["no_of_frames_to_match"]
        match, maxmatch = None, 0
        while j < jmax and j < possible_edge["edge"].distinct_frames.no_of_frames():
            print(j)
            edge = possible_edge["edge"]
            img_obj_from_edge: vo2.ImgObj = edge.distinct_frames.get_object(j)
            image_fraction_matched = mt.SURF_match_2(img_obj_from_edge.get_elements(), query_video_ith_frame.get_elements(), 2500, 0.7)
            print("query i: ", i, ", jth frame of " + str(possible_edge["edge"].src) + "to" + str(possible_edge["edge"].dest) + " :", j, image_fraction_matched)
            if image_fraction_matched > 0.15:
                if image_fraction_matched > maxmatch:
                    match, maxmatch = j, image_fraction_matched
            j = j + 1
        if match is None:
            print("not matched")
            possible_edge["last_matched_i_with_j"] = i
            possible_edge["confidence"] = possible_edge["confidence"] - 0.5
            possible_edge["no_of_frames_to_match"] = possible_edge["no_of_frames_to_match"] + 1
        else:
            print("popo query i: ", i, "jth frame of " + str(possible_edge["edge"].src) + "to" + str(possible_edge["edge"].dest) + " :", match, maxmatch)
            possible_edge["last_matched_j"] = match
            possible_edge["last_matched_i_with_j"] = i
            possible_edge["confidence"] = possible_edge["confidence"] + 1
            possible_edge["no_of_frames_to_match"] = possible_edge["no_of_frames_to_match"] - 1

        if j == possible_edge["edge"].distinct_frames.no_of_frames():
            print("edge",str(possible_edge["edge"].src) + "to" + str(possible_edge["edge"].dest),"ended")
            possible_edge["edge_ended"] = True

    def find_edge_with_nodes(self, nodes_matched: list, query_video_distinct_frames: vo2.DistinctFrames,
                             i_at_matched_node: int):
        possible_edges = []
        for node in nodes_matched:
            for edge in node.links:
                if edge.distinct_frames is not None:
                    possible_edges.append({
                        "node": node,
                        "edge": edge,
                        "confidence": 0,
                        "last_matched_i_with_j": i_at_matched_node - 1,
                        "last_matched_j": 0,
                        "no_of_frames_to_match": 3,
                        "edge_ended": False
                    })
        # i_at_matched_node is 0 means that there can be multiple nodes
        # because query video is just started querying for path matching
        is_edge_found = False
        is_edge_partially_found = False
        found_edge = None
        i = i_at_matched_node
        max_confidence = 0

        j=0
        while True and j<100:
            # print("so" + str(i))
            if is_edge_found or is_edge_partially_found:
                break
            for possible_edge in possible_edges:
                if possible_edge["confidence"] < max_confidence:
                    continue
                # print("ho")
                i = possible_edge["last_matched_i_with_j"] + 1
                if i >= query_video_distinct_frames.no_of_frames():
                    is_edge_partially_found = True
                    found_edge = possible_edge
                    break
                if possible_edge["edge_ended"]:
                    #if possible_edge["confidence"] > 0:
                    # edge is found
                    is_edge_found = True
                    found_edge = possible_edge
                    break
                query_video_ith_frame = query_video_distinct_frames.get_object(i)
                self.match_edge_with_frame(possible_edge, i, query_video_ith_frame)
                max_confidence = possible_edge["confidence"]
            j=j+1

        if is_edge_found:
            self.matched_path.append(found_edge)
            next_node_identity = found_edge["edge"].dest
            next_matched_nodes = []
            next_matched_nodes.append(graph_obj.get_node(next_node_identity))
            # next_matched_nodes will only contain one node which is the the nest node
            self.find_edge_with_nodes(next_matched_nodes, query_video_distinct_frames, i)
            # i = found_edge["last_matched_i_with_j"] + 1
        elif is_edge_partially_found:
            self.matched_path.append(found_edge)
            j = found_edge["last_matched_j"]
            last_jth_matched_img_obj = found_edge["edge"].distinct_frames.get_object(j)
            print("This edge is partially found upto " + str(last_jth_matched_img_obj.time_stamp))

    def print_path(self):
        for found_edge in self.matched_path:
            edge: Edge = found_edge["edge"]
            print("edge" + str(edge.src) + "_" + str(edge.dest))



class NodeEdgeRealtimeMatching:
    matched_path = []
    query_video_distinct_frames: vo2.DistinctFrames = vo2.DistinctFrames()
    query_ended: bool = False
    # matched is array of dictionary of type { "node: Node", "edge: Edge", "confidence: float",
    # "last_matched_i_with_j: int", "last_matched_j: int", "no_of_frames_to_match: int", "edge_ended: bool"}

    def __init__(self, graph_obj: Graph):
        some_query_img_objects = (self.query_video_distinct_frames.get_objects(0, 2))
        # img_objects_list contains 3 elements
        nodes_matched = self.match_node_with_frames(some_query_img_objects, graph_obj)
        # nodes_matched = []
        # nodes_matched.append(graph_obj.get_node(0))
        # nodes_matched.append(graph_obj.get_node(6))
        self.find_edge_with_nodes(nodes_matched, 0)
        return

    def match_node_with_frames(self, some_query_img_objects: list, graph_obj: Graph):
        """
        :param some_query_img_objects:
        :param graph_obj:

        :return:
        final_node_list = a list containing matched nodes in descending order of probability
        """
        search_list = graph_obj.Nodes
        node_confidence = []
        # node_confidence is list of (node.identity:int , confidence:int , total_fraction_matched:float)
        for node in search_list:
            for img_obj in some_query_img_objects:
                node_images: vo2.DistinctFrames = node.node_images
                if node_images is not None:
                    for data_obj in node_images.get_objects():
                        image_fraction_matched = mt.SURF_match_2(img_obj.get_elements(), data_obj.get_elements(),
                                                                 2500, 0.7)
                        if image_fraction_matched > 0.1:
                            print("Match found btw" + str(img_obj.get_time()) + " of query video and " + str(
                                data_obj.get_time()) + " of node data")
                            if len(node_confidence) > 0 and node_confidence[-1][0] == node.identity:
                                entry = node_confidence[-1]
                                node_confidence[-1] = (node.identity, entry[1] + 1, entry[2] + image_fraction_matched)
                                # print(str(node.identity) + " matched by " + str(image_fraction_matched))
                            else:
                                node_confidence.append((node.identity, 1, image_fraction_matched))
        sorted(node_confidence, key=lambda x: (x[1], x[2]), reverse=True)
        print(node_confidence)
        final_node_list = []
        for entry in node_confidence:
            final_node_list.append(graph_obj.get_node(entry[0]))
        return final_node_list

    def match_edge_with_frame(self, possible_edge, i: int, query_video_ith_frame: vo2.ImgObj):
        # possible edge here is passed as reference

        j = possible_edge["last_matched_j"]
        jmax = possible_edge["last_matched_j"] + possible_edge["no_of_frames_to_match"]
        match, maxmatch = None, 0
        while j < jmax and j < possible_edge["edge"].distinct_frames.no_of_frames():
            print(j)
            edge = possible_edge["edge"]
            img_obj_from_edge: vo2.ImgObj = edge.distinct_frames.get_object(j)
            image_fraction_matched = mt.SURF_match_2(img_obj_from_edge.get_elements(),
                                                     query_video_ith_frame.get_elements(), 2500, 0.7)
            print("query i: ", i,
                  ", jth frame of " + str(possible_edge["edge"].src) + "to" + str(possible_edge["edge"].dest) + " :", j,
                  image_fraction_matched)
            if image_fraction_matched > 0.15:
                if image_fraction_matched > maxmatch:
                    match, maxmatch = j, image_fraction_matched
            j = j + 1
        if match is None:
            print("not matched")
            possible_edge["last_matched_i_with_j"] = i
            possible_edge["confidence"] = possible_edge["confidence"] - 0.5
            possible_edge["no_of_frames_to_match"] = possible_edge["no_of_frames_to_match"] + 1
        else:
            print("popo query i: ", i,
                  "jth frame of " + str(possible_edge["edge"].src) + "to" + str(possible_edge["edge"].dest) + " :",
                  match, maxmatch)
            possible_edge["last_matched_j"] = match
            possible_edge["last_matched_i_with_j"] = i
            possible_edge["confidence"] = possible_edge["confidence"] + 1
            possible_edge["no_of_frames_to_match"] = possible_edge["no_of_frames_to_match"] - 1

        if j == possible_edge["edge"].distinct_frames.no_of_frames():
            print("edge", str(possible_edge["edge"].src) + "to" + str(possible_edge["edge"].dest), "ended")
            possible_edge["edge_ended"] = True

    def find_edge_with_nodes(self, nodes_matched: list, i_at_matched_node: int):
        possible_edges = []
        for node in nodes_matched:
            for edge in node.links:
                if edge.distinct_frames is not None:
                    possible_edges.append({
                        "node": node,
                        "edge": edge,
                        "confidence": 0,
                        "last_matched_i_with_j": i_at_matched_node - 1,
                        "last_matched_j": 0,
                        "no_of_frames_to_match": 3,
                        "edge_ended": False
                    })
        # i_at_matched_node is 0 means that there can be multiple nodes
        # because query video is just started querying for path matching
        is_edge_found = False
        is_edge_partially_found = False
        found_edge = None
        i = i_at_matched_node
        max_confidence = 0

        j = 0
        while True and j < 100:
            # print("so" + str(i))
            if is_edge_found or is_edge_partially_found:
                break
            for possible_edge in possible_edges:
                if possible_edge["confidence"] < max_confidence:
                    continue
                # print("ho")
                i = possible_edge["last_matched_i_with_j"] + 1
                if i >= self.query_video_distinct_frames.no_of_frames():
                    if self.query_ended:
                        is_edge_partially_found = True
                        found_edge = possible_edge
                        break
                    else:
                        # try to add a new frame now

                if possible_edge["edge_ended"]:
                    # if possible_edge["confidence"] > 0:
                    # edge is found
                    is_edge_found = True
                    found_edge = possible_edge
                    break
                query_video_ith_frame = self.query_video_distinct_frames.get_object(i)
                self.match_edge_with_frame(possible_edge, i, query_video_ith_frame)
                max_confidence = possible_edge["confidence"]
            j = j + 1

        if is_edge_found:
            self.matched_path.append(found_edge)
            next_node_identity = found_edge["edge"].dest
            next_matched_nodes = []
            next_matched_nodes.append(graph_obj.get_node(next_node_identity))
            # next_matched_nodes will only contain one node which is the the nest node
            self.find_edge_with_nodes(next_matched_nodes, self.query_video_distinct_frames, i)
            # i = found_edge["last_matched_i_with_j"] + 1
        elif is_edge_partially_found:
            self.matched_path.append(found_edge)
            j = found_edge["last_matched_j"]
            last_jth_matched_img_obj = found_edge["edge"].distinct_frames.get_object(j)
            print("This edge is partially found upto " + str(last_jth_matched_img_obj.time_stamp))

    def add_img_obj(self, img_obj):
        # exceptions are checked for in query_video_distinct_frames.add_img_obj itself
        self.query_video_distinct_frames.add_img_obj(img_obj)

    def print_path(self):
        for found_edge in self.matched_path:
            edge: Edge = found_edge["edge"]
            print("edge" + str(edge.src) + "_" + str(edge.dest))


    def save_distinct_ImgObj_modified_version(video_str, folder, frames_skipped: int = 0, check_blurry: bool = False,
                         hessian_threshold: int = 2500, ensure_min=False):
        ensure_path(folder + "/jpg")

        frames_skipped += 1

        if video_str == "webcam":
            video_str = 0
        cap = cv2.VideoCapture(video_str)
        # cap= cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 200)

        distinct_frames = DistinctFrames()
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

        a = (len(keypoints), descriptors)
        img_obj = ImgObj(a[0], a[1], i)
        save_to_memory(img_obj, 'image' + str(i) + '.pkl', folder)
        cv2.imwrite(folder + '/jpg/image' + str(i) + '.jpg', gray)
        distinct_frames.add_img_obj(img_obj)

        while True:
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
                        i = i + 1
                        continue
                    check_next_frame = False

                keypoints, descriptors = detector.detectAndCompute(gray, None)
                b = (len(keypoints), descriptors)
                image_fraction_matched = mt.SURF_match_2((a[0], a[1]), (b[0], b[1]), 2500, 0.7, False)
                if image_fraction_matched < 0.1 or (ensure_min and i - i_prev > 50):
                    img_obj2 = ImgObj(b[0], b[1], i)
                    save_to_memory(img_obj2, 'image' + str(i) + '.pkl', folder)
                    cv2.imwrite(folder + '/jpg/image' + str(i) + '.jpg', gray)
                    distinct_frames.add_img_obj(img_obj2)
                    a = b
                    i_prev = i

                i = i + 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
        distinct_frames.calculate_time()
        return distinct_frames


graph_obj: Graph = load_graph()
# query_video_frames1 = vo2.save_distinct_ImgObj("testData/query videos/VID_20190610_203834.webm", "query_distinct_frame/case1", 1, True, ensure_min=True)
# query_video_frames1 = vo2.save_distinct_ImgObj("testData/query videos/VID_20190610_204018.webm", "query_distinct_frame/case2", 1, True, ensure_min=True)
# query_video_frames1 = vo2.save_distinct_ImgObj("testData/query videos/VID_20190610_204056.webm", "query_distinct_frame/case3", 1, True, ensure_min=True)
query_video_frames1 = vo2.read_images("query_distinct_frame/case3")
node_edge_matching_obj = NodeEdgeMatching(graph_obj, query_video_frames1)
node_edge_matching_obj.print_path()
