import cv2
import numpy as np
import os
import shutil

import video_operations_3 as vo2
import matcher as mt
from graph import Graph, Edge, Node, FloorMap, load_graph
from video_operations_3 import ensure_path, DistinctFrames, ImgObj, save_to_memory, is_blurry_grayscale

query_video_distinct_frames = DistinctFrames()
query_video_ended = False


class NodeEdgeRealTimeMatching:
    matched_path: list = []
    nodes_matched: list = []
    i_at_matched_node: int = 0
    possible_edges = []
    previous_i: int = 0
    continuously_no_matches_for_i: int = 0
    # matched is array of dictionary of type { "node: Node", "edge: Edge", "confidence: float",
    # "last_matched_i_with_j: int", "last_matched_j: int", "no_of_frames_to_match: int", "edge_ended: bool"}

    def __init__(self, graph_obj: Graph):
        # some_query_img_objects = (query_video_distinct_frames.get_objects(0, 2))
        # img_objects_list contains 3 elements
        # nodes_matched = self.match_node_with_frames(some_query_img_objects, graph_obj)
        print("atleast started")
        # nodes_matched = []
        # self.nodes_matched.append(graph_obj.get_node(2))
        self.nodes_matched.append(graph_obj.get_node(2))
        # self.find_edge_with_nodes(0)
        return

    @staticmethod
    def match_node_with_frames(some_query_img_objects: list, graph_obj: Graph):
        """
        :param some_query_img_objects:
        :param graph_obj:
        :return:
        """
        search_list = graph_obj.Nodes
        node_confidence = []
        # node_confidence is list of (node.identity:int , confidence:int , total_fraction_matched:float)
        for node in search_list:
            for img_obj in some_query_img_objects:
                node_images: vo2.DistinctFrames = node.node_images
                if node_images is not None:
                    for data_obj in node_images.get_objects():
                        image_fraction_matched = mt.SURF_returns(img_obj.get_elements(), data_obj.get_elements(),
                                                                 2500, 0.7)
                        if image_fraction_matched > 0.05:
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
        """Match a possible edge object with query_video_ith_frame

                possible edge here is passed as reference.

                Parameters
                ----------
                possible_edge : possible edge object,
                i: ,
                query_video_ith_frame: ith frame of query video distinct frames,
        """
        j = possible_edge[
                "last_matched_j"] + 1
        # if last_matched_j is 3rd frame, now I will start matching from 4th frame,
        # gave better and more real time results
        jmax = possible_edge["last_matched_j"] + possible_edge[
            "no_of_frames_to_match"]  # j should iterate upto this value
        match, maxmatch = None, 0
        while j < jmax and j < possible_edge["edge"].distinct_frames.no_of_frames():
            # print(j)
            edge = possible_edge["edge"]
            img_obj_from_edge: vo2.ImgObj = edge.distinct_frames.get_object(j)
            image_fraction_matched = mt.SURF_returns(img_obj_from_edge.get_elements(),
                                                     query_video_ith_frame.get_elements(), 2500, 0.7)
            # print("query i: ", i, ", jth frame of " + str(possible_edge["edge"].src) + "to" +
            # str(possible_edge["edge"].dest) + " :", j, image_fraction_matched)
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
            if possible_edge["no_of_frames_to_match"] < 5:
                possible_edge["no_of_frames_to_match"] = possible_edge["no_of_frames_to_match"] + 1
            if possible_edge["no_of_continuous_no_match"] == 3:
                # handling the case if the query frame is just not matching
                possible_edge["last_matched_i_with_j"] = possible_edge["last_matched_i_with_j"] + 1
                # restoring some confidence
                possible_edge["confidence"] = possible_edge["confidence"] + 1
                # also little restoration in no_of_frames_to_match
                possible_edge["no_of_frames_to_match"] = possible_edge["no_of_frames_to_match"] - 1
        else:
            img_obj_from_edge: vo2.ImgObj = edge.distinct_frames.get_object(match)
            print("popo query i: ", i, ", jth frame", match, img_obj_from_edge.time_stamp, maxmatch)
            possible_edge["last_matched_j"] = match
            possible_edge["last_matched_i_with_j"] = i
            possible_edge["confidence"] = possible_edge["confidence"] + 1
            if possible_edge["no_of_frames_to_match"] > 2:
                possible_edge["no_of_frames_to_match"] = possible_edge["no_of_frames_to_match"] - 1
            possible_edge["no_of_continuous_no_match"] = 0

        if j == possible_edge["edge"].distinct_frames.no_of_frames():
            possible_edge["edge_ended"] = True

    def find_edge_with_nodes(self):
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
                        "edge_ended": False
                    })
        # i_at_matched_node is 0 means that there can be multiple nodes
        # because query video is just started querying for path matching
        is_edge_found = False
        is_edge_partially_found = False
        found_edge = None
        i = self.i_at_matched_node
        max_confidence = 0
        # print("po")
        j = 0
        while True and j < 100:
            # print("so" + str(i))
            if is_edge_found or is_edge_partially_found:
                break
            for possible_edge in self.possible_edges:
                if possible_edge["confidence"] < max_confidence:
                    continue
                # print("ho")
                i = possible_edge["last_matched_i_with_j"] + 1
                if self.previous_i == i :
                    self.continuously_no_matches_for_i += 1

                else:
                    self.continuously_no_matches_for_i = 0
                    self.previous_i = i

                if self.continuously_no_matches_for_i >= 3:
                    i += 1
                    self.previous_i = i

                if i >= query_video_distinct_frames.no_of_frames():
                    if query_video_ended:
                        is_edge_partially_found = True
                        found_edge = possible_edge
                        break
                    else:
                        return
                if possible_edge["edge_ended"]:
                    # if possible_edge["confidence"] > 0:
                    # edge is found
                    is_edge_found = True
                    found_edge = possible_edge
                    break
                print("yo are travelling on" + str(possible_edge["edge"].src) + "to" + str(possible_edge["edge"].dest))
                j = possible_edge["last_matched_j"]
                print("i:", i, "j:", j)
                last_jth_matched_img_obj = possible_edge["edge"].distinct_frames.get_object(j)
                time_stamp = last_jth_matched_img_obj.get_time()
                total_time = possible_edge["edge"].distinct_frames.get_time()
                fraction = time_stamp / total_time if total_time != 0 else 0
                graph_obj.on_edge(possible_edge["edge"].src, possible_edge["edge"].dest, fraction)
                graph_obj.display_path(0)

                query_video_ith_frame = query_video_distinct_frames.get_object(i)
                self.match_edge_with_frame(possible_edge, i, query_video_ith_frame)
                max_confidence = possible_edge["confidence"]
            j = j + 1
        print("go")
        if is_edge_found:
            if found_edge["confidence"] < 1:
                for possible_edge in self.possible_edges:
                    if found_edge["edge"].src == possible_edge["edge"].src:
                        if found_edge["edge"].dest == possible_edge["edge"].dest:
                            # i am restting this possible_edge
                            possible_edge["last_matched_j"] = 0
                            possible_edge["no_of_frames_to_match"] = 3
                            possible_edge["no_of_continuous_no_match"] = 0
                            possible_edge["edge_ended"] = False
            else:
                self.matched_path.append(found_edge)
                next_node_identity = found_edge["edge"].dest
                next_matched_nodes = []
                next_matched_nodes.append(graph_obj.get_node(next_node_identity))
                self.nodes_matched = next_matched_nodes
                print("confirmed: you crossed edge" + str(found_edge["edge"].src) + "_" + str(found_edge["edge"].dest))
                # next_matched_nodes will only contain one node which is the the nest node
                self.possible_edges = []
                self.i_at_matched_node = i
                self.find_edge_with_nodes()
                # i = found_edge["last_matched_i_with_j"] + 1
        elif is_edge_partially_found:
            self.matched_path.append(found_edge)
            j = found_edge["last_matched_j"]
            last_jth_matched_img_obj = found_edge["edge"].distinct_frames.get_object(j)
            print("edge" + str(found_edge["edge"].src) + "_" + str(found_edge["edge"].dest))
            print("This edge is partially found upto " + str(last_jth_matched_img_obj.time_stamp))

    def print_path(self):
        for found_edge in self.matched_path:
            edge: Edge = found_edge["edge"]
            print("edge" + str(edge.src) + "_" + str(edge.dest))


graph_obj: Graph = load_graph()
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
        else:
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
            b = (len(keypoints), descriptors, vo2.serialize_keypoints(keypoints), gray.shape)
            image_fraction_matched = mt.SURF_returns(a, b, 2500, 0.7, True)
            if image_fraction_matched < 0.09 or (ensure_min and i - i_prev > 50):
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
    query_video_ended = True
    query_video_distinct_frames.calculate_time()
    return query_video_distinct_frames


if __name__ == '__main__':
    url = "http://192.168.43.1:8080/shot.jpg"
    save_distinct_realtime_modified_ImgObj("testData/night sit 0 june 18/query video/VID_20190618_202916.webm",
                                           "query_distinct_frame/night", 4,
                                           check_blurry=True, ensure_min=True, livestream=False)
