import cv2
import numpy as np
import os
import shutil
import time
from imutils import paths
import video_operations_2 as vo2
import matcher as mt
import graph
from graph import Graph


class NodeEdgeMatching:
    matched_path = []

    def __init__(self):
        return

    def match_node_with_frames(self, query_video_frames: list):
        "returns matched nodes using query_video_frames"
        return

    def match_edge_with_frame(self, possible_edge, i: int, query_video_ith_frame: vo2.ImgObj):
        # possible edge here is passed as reference
        j = possible_edge["last_matched_j"]
        jmax = possible_edge["last_matched_j"] + possible_edge["no_of_frames_to_match"]
        while j < jmax and j < possible_edge["edge"].distinct_frames.no_of_frames:
            match, maxmatch = None, 0
            edge = possible_edge["edge"]
            img_obj_from_edge = edge.distinct_frames.get_object(j)
            image_fraction_matched = mt.SURF_match_2(img_obj_from_edge, query_video_ith_frame, 2500, 0.7)
            if image_fraction_matched > 0.15:
                if image_fraction_matched > maxmatch:
                    match, maxmatch = j, image_fraction_matched
            j = j + 1
        if match is None:
            possible_edge["confidence"] = possible_edge["confidence"] - 1
            possible_edge["no_of_frames_to_match"] = possible_edge["no_of_frames_to_match"] + 1
        else:
            possible_edge["last_matched_j"] = match
            possible_edge["last_matched_i_with_j"] = i
        if j == (possible_edge["edge"].distinct_frames.no_of_frames - 1):
            possible_edge["edge_ended"] = True

    def find_edge_with_nodes(self, nodes_matched: list, query_video_distinct_frames: vo2.DistinctFrames,
                             i_at_matched_node: int):
        possible_edges = []
        for node in nodes_matched:
            for edge in node.links:
                possible_edges.append({
                    "node": node,
                    "edge": edge,
                    "confidence": 0,
                    "last_matched_i_with_j": i_at_matched_node - 1,
                    "last_matched_j": -1,
                    "no_of_frames_to_match": 3,
                    "edge_ended": False
                })

        # i_at_matched_node is 0 means that there can be multiple nodes because query video is just started querying for path matching
        is_edge_found = False
        is_edge_partially_found = False
        found_edge = None
        i = i_at_matched_node
        max_confidence = 0
        while True:
            if is_edge_found:
                break
            for possible_edge in possible_edges:
                if possible_edge["confidence"] < max_confidence:
                    continue
                i = possible_edge["last_matched_i_with_j"] + 1
                if i >= query_video_distinct_frames.no_of_frames():
                    is_edge_partially_found = True
                    found_edge = possible_edge
                    break
                if possible_edge["edge_ended"]:
                    # edge is found
                    is_edge_found = True
                    found_edge = possible_edge
                    break
                query_video_ith_frame = query_video_distinct_frames.get_object(i)
                self.match_edge_with_frame(possible_edge, i, query_video_ith_frame)
                max_confidence = possible_edge["confidence"]
        if is_edge_found:
            self.matched_path.append(found_edge)
            next_node_identity = found_edge["edge"].dest
            next_matched_nodes = []
            # next_matched_nodes will only contain one node which is the the nest node
            for Nd in nodes_matched:
                if Nd.identity == next_node_identity:
                    next_matched_nodes.append(Nd)
            self.find_edge_with_nodes(next_matched_nodes, query_video_distinct_frames, i)
            # i = found_edge["last_matched_i_with_j"] + 1
        elif is_edge_partially_found:
            self.matched_path.append(found_edge)
            j = found_edge["last_matched_j"]
            last_jth_matched_img_obj = found_edge["edge"].distinct_frames.get_object(j)
            print("This edge is partially found upto " + last_jth_matched_img_obj.time_stamp)