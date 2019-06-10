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

    def __init__(self):
        return

    def match_edge_with_frame(self, possible_edge, query_video_ith_frame: vo2.ImgObj):
        # possible edge here is passed as reference
        j = possible_edge["last_matched_j"]
        jmax = possible_edge["last_matched_j"] + possible_edge["no_of_frames_to_match"]
        while j < jmax and j < len(possible_edge["edge"]):
            match, maxmatch = None, 0
            edge = possible_edge["edge"]
            img_obj_from_edge = edge.distinct_frames.get_object(j)
            image_fraction_matched = mt.SURF_match_2(img_obj_from_edge, query_video_ith_frame, 2500, 0.7)
            if image_fraction_matched > 0.15:
                if image_fraction_matched > maxmatch:
                    match, maxmatch = j, image_fraction_matched
            if match is None:
                possible_edge["confidence"] = possible_edge["confidence"] - 1
                possible_edge["no_of_frames_to_match"] = possible_edge["no_of_frames_to_match"] + 1
            else:
                possible_edge["last_matched_j"] = match
            j = j + 1

    def find_edge_with_multiple_nodes(self, nodes_matched: list, query_video_distinct_frames: vo2.DistinctFrames):
        possible_edges = []
        for node in nodes_matched:
            for edge in node.links:
                possible_edges.append({
                    "node": node,
                    "edge": edge,
                    "confidence": 0,
                    "last_matched_j": -1,
                    "no_of_frames_to_match": 3
                })

        i = 0
        max_confidence = 0
        while i < query_video_distinct_frames.no_of_frames():
            j = 0
            for possible_edge in possible_edges:
                if (possible_edge["confidence"] <= max_confidence):
                    continue
                if max_confidence < possible_edge["confidence"]:
                    max_confidence = possible_edge["confidence"]
                self.match_edge_with_frame(possible_edge, query_video_distinct_frames.get_object(i))





