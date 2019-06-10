import cv2
import graph
import video_operations_2 as vo2
import matcher as mt
from operator import itemgetter


def god(video_str: str, check_blurry: bool, hessian_threshold: int = 2500):
    graphObj = graph.load_graph()
    node_confidence = []
    edge_confidence = []
    start_node = None
    end_node = None
    status_code = 0
    # 0 - Complete Mystery, 1 - On node, 2 - Previous node known but edge is mystery, 3 - On edge

    detector = cv2.xfeatures2d_SURF.create(hessian_threshold)
    cap = cv2.VideoCapture(video_str)

    i, i_prev = 0, -1
    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2GRAY)
            if check_blurry:
                if vo2.is_blurry_grayscale(gray):
                    i = i + 1
                    continue
            keypoints, descriptors = detector.detectAndCompute(gray, None)
            img_obj = vo2.ImgObj(len(keypoints), descriptors, i)

            query_manager(i_prev, i, img_obj, graphObj, node_confidence, edge_confidence, start_node,
                          end_node, status_code)

            # update gray with location

            cv2.imshow('Frame!', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        i = i + 1


def query_manager(i_prev, i_cur, img_obj, graphObj, node_confidence, edge_confidence, start_node,
                  end_node, status_code):
    del_i = i_cur - i_prev
    if status_code == 0:
        node_confidence, start_node = find_node(del_i, img_obj, graphObj, node_confidence)


def find_node(del_i, img_obj, graphObj, node_confidence, start_node=None):
    if len(node_confidence) == 0:
        search_list = graphObj.Nodes
    else:
        search_list = []
        for entry in node_confidence:
            nd = graphObj.get_node(entry[0])
            search_list.append(nd)

    for node in search_list:
        for data_obj in node.node_images.get_objects():
            image_fraction_matched = mt.SURF_match_2(img_obj.get_elements(), data_obj.get_elements(),
                                                     2500, 0.7)
            if image_fraction_matched > 0.2:
                if len(node_confidence) > 0 and node_confidence[-1][0] == node.identity:
                    entry = node_confidence[-1]
                    node_confidence[-1] = (node.identity, entry[1] + 1, entry[2] + image_fraction_matched)
                    print(str(node.identity) + " matched by " + str(image_fraction_matched))
                else:
                    node_confidence.append((node.identity, 1, image_fraction_matched))
    sorted(node_confidence, key=lambda x: (x[1], x[2]), reverse=True)

    if len(node_confidence) == 1:
        if node_confidence[0][1] >= 2 or (node_confidence[0][1] == 1 and del_i >= 20):
            start_node = node_confidence[0][0]
    elif len(node_confidence) > 1:
        if (node_confidence[0][1] - node_confidence[1][1]) >= 2 or ((
                                                                            node_confidence[0][1] - node_confidence[1][
                                                                        1]) == 1 and del_i >= 20) or (
                (node_confidence[0][2] - node_confidence[1][2]) >= 0.1 and del_i >= 40
        ):
            start_node = node_confidence[0][0]
    print("Node found at " + str(start_node))
    return node_confidence, start_node


graph1 = graph.load_graph()
graph1.print_graph(0)
