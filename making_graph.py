import video_operations_2 as vo
import path_graph
import os
import shutil


graph= path_graph.Graph()

def create_folder(directory):
    try:
        if not os.path.exists('./storage/'+directory+'/'):
            os.makedirs('./storage/'+directory+'/')
    except OSError:
        print('Error :creating directory' + directory)


def edge_from_specific_pt(i_init, j_init, frames1, frames2,edge, new_edge):
    """
    Called when frames1[i_init][1] matches best with frames2[j_init][1]. This function checks
    subsequent frames of frames1 and frames2 to see if edge is detected.

    Parameters
    ----------
    i_init: index of the frame in frames1 list , which matches with the corresponding frame
            in frame2 list
    j_init: index of the frame in frames2 list , which matches with the corresponding frame
            in frame1 list
    frames1:
    frames2: are lists containing tuples of the form
            (time_stamp, frame, len_keypoints, descriptors) along path1 and path2

    Returns
    -------
    ( status, i_last_matched, j_last_matched ),
        status: if edge is found or not (starting from i_init and j_init)
        i_last_matched: index of last matched frame of frames1
        j_last_matched: index of last matched frame of frames2

    """
    confidence = 5
    """
    No edge is declared when confidence is zero.

    Confidence is decreased by 1 whenever match is not found for (i)th frame of frame1 among 
    the next 4 frames after j_last_matched(inclusive)

    If match is found for (i)th frame, i_last_matched is changed to i, j_last_matched is changed to
    the corresponding match; and confidence is restored to initial value(5)
    """
    match, maxmatch, i, i_last_matched, j_last_matched = None, 0, i_init + 1, i_init, j_init
    """
    INV:
    i = index of current frame (in frames1) being checked for matches; i_last_matched<i<len(frames1)
    i_last_matched = index of last frame (in frames1 ) matched; i_init<=i_last_matched<len(frames1)
    j_last_matched = index of last frame (in frames2 ) matched(with i_last_matched);
                        j_init<=j_last_matched<len(frames2)
    match = index of best matched frame (in frames2) with (i)th frame in frames1. j_last_matched<=match<=j
    maxmatch = fraction matching between (i)th and (match) frames
    """
    while True:
        for j in range(j_last_matched + 1, j_last_matched + 5):
            if j >= len(frames2):
                break
            image_fraction_matched = mt.SURF_match_2((frames1[i][2], frames1[i][3]), (frames2[j][2], frames2[j][3]),
                                                     2500, 0.7)
            if image_fraction_matched > 0.15:
                if image_fraction_matched > maxmatch:
                    match, maxmatch = j, image_fraction_matched
        if match is None:
            confidence = confidence - 1
            if confidence == 0:
                break
        else:
            confidence = 5
            j_last_matched = match
            i_last_matched = i
        i = i + 1
        if i >= len(frames1):
            break
        match, maxmatch = None, 0

    if i_last_matched > i_init and j_last_matched > j_init:
        print("Edge found from :")
        print(str(frames1[i_init][0]) + "to" + str(frames1[i_last_matched][0]) + "of video 1")
        print(str(frames2[j_init][0]) + "to" + str(frames2[j_last_matched][0]) + "of video 2")
        if i_init ==0:
            print("working")
            create_folder(str(edge.src)+"_"+str(new_edge.src))
            graph.add_edge(edge.src, new_edge.src,str(edge.src)+"_"+str(new_edge.src) )
            graph.add_edge(new_edge.src, edge.dest, edge.edge_data)
            # graph.remove_edge(edge.src, edge.dest)
            # for l in range(i_init, i_last_matched+1):
            #     frames1[l]=(0,0)
            # for l in range(j_init, j_last_matched + 1):
            #     create_folder(folder+"_"+video_number)
            #     shutil.copy("storage/"+str(folder)+"/image"+str(frames2[l][0])+".jpg", "storage/"+str(folder)+"_"+video_number )
            #     os.remove("storage/"+str(folder)+"/image"+str(frames2[l][0])+".jpg")
            #     frames2[l] = (0, 0)
        return True, i_last_matched, j_last_matched
    else:
        return False, i_init, j_init


def compare_videos(frames1, frames2, edge, new_edge):
    """
    :param frames1:
    :param frames2: are lists containing tuples of the form (time_stamp, frame) along path1 and path2

    (i)th frame in frames1 is compared with all frames in frames2[lower_j ... (len2)-1].
    If match is found then edge_from_specific_pt is called from indexes i and match
    if edge found then i is incremented to i_last_matched (returned from edge_from_specific_pt) and
    lower_j is incremented to j_last_matched
    """

    len1, len2 = len(frames1), len(frames2)
    lower_j = 0
    i = 0
    while (i < len1):
        if (frames1[i]==0):
            continue
        else:
            match, maxmatch = None, 0
            for j in range(lower_j, len2):
                if (frames1[i] == 0):
                    continue
                else:
                    image_fraction_matched = mt.SURF_match_2((frames1[i][2], frames1[i][3]), (frames2[j][2], frames2[j][3]),
                                                             2500, 0.7)
                    if image_fraction_matched > 0.15:
                        if image_fraction_matched > maxmatch:
                            match, maxmatch = j, image_fraction_matched
            if match is not None:
                if i >= len1 or lower_j >= len2:
                    break
                status, i, j = edge_from_specific_pt(i, match, frames1, frames2, edge, new_edge)
                lower_j = j
            i = i + 1

def video_to_edges(video_number: str, path :str):
    if video_number=="1":
        graph.add_node()
        graph.add_node()
        shutil.copytree(path, "storage/0_1")
        graph.add_edge(0,1,"0_1")
    else:
        print(path)
        frames_new_edge = vo.read_images(path)
        no_of_nodes= graph.length
        graph.add_node()
        graph.add_node()
        graph_edges_copy =graph.edges[:]
        shutil.copytree(path, "storage/"+str(no_of_nodes+1)+"_"+str(no_of_nodes+2))
        new_edge=graph.add_edge(no_of_nodes+1, no_of_nodes+2, str(no_of_nodes+1)+"_"+str(no_of_nodes+2))
        # for folder in sorted(os.listdir("storage"))[:]:
        for edge in graph_edges_copy:
            folder = edge.edge_data
            frames_of_edges =vo.read_images('storage/'+folder)
            # frames_being_used_new_edge = frames_new_edge
            # compare_videos_and_print(frames_being_used, frames_of_edges)
            compare_videos(frames_of_edges,frames_new_edge, edge, new_edge)


video_to_edges("1", "v1_n")
video_to_edges("2", "v2_n")