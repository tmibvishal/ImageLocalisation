import cv2
import numpy as np

graph_frame=None
query_video_frame= None

def run_query_frame(img):
    global query_video_frame
    query_video_frame=img
    return show_frames()

def run_graph_frame(img):
    global graph_frame
    graph_frame=img
    show_frames()

def show_frames ():
    global graph_frame
    global query_video_frame

    if graph_frame is not None and query_video_frame is not None:
        graph_frame1= cv2.resize(graph_frame, (450,550),interpolation = cv2.INTER_AREA)
        grey_3_channel = cv2.cvtColor(query_video_frame, cv2.COLOR_GRAY2BGR)
        grey_3_channel1 = cv2.resize(grey_3_channel, (800, 550), interpolation=cv2.INTER_AREA)
        numpy_horizontal_concat = np.concatenate((graph_frame1, grey_3_channel1), axis=1)
        cv2.imshow('Live Stream and localization', numpy_horizontal_concat)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            return True
        else:
            return False


