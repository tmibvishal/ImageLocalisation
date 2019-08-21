# ImageLocalisation

## Setup
```bash
pip3 install opencv-python==3.4.2.16
```

## How to use
1. Create database using 'graph2.py' ( Refer 'Creating database' section below)
2. Query video and track location using 'localisation_final.py' ( Refer 'Real Time Image based Path Tracking Algorithm' section below)

## Using Matcher (matcher.py)
### For comparing two images directly and getting fraction match
Read the two images using cv2.imread() and call the function 
```python
SURF_match (img1, img2, hessianThreshold: int = 400, ratio_thresh: float = 0.7, symmetry_match: bool = True)
```
### For comparing two images using the extracted descriptors and keypoints of images
```python
SURF_match (key_des_1, key_des_2, hessianThreshold: int = 400, ratio_thresh: float = 0.7, symmetry_match: bool = True)
where
key_des_1 : (length of keypoints, description) pair of image 1,
key_des_2 : (length of keypoints, description) pair of image 2,
```

## Creating database ( graph2.py )
#### 1. Set path of floor map image, path to save graph in method run()
```python
def run(code: int):
    if code == 0:
        graph = Graph()
        graph.add_floor_map(0, "graph/maps/map0.jpg") # <- put appropriate image path ( image of the floor map ) as the second parameter here
        graph.mark_nodes(0)
        graph.make_connections(0)
        graph.print_graph(0)
        graph.save_graph("new_objects", "graph.pkl") # <- put appropriate directory (where graph object is to be saved) as the first parameter here
```
#### 2. Call method run() with code 0
```python
run(0)
```
   1. A window 'Mark Nodes' should open up with the map image in it. In the 'Mark Nodes' window, mark the nodes by mouse clicks. Each node will show up as a circle with node identity written above it. When completed, press 'q' to continue to the next step.
   2. In the 'Make connections' window, make connections between nodes by putting cursor down at source node, dragging it till destination node and then lifting it up. Each connection will show up as an arrow with the edge name written above it. When done press 'q'.
   3. The final map will be printed in a new window. Press 'q' to exit.
   
#### 3. Saving node and edge data
   1. Record videos of all nodes and edges.
   2. In one folder (say "Node_data_path"), keep all the node videos with file names as <node_identity>.<extension> ( e.g. '1.mp4' ) 
   3. In a separate folder (say "Edge_data_path"), keep all the edge videos with file names as <source_node_identity>_<destination_node_identity>.<extension> ( e.g. "0_1.mkv" )
    
#### 4. Set node data and edge data path in method run()
```python
def run(code: int):
#some python code here
    if code == 2:
        graph: Graph = load_graph("new_objects/graph.pkl") # <- Set complete path of graph object created above here
        graph.read_nodes("Node_data_path", 4) # <- Set node data directory here
        graph.read_edges("Edge_data_path", 4) # <- Set edge data directory here
        graph.save_graph("new_objects", "graph.pkl") # <- Set the same path as step 1 here
```

#### 5. Call method run() with code 2
```python
run(2)
```
   You should see all the videos being played in a window in grayscale. On completion, the data has been saved and the graph object is now complete
   
## Real Time Image based Path Tracking Algorithm ( localisation_final.py )

### Structure
We implement two classes:
1. PossibleEdge
```python
class PossibleEdge:
"""
Attributes
__________
name : str
    name of the edge in format id1_id2 , where id1 and id2 are identities of source and destination node
edge : Edge (object)
    edge object of the corresponding edge
no_of_frames : int
    no of distinct frames stored in the edge 
to_match_params : tuple
    indicate range of the frames of the edge to be matched to the query frame
"""
```
2. RealTimeMatching
```python
class RealTimeMatching:
"""
Attributes
__________
confirmed_path : list
    first element contains identity of first node
probable_path : list
    contains PossibleEdge object of current edge
possible_edges : list
    It contains PossibleEdge objects of current edge and nearby edges
next_possible_edges : list
    possible_edges for the upcoming (next) query frame
graph_obj : Graph (object)
    Object containing whole database
query_objects : DistinctFrames (object)
    Contains all query frames
last_5_matches : list
    contains last 5 matches as (edge_index_matched, edge_name)
max_confidence_edges : int 
    no of edges with max confidence, i.e. those which are to be checked first
current_location_str : str
"""
```

### Steps to run
#### 1. Open the graph_obj
``` python
graph_obj: Graph = load_graph(graphPath)
```
#### 2. Create the empty RealTimeMatching object
``` python
realTimeMatching = RealTimeMatching(graph_obj)
```
#### 3. Call save_query_objects method of the RealTimeMatching object

   Method save_query_objects
```python
        def save_query_objects(self, video_path, folder="query_distinct_frame", livestream=False, write_to_disk=False,
                           frames_skipped=0):

        """
        Receives and reads query video, generates non-blurry gray image frames, creates ImgObj and
        updates query_objects
        :param video_path: The address of query video , can be a path or a url, str format
        :param folder: Path of folder to save query frames, str format
        :param livestream: bool, If True: then video_path is a url , If False: video_path is a path on disk
        :param write_to_disk: bool, If True, then query frames will be saved to specified folder in .pkl and .jpg formats
        :param frames_skipped: int, No of frames to be skipped in query video
        :return: None
        """
 ```
   - in case of live stream from a ip camera
       
       Set livestream to True and video_path to appropriate url
 ```python
    url = "http://192.168.43.1:8080/shot.jpg"
    folder_path_where_to_save_frames = "query_distinct_frame"
    no_of_frames_to_skip = 1
    realTimeMatching.save_query_objects(url, folder_path_where_to_save_frames, frames_skipped=no_of_frames_to_skip,livestream=True)
```
  - in case you query the video from local storage ( for testing )
       
       Set livestream to False and video_path to appropriate path on disk
```python
    query_video_local_url = "testData/MOV.MP4"
    folder_path_where_to_save_frames = "query_distinct_frame"
    no_of_frames_to_skip = 1
    realTimeMatching.save_query_objects(query_video_local_url, folder_path_where_to_save_frames, frames_skipped=no_of_frames_to_skip, livestream=False)
```
