# Real Time Image  based Path Tracking Algorithm

### User Interface
We have the following possible edge interface
```python
({
    "node": "type: Node | init: node",
    "edge": "type: Edge | init: edge",
    "confidence": "type: int | init: 0",
    "last_matched_i_with_j": "type: int | init: self.i_at_matched_node - 1",
    "last_matched_j": "type: int | init: 0",
    "no_of_frames_to_match": "type: int | init: 3",
    "no_of_continuous_no_match": "type: int | init: 0",
    "edge_ended": "type: bool | init: False"
})
```
We achieve this interface by declaring possible edge as dictionary

### Step 1
1. We open the graph_obj
``` python
graph_obj: Graph = load_graph()
```
2. Then we make the empty NodeEdgeRealTimeMatching object
```python
node_and_edge_real_time_matching = NodeEdgeRealTimeMatching(graph_obj)
```
3. After that we run save_distinct_realtime_modified_ImgObj
    - in case of live stream from a ip camera
    ```python
    url = "http://192.168.43.1:8080/shot.jpg"
    folder_path_where_to_save_frames = "query_distinct_frame"
    no_of_frames_to_skip = 4
    save_distinct_realtime_modified_ImgObj(url, folder_path_where_to_save_frames, no_of_frames_to_skip,check_blurry=False, ensure_min=True, livestream=True)
    ```
    - in case you query the video from local storage
    ```python
    query_video_local_url = "testData/MOV.MP4"
    folder_path_where_to_save_frames = "query_distinct_frame"
    no_of_frames_to_skip = 4
    save_distinct_realtime_modified_ImgObj(url, folder_path_where_to_save_frames, no_of_frames_to_skip,
                                               check_blurry=False, ensure_min=True, livestream=False)
    ```
