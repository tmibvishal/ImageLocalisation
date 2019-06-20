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



### Goals
1. Improving and fixing bugs of chatbox
2. Adding the feature of sending multiple files and allowing receiver user to accept any number of files
3. Connecting the workflow of Anweshan and Arpit<br />
  **a. Anweshan:** Before Connecting the users<br />
  **b. Arpit:** After files are send by user1 and accepted by user2<br />
  
### Worflow

#### A. Chatbox
1. Socket.io is used in the chatbox
2. Socket Events:  
    - Chat:  
        - **user1** & **user2** connected. Message from **user1** to **user2**  
            - **user1** to **Server**      
            ```typescript
            "message" : {
                messageValue: "<message body>"  
            }
            ```  
            - **Server** to **user1** and **user2**  
            ```typescript
            "message" : { 
                username: "<username1>", 
                messageValue: "<message body>",  
                timeStamp: "<timestamp>"
            }  
            ```  
3. Chat messages are temporary and are not stored anywhere because it is assumed for now that connection is terminated on closing the tab


#### B. Accepting and Rejecting incoming FileList request
1. Suppose user1 sends the multiple files request to user2 using [FileList](https://developer.mozilla.org/en-US/docs/Web/API/FileList) object
2. Socket Events
    - FileList Send Request
        - **user1** & **user2** connected. FileList Send Request from **user1** to **user2**  
            - **user1** to **Server**      
            ```typescript
            "fileListSendRequest" : {
                fileList: "type: FileList"
            }
            ```  
            then user1 filesSendingState is set to "waiting"
            - **Server** to **user2**  
            ```typescript
            "fileListSendRequest" : {
                fileList: "type: FileList"  
            }
            ```  
3. Now based on what files are accepted by user2 
**Socket Events:**
    - File List Request Answer: 
        - **user2** to **Server**      
        ```typescript
        "fileListRequestAnswer" : {
            acceptedFilesAnswers: "type: boolean[]>"  
        }
        ```  
        then user1 and user2 filesSendingState is set to "sending" and "receiving" repectively if atleast 1 file is accepted otherwise user1 and user2 filesSendingState status becomes "idle" and no file is transfered
        - **Server** to **user1**  
        ```typescript
        "fileListRequestAnswer" : {
            acceptedFilesAnswers: "type: boolean[]>"  
        }
        ```   
3. acceptedFilesAnswers is an array with length(acceptedFilesAnswers) = length(fileList) that contains answer to individual files that were sent by user1. For files which has answer true, file sending process begins

### Links
- https://pad.devclub.in/p/filesend