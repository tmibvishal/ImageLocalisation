import graph as gp

# Deleting nodegraph.jpg ( for initial phase )
if os.path.exists("images/nodegraph1.jpg"):
    os.remove("images/nodegraph1.jpg")
if os.path.exists("images/nodegraph2.jpg"):
    os.remove("images/nodegraph2.jpg")

graph = gp.Graph()

no_of_floors = 2
for z in range(no_of_floors):
    graph.mark_nodes(z)
    graph.make_connections(z)
for z in range(no_of_floors):
    graph.print_graph(z)


# To add a floor,
# 1) change the no_of_floors variable ( = 2 if ground floor and 1st floor )
# 2) add a file strictly named "mapz.jpg" where z = floorNo ( z = 0 for ground floor )
# e.g. map0.jpg for ground floor with no_of_floors = 1



# To create nodes manually, use
# graph.create_node('entrance', 256, 256, 0, [1])
# graph.create_node('lift-grd', 50, 50, 0, [0])

# To check available events, use
# events = [i for i in dir(cv2) if 'EVENT' in i]
# print(events)

# To use a black image instead of map, use
# import numpy as np
# img = np.ones([512, 512, 3], np.uint8)
# instead of
# img = cv2.imread('map.jpg')